import os
import subprocess
import ase.io
import ase.db
import numpy as np
import torch
from pathlib import Path
import time
import shutil
import argparse
import sys
import datetime
import json

from utils import load_xpainn_model, prep_xpainn_input, get_connectivity, ConnectivityCompressor

def extract_nn_module_weights(model_path, output_path=None):
    model_path_str = str(model_path)
    loaded_content = torch.load(model_path, map_location='cpu')
    nn_module_weights = {}

    if isinstance(loaded_content, torch.nn.Module):
        nn_module_weights = loaded_content.state_dict()
        if not nn_module_weights:
            raise ValueError(f"Empty state_dict extracted from torch.nn.Module at {model_path_str}")

    elif isinstance(loaded_content, dict) and 'state_dict' in loaded_content:
        lightning_model_state_dict = loaded_content['state_dict']
        keys_found = 0
        for key, value in lightning_model_state_dict.items():
            if key.startswith("model."):
                new_key = key.split("model.", 1)[1]
                nn_module_weights[new_key] = value
                keys_found += 1
        if keys_found == 0:
            raise ValueError(f"No keys starting with 'model.' found in PyTorch Lightning checkpoint {model_path_str}")
            
    elif isinstance(loaded_content, dict):
        if any(key.startswith("model.") for key in loaded_content.keys()):
            keys_found = 0
            for key, value in loaded_content.items():
                if key.startswith("model."):
                    new_key = key.split("model.", 1)[1]
                    nn_module_weights[new_key] = value
                    keys_found += 1
            if keys_found == 0:
                 raise ValueError(f"Found 'model.' prefixed keys in {model_path_str} but extracted no weights after stripping")
        else:
            nn_module_weights = loaded_content
    else:
        raise TypeError(f"Unrecognized format for file {model_path_str}. Expected torch.nn.Module, PyTorch Lightning checkpoint, or state_dict")

    if not nn_module_weights:
        raise ValueError(f"Failed to extract valid neural network module weights from {model_path_str}")

    if output_path is None:
        p_model_path = Path(model_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = p_model_path.parent / f"{p_model_path.stem}_nn_module_weights_{timestamp}.pt"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(nn_module_weights, output_path)
    return str(output_path)

def run_command(command, shell=True, check=True, verbose=False):
    if verbose:
        print(f"Running: {command[:100]}...")
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['LC_ALL'] = 'C.UTF-8'
        env['LANG'] = 'C.UTF-8'
        
        result = subprocess.run(command, shell=shell, check=check, capture_output=True, text=True, 
                               encoding='utf-8', errors='replace', env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def iterative_biasing_workflow(args):
    number_of_loops = args.num_iterations
    mols_to_generate = args.mols_per_iter
    mols_to_predict_batch = 256
    min_mols_for_retrain = args.min_mols_for_retrain
    selection_threshold_e1 = args.e1_threshold
    k_std_factor = args.k_std_factor
    use_dynamic_threshold = args.use_dynamic_threshold
    retrain_epochs = args.retrain_epochs
    retrain_patience = 5
    bond_connection_cutoff = 1.85
    
    base_dir = Path(args.base_dir)
    initial_generator_dir = base_dir / args.initial_generator_dir
    xpainn_ckpt_path = base_dir / args.xpainn_model_path
    if args.results_suffix:
        results_suffix = args.results_suffix
    else:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_suffix = f"run_{current_time}"

    results_root = Path(args.results_dir) if args.results_dir else base_dir / "results"
    base_results_dir = results_root / f"iterative_results_{results_suffix}"
    gschnet_script_dir = base_dir / "schnetpack-gschnet/src/scripts"
    config_dir = base_dir / "ldhgeneration/configs"
    qm9_db_path = base_dir / "data/qm9.db"
    
    base_results_dir.mkdir(parents=True, exist_ok=True)


    required_paths = [
        (initial_generator_dir, "pretrained generator model directory"),
        (xpainn_ckpt_path, "XPaiNN model checkpoint"),
        (gschnet_script_dir / "gschnet_train", "gschnet_train script"),
        (gschnet_script_dir / "gschnet_generate", "gschnet_generate script"),
        (gschnet_script_dir / "check_validity.py", "check_validity.py script"),
        (config_dir, "config directory")
    ]
    
    for path, desc in required_paths:
        if not path.exists():
            print(f"Error: {desc} not found: {path}")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xpainn_model = load_xpainn_model(str(xpainn_ckpt_path), device)
    xpainn_model.eval()

    compressor = ConnectivityCompressor()

    current_generator_dir = initial_generator_dir
    last_e1_metric_value = float('inf')
    no_significant_change_counter = 0
    actual_completed_iterations = 0

    for i in range(1, number_of_loops + 1):
        iteration_dir = base_results_dir / f"iteration_{i}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        generator_to_use_for_gen = current_generator_dir
        generated_db_path = iteration_dir / "generated_molecules.db"
        filtered_db_path = iteration_dir / "filtered_molecules.db"
        selected_db_path = iteration_dir / "selected_for_retrain.db"

        if i > 1:
            last_selected_db_path = base_results_dir / f"iteration_{i-1}" / "selected_for_retrain.db"

            if not last_selected_db_path.is_file():
                print(f"Error: Training data not found: {last_selected_db_path}")
                break

            with ase.db.connect(last_selected_db_path) as db_conn:
                database_size = db_conn.count()

            if database_size < min_mols_for_retrain:
                pass
            else:
                safety_margin = 10
                effective_db_size = database_size - safety_margin
                val_ratio = 0.1 
                split_val = max(1, int(effective_db_size * val_ratio)) 
                split_train = effective_db_size - split_val            

                model_candidates = [
                    current_generator_dir / "best_model",
                    current_generator_dir / "checkpoints" / "last.ckpt"
                ]
                pretrained_model_file = next((f for f in model_candidates if f.is_file()), None)
                
                if pretrained_model_file is None:
                    print(f"Error: Model/checkpoint not found in {current_generator_dir}")
                    break

                model_output_dir = iteration_dir / "model_retrained"
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                weights_output_file = model_output_dir / "extracted_nn_module_weights.pt"
                extracted_weights_path = extract_nn_module_weights(pretrained_model_file, output_path=weights_output_file)

                current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                run_id_for_train = f"iter{i}_{current_timestamp}"

                retrain_cmd = [
                    f"{gschnet_script_dir}/gschnet_train",
                    f"--config-dir={config_dir}",
                    f"experiment=iterative_finetune",
                    f"data.datapath={last_selected_db_path}",
                    f"data.num_train={split_train}",
                    f"data.num_val={split_val}",
                    f"run.path={model_output_dir}",
                    f"run.id={run_id_for_train}",
                    f"trainer.max_epochs={retrain_epochs}",
                    f"++callbacks.early_stopping.patience={retrain_patience}",
                    f"++task.init_weights_path={extracted_weights_path}",
                    f"++globals.lr=1e-5"
                ] + ([f"++trainer.accelerator=gpu", f"++trainer.devices=1"] if device.type == 'cuda' else [])

                retrain_cmd_str = " ".join(retrain_cmd)
                if not run_command(retrain_cmd_str):
                    print(f"Training failed in iteration {i}")
                    break
                
                actual_trained_model_dir = model_output_dir / run_id_for_train
                current_generator_dir = actual_trained_model_dir if (actual_trained_model_dir / "best_model").exists() else model_output_dir

                generator_to_use_for_gen = current_generator_dir

        if generated_db_path.exists():
            generated_db_path.unlink()
            
        generate_cmd = [
            f"{gschnet_script_dir}/gschnet_generate",
            f"modeldir={generator_to_use_for_gen}",
            f"n_molecules={mols_to_generate}",
            f"outputfile={generated_db_path}",
            f"batch_size=100",
            f"max_n_atoms=35",
        ] + (["use_gpu=True"] if device.type == 'cuda' else [])

        generate_cmd_str = " ".join(generate_cmd)
        if not run_command(generate_cmd_str):
            print(f"Generation failed in iteration {i}")
            break

        if args.skip_filter:
            filtered_db_path = generated_db_path
        else:
            filter_cmd = [
                f"python {gschnet_script_dir}/check_validity.py",
                f"{generated_db_path}",
                f"--compute_uniqueness",
                f"--compare_db_path {qm9_db_path}",
            ]
            
            split_path = generator_to_use_for_gen / "split.npz"
            if split_path.exists():
                filter_cmd.append(f"--compare_db_split_path {split_path}")
            
            filter_cmd.extend([
                f"--ignore_enantiomers",
                f"--timeout 2",
                f"--results_db_path {filtered_db_path}",
                f"--results_db_flags unique",
                f"--min_total_atoms {args.min_total_atoms}",
                f"--min_heavy_atoms {args.min_heavy_atoms}"
            ])
            
            filter_cmd_str = " ".join(filter_cmd).encode('ascii', 'ignore').decode('ascii')
            if not run_command(filter_cmd_str):
                filtered_db_path = generated_db_path

        all_predicted_e1 = []

        with ase.db.connect(filtered_db_path) as db:
            total_mols_in_db = db.count()
            if total_mols_in_db == 0:
                print("No molecules found in database")
                break

            atoms_buffer = []
            
            for row_idx, row in enumerate(db.select()):
                atoms = row.toatoms()
                atoms_buffer.append(atoms)

                if len(atoms_buffer) == mols_to_predict_batch or row_idx == total_mols_in_db - 1:
                    if not atoms_buffer: 
                        continue

                    batch_input_data = []
                    for current_atoms in atoms_buffer:
                        input_data = prep_xpainn_input(current_atoms, device)
                        if input_data is not None:
                            batch_input_data.append(input_data)

                    if batch_input_data:
                        from torch_geometric.data import Batch
                        pyg_batch = Batch.from_data_list(batch_input_data).to(device)
                        
                        with torch.no_grad():
                            predictions = xpainn_model(pyg_batch)
                            
                            if isinstance(predictions, dict) and 'E1' in predictions:
                                e1_preds_batch = predictions['E1'].cpu().numpy()
                            elif isinstance(predictions, dict) and 'CC2' in predictions:
                                e1_preds_batch = predictions['CC2'].cpu().numpy()
                            elif isinstance(predictions, torch.Tensor):
                                e1_preds_batch = predictions.cpu().numpy()
                            else:
                                raise ValueError(f"Cannot extract E1 values from model output: {predictions}")
                            
                            if hasattr(e1_preds_batch, 'ndim') and e1_preds_batch.ndim > 1:
                                e1_preds_batch = e1_preds_batch.flatten()
                                
                            if hasattr(e1_preds_batch, '__iter__'):
                                all_predicted_e1.extend([float(x) for x in e1_preds_batch])
                            else:
                                all_predicted_e1.append(float(e1_preds_batch))

                    atoms_buffer = []

        if all_predicted_e1:
            e1_predictions_path = iteration_dir / "all_predicted_e1_values.npy"
            np.save(e1_predictions_path, np.array(all_predicted_e1))
            
            e1_stats = {
                "iteration": i,
                "mean_e1": float(np.mean(all_predicted_e1)),
                "std_e1": float(np.std(all_predicted_e1)),
                "min_e1": float(np.min(all_predicted_e1)),
                "max_e1": float(np.max(all_predicted_e1)),
                "count": len(all_predicted_e1)
            }
            stats_file_path = iteration_dir / "e1_statistics.json"
            with open(stats_file_path, 'w') as f:
                json.dump(e1_stats, f, indent=2)

            actual_completed_iterations = i

            if args.enable_early_stopping:
                metric_key = 'mean_e1' if args.early_stop_metric == 'mean' else 'min_e1'
                current_metric_value = e1_stats.get(metric_key)
                
                if i == 1:
                    last_e1_metric_value = current_metric_value
                elif current_metric_value is not None and last_e1_metric_value != float('inf'):
                    abs_change = abs(current_metric_value - last_e1_metric_value)
                    
                    no_significant_change_counter = no_significant_change_counter + 1 if abs_change < args.early_stop_threshold_e1 else 0
                    
                    if no_significant_change_counter >= args.early_stop_patience:
                        print(f"Early stopping triggered at iteration {i}")
                        break
                    
                    last_e1_metric_value = current_metric_value

        if use_dynamic_threshold and len(all_predicted_e1) > 0:
            mean_e1 = np.mean(all_predicted_e1)
            std_e1 = np.std(all_predicted_e1)
            dynamic_threshold = mean_e1 - k_std_factor * std_e1
            actual_threshold = max(dynamic_threshold, 1.5)
            selection_threshold_e1 = actual_threshold
        
        selected_molecules_atoms = []
        molecules_selected_count = 0
        
        with ase.db.connect(filtered_db_path) as db_reselect:
            for row_idx, (row, e1_val) in enumerate(zip(db_reselect.select(), all_predicted_e1)):
                e1_scalar = float(e1_val)
                
                if e1_scalar < selection_threshold_e1:
                    selected_molecules_atoms.append(row.toatoms())
                    molecules_selected_count += 1

        if molecules_selected_count < min_mols_for_retrain and selection_threshold_e1 < 10.0:
            selection_threshold_e1 += 0.5
            selected_molecules_atoms = []
            molecules_selected_count = 0
            
            with ase.db.connect(filtered_db_path) as db_reselect:
                for row_idx, (row, e1_val) in enumerate(zip(db_reselect.select(), all_predicted_e1)):
                    if float(e1_val) < selection_threshold_e1:
                        selected_molecules_atoms.append(row.toatoms())
                        molecules_selected_count += 1

        if molecules_selected_count >= min_mols_for_retrain:
            if selected_db_path.exists():
                selected_db_path.unlink()
            
            with ase.db.connect(selected_db_path) as next_db:
                next_db.metadata = {
                    '_distance_unit': 'Ang',
                    'distance_unit': 'Ang',
                    '_property_unit_dict': {
                        'energy_U0': 'eV', 'energy_U': 'eV', 'enthalpy_H': 'eV',
                        'free_energy': 'eV', 'homo': 'eV', 'lumo': 'eV',
                        'gap': 'eV', 'zpve': 'eV'
                    },
                    'property_unit_dict': {
                        'energy_U0': 'eV', 'energy_U': 'eV', 'enthalpy_H': 'eV',
                        'free_energy': 'eV', 'homo': 'eV', 'lumo': 'eV',
                        'gap': 'eV', 'zpve': 'eV'
                    }
                }
                
                for atoms_idx, atoms_obj in enumerate(selected_molecules_atoms):
                    connectivity, neighbors = get_connectivity(atoms_obj, cutoff=bond_connection_cutoff)
                    compressed_con_mat = compressor.compress(connectivity)
                    
                    n_atoms = len(atoms_obj)
                    atom_mask = np.ones(n_atoms, dtype=bool)
                    max_neighbors = int(np.max(neighbors))
                    neighbor_mask = np.ones((n_atoms, max_neighbors), dtype=bool)
                    cell_offset = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)
                    
                    data_to_write = {
                        'con_mat': compressed_con_mat,
                        'neighbors': neighbors.tolist(),
                        'atom_mask': atom_mask.tolist(),
                        'neighbor_mask': neighbor_mask.tolist(),
                        'cell_offset': cell_offset.tolist()
                    }
                    next_db.write(atoms_obj, data=data_to_write)
        else:
            print(f"Insufficient molecules selected ({molecules_selected_count} < {min_mols_for_retrain}). Stopping.")
            break
    
    all_stats_path = base_results_dir / "all_iterations_stats.json"
    all_stats = {}
    for iter_num_stat in range(1, actual_completed_iterations + 1):
        iteration_dir = base_results_dir / f"iteration_{iter_num_stat}"
        e1_stats_path = iteration_dir / "e1_statistics.json"
        if e1_stats_path.exists():
            with open(e1_stats_path, 'r') as f:
                all_stats[iter_num_stat] = json.load(f)
    
    with open(all_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Single-objective iterative biasing molecular generation workflow")
    
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Project root directory containing models and data folders (default: current directory)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save results (default: <base_dir>/results)')
    
    parser.add_argument('--num_iterations', type=int, default=2,
                        help='Number of iterations (default: 2). Maximum iterations if early stopping is enabled.')
    parser.add_argument('--mols_per_iter', type=int, default=20000,
                        help='Number of molecules to generate per iteration (default: 20000)')
    parser.add_argument('--min_mols_for_retrain', type=int, default=200,
                        help='Minimum number of molecules required for retraining (default: 200)')
    parser.add_argument('--e1_threshold', type=float, default=3.5,
                        help='E1 screening threshold in eV (default: 3.5)')
    parser.add_argument('--k_std_factor', type=float, default=1.0,
                        help='Dynamic threshold coefficient k: select molecules below mean-k*std (default: 1.0)')
    parser.add_argument('--use_dynamic_threshold', action='store_true',
                        help='Use dynamic threshold (based on distribution) instead of fixed threshold')
    parser.add_argument('--retrain_epochs', type=int, default=5,
                        help='Number of epochs for retraining (default: 5)')
    parser.add_argument('--xpainn_model_path', type=str, default="xequinet/qm8-all/checkpoints/QM8_direct_CC2_sweep_best.pt",
                        help='XPaiNN model path relative to base_dir')
    parser.add_argument('--clean', action='store_true', help='Clean previous result directories')
    parser.add_argument('--skip_filter', action='store_true', help='Skip filtering step, use generated molecules directly')
    parser.add_argument('--results_suffix', type=str, default=None,
                       help='Suffix for result directory to distinguish different runs (default: timestamp)')
    parser.add_argument('--min_total_atoms', type=int, default=3,
                        help='Minimum total atoms passed to check_validity.py (default: 3)')
    parser.add_argument('--min_heavy_atoms', type=int, default=2,
                        help='Minimum heavy atoms (non-hydrogen) passed to check_validity.py (default: 2)')
    parser.add_argument('--initial_generator_dir', type=str, default="ldhgeneration/models/qm9pretraining/qm9_pretrain_run1",
                        help='Initial G-SchNet generator model directory (relative to base_dir)')
    
    early_stop_group = parser.add_argument_group('Early Stopping Parameters')
    early_stop_group.add_argument('--enable_early_stopping', action='store_true',
                                  help='Enable early stopping logic based on E1 metric changes')
    early_stop_group.add_argument('--early_stop_metric', type=str, default='mean', choices=['mean', 'min'],
                                  help="E1 statistical metric for early stopping ('mean' or 'min') (default: 'mean')")
    early_stop_group.add_argument('--early_stop_threshold_e1', type=float, default=0.05,
                                  help="Threshold for E1 metric change (eV), below which change is considered insignificant (default: 0.05eV)")
    early_stop_group.add_argument('--early_stop_patience', type=int, default=2,
                                  help="Number of consecutive rounds with small E1 metric changes to trigger early stopping (default: 2)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    iterative_biasing_workflow(args) 