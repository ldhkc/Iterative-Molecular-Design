import os
import subprocess
import ase.io
import ase.db
import numpy as np
import torch
from pathlib import Path
import argparse
import datetime
import json

from utils import load_xpainn_model, prep_xpainn_input, get_connectivity, ConnectivityCompressor, SCScorer, get_smiles_from_ase

class MultiObjectiveSelector:
    
    def __init__(self, objectives, weights=None, strategy='weighted', dynamic_params=None):
        self.objectives = objectives
        self.strategy = strategy
        
        if strategy == 'weighted':
            if weights is None:
                self.weights = [1.0/len(objectives)] * len(objectives)
            else:
                self.weights = weights
                assert len(weights) == len(objectives), "Weight count must match objectives count"
                assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
        
        self.dynamic_threshold_enabled = dynamic_params.get('enabled', True) if dynamic_params else True
        self.target_selection_rate = dynamic_params.get('target_rate', 0.1) if dynamic_params else 0.1
        self.min_selection_rate = dynamic_params.get('min_rate', 0.05) if dynamic_params else 0.05
        self.threshold_history = {obj['property']: [] for obj in objectives}
    
    def normalize_property(self, values, property_name):
        if len(values) == 0:
            return values
        
        current_min, current_max = np.min(values), np.max(values)
        
        ranges = {
            'gap': (max(0.0, current_min), min(10.0, current_max)),
            'alpha': (max(20.0, current_min), min(200.0, current_max)),
            'ea': (max(-2.0, current_min), min(2.0, current_max)),
            's1': (max(0.0, current_min), min(10.0, current_max)),
            'scscore': (max(1.0, current_min), min(5.0, current_max))
        }
        
        min_val, max_val = ranges.get(property_name, (current_min, current_max))
        
        normalized = (values - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    def calculate_score(self, predictions):
        if self.strategy == 'weighted':
            return self._weighted_score(predictions)
        elif self.strategy == 'and':
            return self._and_score(predictions)
        elif self.strategy == 'or':
            return self._or_score(predictions)
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
    
    def _weighted_score(self, predictions, dynamic_thresholds=None):
        """
        Calculate weighted scores for predictions.
        
        Args:
            predictions: Dictionary of property predictions
            dynamic_thresholds: Optional dictionary of dynamic thresholds per property
            
        Returns:
            Array of weighted scores for each molecule
        """
        total_score = np.zeros(len(next(iter(predictions.values()))))
        
        for i, objective in enumerate(self.objectives):
            prop_name = objective['property']
            direction = objective['direction']
            weight = self.weights[i]
            
            # Use dynamic threshold if provided, otherwise use default
            if dynamic_thresholds and prop_name in dynamic_thresholds:
                threshold = dynamic_thresholds[prop_name]
            else:
                threshold = objective['threshold']
            
            if prop_name in predictions:
                values = predictions[prop_name]
                
                if direction == 'min':
                    distance = np.abs(values - threshold)
                    max_distance = np.max(distance)
                    base_score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
                    
                    below_threshold_bonus = np.where(values < threshold, 0.5, 0.0)
                    
                    progressive_bonus = np.where(values < threshold, 
                                               (threshold - values) / threshold * 0.3, 0.0)
                    
                    score = base_score + below_threshold_bonus + progressive_bonus
                    
                else:
                    distance = np.abs(values - threshold)
                    max_distance = np.max(distance)
                    base_score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
                    
                    above_threshold_bonus = np.where(values > threshold, 0.5, 0.0)
                    
                    progressive_bonus = np.where(values > threshold, 
                                               (values - threshold) / abs(threshold) * 0.3, 0.0)
                    
                    score = base_score + above_threshold_bonus + progressive_bonus
                
                total_score += weight * score
        
        return total_score
    
    def calculate_dynamic_thresholds(self, predictions, iteration):
        dynamic_thresholds = {}
        
        for objective in self.objectives:
            prop_name = objective['property']
            direction = objective['direction']
            base_threshold = objective['threshold']
            
            if prop_name in predictions:
                values = predictions[prop_name]
                
                percentile_map = {
                    'min': {1: 10, 2: 10, 3: 10, 4: 5, 5: 5, 6: 5},
                    'max': {1: 90, 2: 90, 3: 90, 4: 95, 5: 95, 6: 95}
                }
                
                percentile = percentile_map.get(direction, {}).get(iteration, 2 if direction == 'min' else 98)
                dynamic_threshold = np.percentile(values, percentile)
                
                if direction == 'min':
                    dynamic_threshold = min(dynamic_threshold, base_threshold)
                else:
                    dynamic_threshold = max(dynamic_threshold, base_threshold)
                
                dynamic_thresholds[prop_name] = dynamic_threshold
                self.threshold_history[prop_name].append(dynamic_threshold)
                
        
        return dynamic_thresholds
    
    def _and_score(self, predictions):
        scores = []
        for objective in self.objectives:
            prop_name = objective['property']
            direction = objective['direction']
            threshold = objective['threshold']
            
            if prop_name in predictions:
                values = predictions[prop_name]
                
                if direction == 'min':
                    satisfied = values < threshold
                else:
                    satisfied = values > threshold
                
                scores.append(satisfied.astype(float))
        
        return np.mean(scores, axis=0) if scores else np.zeros(len(next(iter(predictions.values()))))
    
    def _or_score(self, predictions):
        scores = []
        for objective in self.objectives:
            prop_name = objective['property']
            direction = objective['direction']
            threshold = objective['threshold']
            
            if prop_name in predictions:
                values = predictions[prop_name]
                
                if direction == 'min':
                    satisfied = values < threshold
                else:
                    satisfied = values > threshold
                
                scores.append(satisfied.astype(float))
        
        return np.max(scores, axis=0) if scores else np.zeros(len(next(iter(predictions.values()))))
    
    def select_molecules(self, predictions, threshold=0.5):
        scores = self.calculate_score(predictions)
        
        if self.strategy == 'weighted':
            selected_indices = scores >= threshold
        else:
            selected_indices = scores > 0.5
        
        return selected_indices, scores

    def adaptive_select_molecules(self, predictions, min_required, initial_threshold=0.5):
        current_threshold = initial_threshold
        max_attempts = 15
        
        for attempt in range(max_attempts):
            selected_indices, scores = self.select_molecules(predictions, current_threshold)
            selected_count = np.sum(selected_indices)
            
            if selected_count >= min_required:
                return selected_indices, scores, current_threshold
            else:
                if selected_count > 0:
                    selection_ratio = selected_count / len(scores)
                    target_ratio = min_required / len(scores)
                    ratio_factor = target_ratio / selection_ratio
                    current_threshold *= ratio_factor
                else:
                    current_threshold -= 0.1
                
                current_threshold = max(0.05, min(0.9, current_threshold))
        
        selected_indices, scores = self.select_molecules(predictions, current_threshold)
        selected_count = np.sum(selected_indices)
        return selected_indices, scores, current_threshold

    def adaptive_select_with_dynamic_threshold(self, scores, min_required, initial_threshold=0.5):
        current_threshold = initial_threshold
        max_attempts = 15
        
        for attempt in range(max_attempts):
            selected_indices = scores >= current_threshold
            selected_count = np.sum(selected_indices)
            
            if selected_count >= min_required:
                return selected_indices, current_threshold
            else:
                if selected_count > 0:
                    selection_ratio = selected_count / len(scores)
                    target_ratio = min_required / len(scores)
                    ratio_factor = target_ratio / selection_ratio
                    current_threshold *= ratio_factor
                else:
                    current_threshold -= 0.1
                
                current_threshold = max(0.05, min(0.9, current_threshold))
        
        selected_indices = scores >= current_threshold
        selected_count = np.sum(selected_indices)
        return selected_indices, current_threshold

    def direct_property_selection(self, predictions, min_required, iteration):
        
        dynamic_thresholds = {}
        for objective in self.objectives:
            prop_name = objective['property']
            direction = objective['direction']
            base_threshold = objective['threshold']
            
            if prop_name in predictions:
                values = predictions[prop_name]
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                k_factor = min(3.0, 1.5 + (iteration - 1) * 0.3)
                
                percentile_config = {
                    'min': [20, 20, 20, 10, 10, 10, 5],
                    'max': [80, 80, 80, 90, 90, 90, 95]
                }
                percentile_idx = min(iteration - 1, 6)
                percentile = percentile_config[direction][percentile_idx]
                
                dynamic_threshold = np.percentile(values, percentile)
                std_threshold = mean_val + (k_factor * std_val if direction == 'max' else -k_factor * std_val)
                
                if direction == 'min':
                    dynamic_threshold = min(dynamic_threshold, std_threshold, base_threshold)
                else:
                    dynamic_threshold = max(dynamic_threshold, std_threshold, base_threshold)
                
                dynamic_thresholds[prop_name] = dynamic_threshold
        
        total_molecules = len(next(iter(predictions.values())))
        final_selection = np.ones(total_molecules, dtype=bool)
        
        for prop_name, threshold in dynamic_thresholds.items():
            values = predictions[prop_name]
            direction = None
            for objective in self.objectives:
                if objective['property'] == prop_name:
                    direction = objective['direction']
                    break
            
            if direction == 'min':
                prop_selection = values < threshold
            else:
                prop_selection = values > threshold
            
            final_selection &= prop_selection
        
        if np.sum(final_selection) < min_required:
            combined_scores = np.zeros(total_molecules)
            for prop_name in predictions.keys():
                values = predictions[prop_name]
                direction = None
                for objective in self.objectives:
                    if objective['property'] == prop_name:
                        direction = objective['direction']
                        break
                
                if direction == 'min':
                    normalized = 1.0 - (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
                else:
                    normalized = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
                
                combined_scores += normalized
            
            top_indices = np.argsort(combined_scores)[-min_required:]
            final_selection = np.zeros(total_molecules, dtype=bool)
            final_selection[top_indices] = True
        
        return final_selection, dynamic_thresholds

    def calculate_adaptive_min_mols(self, database_size, base_min_mols):
        if database_size < 1000:
            return max(100, int(database_size * 0.6))
        elif database_size < 5000:
            return max(500, int(database_size * 0.5))
        else:
            return min(base_min_mols, int(database_size * 0.4))

    def calculate_safe_split_guaranteed(self, database_size, min_mols_for_retrain):
        adaptive_min_mols = self.calculate_adaptive_min_mols(database_size, min_mols_for_retrain)
        
        if database_size <= adaptive_min_mols + 5:
            return database_size - 1, 1, 0
        
        max_safe_buffer = database_size - adaptive_min_mols - 5
        buffer_size = min(max_safe_buffer, int(database_size * 0.08))
        
        effective_size = database_size - buffer_size
        
        val_size = max(1, min(int(effective_size * 0.1), int(effective_size * 0.2)))
        train_size = effective_size - val_size
        
        assert train_size >= adaptive_min_mols, f"Train size ({train_size}) smaller than required ({adaptive_min_mols})"
        assert train_size + val_size + buffer_size <= database_size, f"Total ({train_size + val_size + buffer_size}) exceeds database size ({database_size})"
        
        return train_size, val_size, buffer_size

    def smart_retry_split(self, database_size, min_mols_for_retrain, retry_count):
        if retry_count >= 3:
            return database_size - 1, 1, 0
        
        adaptive_min_mols = self.calculate_adaptive_min_mols(database_size, min_mols_for_retrain)
        train_size = adaptive_min_mols
        val_size = 1 if retry_count == 2 else max(1, int((database_size - train_size) * 0.05))
        buffer_size = database_size - train_size - val_size
        
        return train_size, val_size, buffer_size

    def calculate_smart_buffer_ratio(self, database_size, iteration, min_mols_for_retrain):
        split_train, split_val, safety_buffer = self.calculate_safe_split_guaranteed(database_size, min_mols_for_retrain)
        
        buffer_ratio = safety_buffer / database_size if database_size > 0 else 0
        
        return buffer_ratio, split_train, split_val, safety_buffer

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

def run_command(command, shell=True, check=True, verbose=True):
    if verbose:
        print(f"\nRunning command:\n{command}\n")
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['LC_ALL'] = 'C.UTF-8'
        env['LANG'] = 'C.UTF-8'
        
        subprocess.run(command, shell=shell, check=check, capture_output=True, text=True, 
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

def iterative_biasing_workflow_multiobj(args):
    number_of_loops = args.num_iterations
    mols_to_generate = args.mols_per_iter
    mols_to_predict_batch = 256
    min_mols_for_retrain = args.min_mols_for_retrain
    retrain_epochs = args.retrain_epochs
    retrain_patience = 5
    bond_connection_cutoff = 1.85
    objectives = []
    for i, (prop, direction, threshold) in enumerate(zip(
        args.properties, args.directions, args.thresholds
    )):
        objectives.append({
            'property': prop,
            'direction': direction,
            'threshold': threshold
        })
    
    base_dir = Path(args.base_dir)
    initial_generator_dir = base_dir / args.initial_generator_dir
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
    
    predictors = {}
    for prop in args.properties:
        if prop == 'scscore':
            try:
                predictors[prop] = SCScorer(model_path=str(base_dir / args.scscore_model_path))
            except Exception as e:
                print(f"Warning: Failed to load SCScore predictor: {e}")
                args.properties = [p for p in args.properties if p != 'scscore']
                if len(args.directions) > len(args.properties):
                    args.directions = args.directions[:len(args.properties)]
                if len(args.thresholds) > len(args.properties):
                    args.thresholds = args.thresholds[:len(args.properties)]
        else:
            model_path = getattr(args, f'{prop}_model_path')
            try:
                predictors[prop] = load_xpainn_model(str(base_dir / model_path), device)
                predictors[prop].eval()
            except Exception as e:
                print(f"Failed to load {prop} predictor: {e}")
                return
    
    dynamic_params = {
        'enabled': getattr(args, 'enable_dynamic_thresholds', True),
        'target_rate': getattr(args, 'target_selection_rate', 0.1),
        'min_rate': getattr(args, 'min_selection_rate', 0.05)
    }
    
    selector = MultiObjectiveSelector(
        objectives=objectives,
        weights=None,
        strategy='direct',
        dynamic_params=dynamic_params
    )
    
    compressor = ConnectivityCompressor()
    current_generator_dir = initial_generator_dir
    last_metric_values = {prop: float('inf') for prop in args.properties}
    convergence_counters = {prop: 0 for prop in args.properties}
    actual_completed_iterations = 0
    
    convergence_thresholds = {
        'gap': 0.05,
        'alpha': 2.0,
        'ea': 0.05,
        'scscore': 0.1
    }
    
    convergence_patience = getattr(args, 'convergence_patience', 3)
    min_converged_ratio = getattr(args, 'min_converged_ratio', 0.7)
    adaptive_threshold_factor = getattr(args, 'adaptive_threshold_factor', 0.1)
    min_threshold_ratio = 0.5
    max_threshold_ratio = 2.0
    
    for i in range(1, number_of_loops + 1):
        print(f"\n{'='*20} Iteration {i} {'='*20}")
        iteration_dir = base_results_dir / f"iteration_{i}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        generator_to_use_for_gen = current_generator_dir
        generated_db_path = iteration_dir / "generated_molecules.db"
        filtered_db_path = iteration_dir / "filtered_molecules.db"
        selected_db_path = iteration_dir / "selected_for_retrain.db"
        
        if i > 1:
            print(f"Step 1: Retraining G-SchNet using iteration {i-1} data...")
            last_selected_db_path = base_results_dir / f"iteration_{i-1}" / "selected_for_retrain.db"

            if not last_selected_db_path.is_file():
                 print(f"Error: Training data for iteration {i} not found: {last_selected_db_path}. Stopping.")
                 break

            try:
                 with ase.db.connect(last_selected_db_path) as db_conn:
                     database_size = db_conn.count()
                 print(f"Retraining database size: {database_size}")

                 if database_size < min_mols_for_retrain:
                     print(f"Warning: Insufficient molecules ({database_size} < {min_mols_for_retrain}). Skipping retraining, using previous model.")
                 else:
                    buffer_ratio, split_train, split_val, safety_buffer = selector.calculate_smart_buffer_ratio(
                        database_size, i, min_mols_for_retrain
                    )

                    model_candidates = [
                        current_generator_dir / "best_model",
                        current_generator_dir / "checkpoints" / "last.ckpt"
                    ]
                    pretrained_model_file = next((f for f in model_candidates if f.is_file()), None)

                    if pretrained_model_file is None:
                        print(f"Error: Cannot find model/checkpoint in {current_generator_dir} for fine-tuning. Stopping.")
                        break

                    model_output_dir = iteration_dir / "model_retrained"
                    model_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    extracted_weights_path = None
                    try:
                        weights_output_file = model_output_dir / "extracted_nn_module_weights.pt"
                        extracted_weights_path = extract_nn_module_weights(
                            pretrained_model_file, 
                            output_path=weights_output_file
                        )
                    except Exception as e_extract:
                        pass

                    if extracted_weights_path:
                        current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        run_id_for_train = f"iter{i}_{current_timestamp}"

                        potential_split_files = [
                            model_output_dir / run_id_for_train / "split.npz",
                            model_output_dir / "split.npz",
                            last_selected_db_path.parent / "split.npz"
                        ]
                        for split_file in potential_split_files:
                            if split_file.exists():
                                split_file.unlink()

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
                            f"++globals.lr=3e-5"
                        ] + ([f"++trainer.accelerator=gpu", f"++trainer.devices=1"] if device.type == 'cuda' else [])

                        retrain_cmd_str = " ".join(retrain_cmd)
                        training_success = False
                        retry_count = 0
                        max_retries = 3
                        
                        while not training_success and retry_count < max_retries:
                            if not run_command(retrain_cmd_str):
                                retry_count += 1
                                print(f"G-SchNet retraining failed for iteration {i} (attempt {retry_count}/{max_retries})")
                                
                                if retry_count < max_retries:
                                    new_split_train, new_split_val, conservative_buffer = selector.smart_retry_split(
                                        database_size, min_mols_for_retrain, retry_count
                                    )
                                    retrain_cmd = [
                                        f"{gschnet_script_dir}/gschnet_train",
                                        f"--config-dir={config_dir}",
                                        f"experiment=iterative_finetune",
                                        f"data.datapath={last_selected_db_path}",
                                        f"data.num_train={new_split_train}",
                                        f"data.num_val={new_split_val}",
                                        f"run.path={model_output_dir}",
                                        f"run.id={run_id_for_train}",
                                        f"trainer.max_epochs={retrain_epochs}",
                                        f"++callbacks.early_stopping.patience={retrain_patience}",
                                        f"++task.init_weights_path={extracted_weights_path}",
                                        f"++globals.lr=3e-5"
                                    ]
                                    if device.type == 'cuda':
                                        retrain_cmd.extend([f"++trainer.accelerator=gpu", f"++trainer.devices=1"])
                                    
                                    retrain_cmd_str = " ".join(retrain_cmd)
                                else:
                                    print(f"Multiple retraining failures, stopping iteration.")
                                    break
                            else:
                                training_success = True
                        
                        if not training_success:
                            print(f"Error occurred during G-SchNet retraining for iteration {i}. Stopping.")
                            break
                        
                        actual_trained_model_dir = model_output_dir / run_id_for_train
                        current_generator_dir = actual_trained_model_dir if (actual_trained_model_dir / "best_model").exists() else model_output_dir
                        generator_to_use_for_gen = current_generator_dir
                    else:
                        generator_to_use_for_gen = current_generator_dir
            
            except Exception as e_prepare:
                print(f"General error during retraining preparation for iteration {i}: {e_prepare}")
                break
        
        print(f"Step 2: Using model from {generator_to_use_for_gen} to generate {mols_to_generate} molecules...")
        
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
            print(f"Error occurred during molecule generation for iteration {i}. Stopping.")
            break
        
        if args.skip_filter:
            print("Step 3: Skipping filtering step, using generated molecules directly")
            filtered_db_path = generated_db_path
        else:
            print("Step 3: Filtering generated molecules...")
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
            
            filter_cmd_str = " ".join(filter_cmd)
            if not run_command(filter_cmd_str):
                print(f"Error occurred during molecule filtering for iteration {i}. Continuing with unfiltered molecules.")
                filtered_db_path = generated_db_path
        
        print(f"Step 4: Predicting multiple property values...")
        
        all_predictions = {prop: [] for prop in args.properties}
        prediction_records = {}
        
        try:
            with ase.db.connect(filtered_db_path) as db:
                total_mols_in_db = db.count()
                print(f"Total molecules to predict in database: {total_mols_in_db}")
                if total_mols_in_db == 0:
                    print("No molecules found in generated/filtered database. Stopping.")
                    break
                
                atoms_buffer = []
                row_ids_buffer = []
                
                for row_idx, row in enumerate(db.select()):
                    atoms = row.toatoms()
                    atoms_buffer.append(atoms)
                    row_ids_buffer.append(row.id)
                    
                    if len(atoms_buffer) == mols_to_predict_batch or row_idx == total_mols_in_db - 1:
                        if not atoms_buffer:
                            continue
                        
                        for row_id in row_ids_buffer:
                            prediction_records[row_id] = {prop: None for prop in args.properties}
                        for prop in args.properties:
                            if prop == 'scscore':
                                for buffer_idx, (current_atoms, row_id) in enumerate(zip(atoms_buffer, row_ids_buffer)):
                                    try:
                                        smiles = get_smiles_from_ase(current_atoms)
                                        if smiles:
                                            scscore_result = predictors[prop].get_score_from_smi(smiles)
                                            score = float(scscore_result[1] if isinstance(scscore_result, tuple) else scscore_result)
                                            
                                            if np.isfinite(score) and 1.0 <= score <= 5.0:
                                                prediction_records[row_id][prop] = score
                                                all_predictions[prop].append(score)
                                            else:
                                                prediction_records[row_id][prop] = None
                                                if np.isfinite(score):
                                                    pass
                                        else:
                                            prediction_records[row_id][prop] = None
                                    except Exception as e:
                                        prediction_records[row_id][prop] = None
                            else:
                                batch_input_data = []
                                valid_indices = []
                                valid_row_ids = []
                                
                                for buffer_idx, current_atoms in enumerate(atoms_buffer):
                                    try:
                                        input_data = prep_xpainn_input(current_atoms, device)
                                        if input_data is not None:
                                            batch_input_data.append(input_data)
                                            valid_indices.append(buffer_idx)
                                            valid_row_ids.append(row_ids_buffer[buffer_idx])
                                    except Exception as e:
                                        continue
                                
                                if batch_input_data:
                                    from torch_geometric.data import Batch
                                    try:
                                        pyg_batch = Batch.from_data_list(batch_input_data).to(device)
                                        
                                        with torch.no_grad():
                                            predictions = predictors[prop](pyg_batch)
                                            
                                            if isinstance(predictions, dict):
                                                property_keys_map = {
                                                    'gap': ['gap', 'gap_b3lyp_631g_Ha', 'HOMO-LUMO_gap', 'energy_gap'],
                                                    'alpha': ['alpha', 'isotropic_polarizability', 'pol', 'polarizability'],
                                                    'ea': ['ea', 'lumo', 'LUMO', 'electron_affinity']
                                                }
                                                
                                                property_preds_batch = None
                                                for key in property_keys_map.get(prop, []):
                                                    if key in predictions:
                                                        property_preds_batch = predictions[key].cpu().numpy()
                                                        break
                                                
                                                if property_preds_batch is None:
                                                    first_key = next(iter(predictions.keys()))
                                                    property_preds_batch = predictions[first_key].cpu().numpy()
                                            elif isinstance(predictions, torch.Tensor):
                                                property_preds_batch = predictions.detach().cpu().numpy()
                                            else:
                                                    raise ValueError(f"Cannot extract values from {prop} model output")
                                            
                                            if hasattr(property_preds_batch, 'ndim') and property_preds_batch.ndim > 1:
                                                property_preds_batch = property_preds_batch.flatten()
                                            if prop == 'ea':
                                                property_preds_batch = -property_preds_batch
                                            
                                            validation_ranges = {
                                                'gap': (0.0, 20.0),
                                                'alpha': (10.0, 500.0),
                                                'ea': (-5.0, 5.0),
                                                'scscore': (1.0, 5.0)
                                            }
                                            
                                            for pred_idx, (row_id, pred_val) in enumerate(zip(valid_row_ids, property_preds_batch)):
                                                pred_float = float(pred_val)
                                                
                                                if np.isfinite(pred_float):
                                                    min_val, max_val = validation_ranges.get(prop, (float('-inf'), float('inf')))
                                                    is_valid = min_val <= pred_float <= max_val
                                                else:
                                                    is_valid = False
                                                
                                                if is_valid:
                                                    prediction_records[row_id][prop] = pred_float
                                                    all_predictions[prop].append(pred_float)
                                                else:
                                                    prediction_records[row_id][prop] = None
                                    
                                    except Exception as e:
                                        for row_id in valid_row_ids:
                                            prediction_records[row_id][prop] = None
                        
                        atoms_buffer = []
                        row_ids_buffer = []
                
                for prop in args.properties:
                    if all_predictions[prop]:
                        predictions_path = iteration_dir / f"all_predicted_{prop}_values.npy"
                        np.save(predictions_path, np.array(all_predictions[prop]))
                
                properties_stats = {"iteration": i}
                for prop in args.properties:
                    if all_predictions[prop]:
                        valid_values = [v for v in all_predictions[prop] if v is not None]
                        if valid_values:
                            properties_stats[prop] = {
                                "mean": float(np.mean(valid_values)),
                                "std": float(np.std(valid_values)),
                                "min": float(np.min(valid_values)),
                                "max": float(np.max(valid_values)),
                                "count": len(valid_values)
                            }
                
                stats_file_path = iteration_dir / "properties_statistics.json"
                with open(stats_file_path, 'w') as f:
                    json.dump(properties_stats, f, indent=2)
                
                actual_completed_iterations = i
                if getattr(args, 'enable_convergence_detection', False) and i > 1:
                    converged_properties = []
                    total_change = 0.0
                    converged_count = 0
                    
                    for prop in args.properties:
                        if prop in properties_stats and last_metric_values[prop] != float('inf'):
                            current_mean = properties_stats[prop]['mean']
                            previous_mean = last_metric_values[prop]
                            change = abs(current_mean - previous_mean)
                            total_change += change
                            
                            threshold = convergence_thresholds.get(prop, 0.05)
                            
                            if i > 2:
                                adaptive_threshold = threshold * (1 + adaptive_threshold_factor * (i - 2))
                                adaptive_threshold = max(threshold * min_threshold_ratio, 
                                                       min(adaptive_threshold, threshold * max_threshold_ratio))
                                threshold = adaptive_threshold
                            
                            if change < threshold:
                                convergence_counters[prop] += 1
                                converged_properties.append(prop)
                                converged_count += 1
                            else:
                                convergence_counters[prop] = 0
                    
                    converged_with_patience = [prop for prop in converged_properties 
                                             if convergence_counters[prop] >= convergence_patience]
                    converged_ratio = len(converged_with_patience) / len(args.properties)
                    avg_change = total_change / len(args.properties)
                    
                    should_stop = (converged_ratio >= min_converged_ratio) or (avg_change < 0.01)
                    
                    if should_stop:
                        print(f"Convergence detected at iteration {i}. Stopping.")
                        break
                
                for prop in args.properties:
                    if prop in properties_stats:
                        last_metric_values[prop] = properties_stats[prop]['mean']
                
                print(f"Step 5: Multi-objective screening...")
                
                batch_predictions = {}
                valid_row_ids = []
                
                for row_id, pred_dict in prediction_records.items():
                    all_valid = all(pred_dict[prop] is not None for prop in args.properties)
                    if all_valid:
                        for prop in args.properties:
                            if prop not in batch_predictions:
                                batch_predictions[prop] = []
                            batch_predictions[prop].append(pred_dict[prop])
                        valid_row_ids.append(row_id)
                
                if batch_predictions and valid_row_ids:
                    for prop in args.properties:
                        batch_predictions[prop] = np.array(batch_predictions[prop])
                    
                    selected_indices, dynamic_thresholds = selector.direct_property_selection(
                        batch_predictions, min_required=min_mols_for_retrain, iteration=i
                    )
                    
                    selected_molecules_atoms = []
                    molecules_selected_count = 0
                    
                    with ase.db.connect(filtered_db_path) as db_reselect:
                        for row_idx, row in enumerate(db_reselect.select()):
                            if row.id in valid_row_ids:
                                valid_idx = valid_row_ids.index(row.id)
                                if selected_indices[valid_idx]:
                                    selected_molecules_atoms.append(row.toatoms())
                                    molecules_selected_count += 1
                    
                    selection_path = iteration_dir / "selection_indices.npy"
                    np.save(selection_path, selected_indices)
                else:
                    molecules_selected_count = 0
        
        except Exception as e:
            print(f"Error during multi-objective prediction/selection for iteration {i}: {e}")
            break
        
        if molecules_selected_count >= min_mols_for_retrain:
            print(f"Step 6: Saving {molecules_selected_count} selected molecules to '{selected_db_path}'...")
            if selected_db_path.exists():
                selected_db_path.unlink()
            
            try:
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
                        try:
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
                            
                                
                        except Exception as write_err:
                            try:
                                next_db.write(atoms_obj)
                            except:
                                pass
                
            except Exception as e:
                print(f"Error saving selected molecules database: {e}")
                break
        else:
            print(f"Insufficient selected molecules ({molecules_selected_count} < {min_mols_for_retrain}). Stopping iteration.")
            break
    
    print("\nMulti-objective iterative biasing workflow completed.")
    
    print(f"\nMulti-objective statistics for each iteration:")
    for iter_num_stat in range(1, actual_completed_iterations + 1):
        iteration_dir = base_results_dir / f"iteration_{iter_num_stat}"
        properties_stats_path = iteration_dir / "properties_statistics.json"
        
        if properties_stats_path.exists():
            with open(properties_stats_path, 'r') as f:
                stats = json.load(f)
                print(f"Iteration {iter_num_stat}:")
                for prop in args.properties:
                    if prop in stats:
                        prop_stats = stats[prop]
                        print(f"  {prop}: mean={prop_stats.get('mean', 'N/A'):.4f}, min={prop_stats.get('min', 'N/A'):.4f}")
    
    print(f"\nTask completed! All results saved to: {base_results_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-objective iterative biasing molecular generation workflow")
    
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Project root directory containing models and data folders (default: current directory)')
    
    parser.add_argument('--num_iterations', type=int, default=2,
                        help='Number of iterations (default: 2)')
    parser.add_argument('--mols_per_iter', type=int, default=20000,
                        help='Number of molecules to generate per iteration (default: 20000)')
    parser.add_argument('--min_mols_for_retrain', type=int, default=500,
                        help='Minimum number of molecules required for retraining (default: 500)')
    parser.add_argument('--retrain_epochs', type=int, default=5,
                        help='Number of epochs for retraining (default: 5)')
    parser.add_argument('--clean', action='store_true', help='Clean previous result directories')
    parser.add_argument('--skip_filter', action='store_true', help='Skip filtering step')
    parser.add_argument('--results_suffix', type=str, default=None,
                        help='Suffix for result directory name (default: timestamp)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save results (default: <base_dir>/results)')
    parser.add_argument('--min_total_atoms', type=int, default=3,
                        help='Minimum total atoms for molecule filtering (default: 3)')
    parser.add_argument('--min_heavy_atoms', type=int, default=2,
                        help='Minimum heavy atoms for molecule filtering (default: 2)')
    parser.add_argument('--initial_generator_dir', type=str, default="ldhgeneration/models/qm9pretraining/qm9_pretrain_run1",
                        help='Initial G-SchNet generator model directory (relative to base_dir)')
    
    parser.add_argument('--properties', nargs='+', default=['gap', 'alpha', 'ea'],
                        help='Target properties list (default: gap alpha ea)')
    parser.add_argument('--directions', nargs='+', default=['min', 'min', 'max'],
                        help='Optimization directions: min or max (default: min min max)')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[3.0, 50.0, -1.0],
                        help='Initial thresholds for each property (default: 3.0 50.0 -1.0)')
    parser.add_argument('--use_direct_selection', action='store_true', default=True,
                        help='Use weight-free direct property selection (similar to single-objective logic)')
    
    parser.add_argument('--gap_model_path', type=str, default="xequinet/qm9-all/checkpoints/gap_best.pt",
                        help='Gap prediction model path (relative to base_dir)')
    parser.add_argument('--alpha_model_path', type=str, default="xequinet/qm9-all/checkpoints/alpha_best.pt",
                        help='Alpha prediction model path (relative to base_dir)')
    parser.add_argument('--ea_model_path', type=str, default="xequinet/qm9-all/checkpoints/ea_best.pt",
                        help='EA prediction model path (relative to base_dir)')
    parser.add_argument('--scscore_model_path', type=str, default="SCScore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz",
                        help='SCScore model weights path (relative to base_dir)')
    
    convergence_group = parser.add_argument_group('Multi-objective Convergence Parameters')
    convergence_group.add_argument('--enable_convergence_detection', action='store_true',
                                   help='Enable multi-objective convergence detection')
    convergence_group.add_argument('--convergence_patience', type=int, default=3,
                                   help='Number of consecutive converged rounds before stopping (default: 3)')
    convergence_group.add_argument('--min_converged_ratio', type=float, default=0.7,
                                   help='Minimum ratio of converged properties to consider stopping (default: 0.7)')
    convergence_group.add_argument('--weight_converged_ratio', type=float, default=0.8,
                                   help='Minimum ratio of converged weighted properties to stop (default: 0.8)')
    convergence_group.add_argument('--adaptive_threshold_factor', type=float, default=0.1,
                                   help='Adaptive threshold adjustment factor (default: 0.1)')
    
    dynamic_group = parser.add_argument_group('Dynamic Threshold Parameters')
    dynamic_group.add_argument('--enable_dynamic_thresholds', action='store_true', default=True,
                               help='Enable dynamic threshold system (default: True)')
    dynamic_group.add_argument('--target_selection_rate', type=float, default=0.1,
                               help='Target selection rate (default: 0.1)')
    dynamic_group.add_argument('--min_selection_rate', type=float, default=0.05,
                               help='Minimum selection rate (default: 0.05)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    iterative_biasing_workflow_multiobj(args)
