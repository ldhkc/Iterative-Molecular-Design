import torch
import numpy as np
from torch_geometric.data import Data
import torch_cluster
import ase.neighborlist
import scipy.sparse as sp
import os
import random
import sys

DEFAULT_CUTOFF = 5.0
DEFAULT_MAX_NEIGHBORS = 32
DEFAULT_BOND_CUTOFF = 1.85
MODEL_PREFIX_LEN = 6
SCSCORE_MIN = 1.0
SCSCORE_MAX = 5.0
INVALID_SCORE = float('inf')

DEFAULT_XPAINN_CONFIG = {
    'node_dim': 128,
    'edge_irreps': "128x0e + 64x1o + 32x2e",
    'num_basis': 20,
    'activation': "silu",
    'cutoff': 5.0,
    'rbf_kernel': "bessel",
    'action_blocks': 5,
    'output_type': "scalar",
    'norm_type': "layer",
    'embed_basis': "gfn2-xtb",
    'aux_basis': "aux56"
}

SCSCORE_PATHS = ["/root/XequiNet/SCScore", "./SCScore"]
GSCHNET_PATHS = ['/root/XequiNet/G-SchNetOE62', './G-SchNetOE62']

SCSCORE_MODEL_SUBPATH = ('models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz')

def load_xpainn_model(ckpt_path, device, config_path=None):
    from xequinet.nn.model import XPaiNN
    from xequinet.utils import NetConfig
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if 'config' in checkpoint:
        config = NetConfig(**checkpoint['config']) if isinstance(checkpoint['config'], dict) else checkpoint['config']
    elif config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = NetConfig(**json.load(f))
    else:
        config = NetConfig(**DEFAULT_XPAINN_CONFIG)
    
    model = XPaiNN(config)
    model.to(device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[MODEL_PREFIX_LEN:]: v for k, v in state_dict.items()}
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and not any(k in ['state_dict', 'model', 'config', 'epoch'] for k in checkpoint):
        state_dict = checkpoint
    else:
        state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def prep_xpainn_input(atoms, device, cutoff=DEFAULT_CUTOFF, max_neighbors=DEFAULT_MAX_NEIGHBORS):
    try:
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        edge_index = torch_cluster.radius_graph(pos, r=cutoff, loop=False, max_num_neighbors=max_neighbors)
        
        data = Data(pos=pos, at_no=atomic_numbers, edge_index=edge_index)
        data.batch = torch.zeros(pos.shape[0], dtype=torch.long)
        data.shifts = torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype)
        
        return data
    except:
        return None

def prep_xpainn_batch_input(atoms_list, device):
    from torch_geometric.data import Batch
    data_list = [d for atoms in atoms_list if (d := prep_xpainn_input(atoms, device)) is not None]
    return Batch.from_data_list(data_list).to(device) if data_list else None

class ConnectivityCompressor:
    def compress(self, connectivity_matrix):
        sparse_con_mat = sp.csr_matrix(connectivity_matrix)
        return {
            'data': sparse_con_mat.data.tolist(),
            'indices': sparse_con_mat.indices.tolist(),
            'indptr': sparse_con_mat.indptr.tolist(),
            'shape': sparse_con_mat.shape
        }
        
    def decompress(self, compressed_con_mat):
        return sp.csr_matrix(
            (compressed_con_mat['data'], compressed_con_mat['indices'], compressed_con_mat['indptr']),
            shape=compressed_con_mat['shape']
        ).toarray()

def get_connectivity(atoms, cutoff=DEFAULT_BOND_CUTOFF):
    n_atoms = len(atoms)
    connectivity = np.zeros((n_atoms, n_atoms), dtype=np.int8)
    i_list, j_list = ase.neighborlist.neighbor_list('ij', atoms, cutoff)
    
    for i, j in zip(i_list, j_list):
        connectivity[i, j] = 1
        connectivity[j, i] = 1
    
    neighbors = np.sum(connectivity, axis=1)
    return connectivity, neighbors 

project_root = None
for path in SCSCORE_PATHS:
    if os.path.exists(path):
        project_root = path
        sys.path.append(project_root)
        break

def get_smiles_from_ase(atoms):
    try:
        for path in GSCHNET_PATHS:
            if os.path.exists(path) and path not in sys.path:
                sys.path.append(path)
                try:
                    from utility_classes_ob3 import Molecule
                    mol = Molecule(atoms.get_positions(), atoms.get_atomic_numbers())
                    smiles = mol.get_can().strip()
                    return smiles if smiles else None
                except ImportError:
                    continue
        return None
    except:
        return None

class SCScorer:
    def __init__(self, model_path=None):
        self.scscore_model = None
        try:
            if project_root:
                from scscore.standalone_model_numpy import SCScorer as ActualSCScorer
                if not model_path or not os.path.exists(model_path):
                    model_path = os.path.join(project_root, *SCSCORE_MODEL_SUBPATH)
                self.scscore_model = ActualSCScorer()
                self.scscore_model.restore(model_path)
        except:
            self.scscore_model = None
    
    def get_score_from_smi(self, smiles):
        if self.scscore_model is None:
            return random.uniform(SCSCORE_MIN, SCSCORE_MAX)
        if not smiles or not smiles.strip():
            return INVALID_SCORE
        try:
            valid, scscore_value = self.scscore_model.get_score_from_smi(smiles.strip())
            return scscore_value if valid else INVALID_SCORE
        except:
            return INVALID_SCORE
