from torch.utils.data import Dataset
import torch

from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------
class MobilityDataset(Dataset):
    """
    Charge une liste de (cells_tensor, times_tensor, mask_tensor) déjà encodés.
 
    En pratique, vous construirez `samples` en parcourant vos CSV :
 
        samples = []
        for user_data in all_users:
            c, t, m = encode_sequence(user_data['cell_ids'],
                                      user_data['timestamps'],
                                      eos_id)
            samples.append((c, t, m))
        ds = MobilityDataset(samples)
 
    Chaque item retourné est (cells, times, mask) — trois tenseurs de longueur
    MAX_SEQ_LEN. La formation des paires (input, cible) est gérée dans la
    boucle d'entraînement (décalage d'un pas).
    """
 
    def __init__(self, sample_path: Path):
        
        data = torch.load(sample_path)

        self.cells = data["cells"]
        self.times = data["times"]
        self.mask  = data["mask"]
 
    def __len__(self) -> int:
        return self.cells.size(0)
 
    def __getitem__(self, idx: int):
        return (
            self.cells[idx],
            self.times[idx],
            self.mask[idx]
        )