import torch
import numpy as np

def max_det_selection(kernel: torch.Tensor, n_adds: int) -> list:
    """
    Implements the kernel-space version of MaxDet for Batch Active Learning.

    :param kernel: The similarity matrix.
    :param n_adds: The number of indices to select.
    :param noise_sigma: noise_sigma**2 is added to the kernel diagonal for determinant maximization.
    :return: A list of selected indices that maximize the determinant.
    """
    diag = kernel.diag() + torch.eye(len(kernel), device=kernel.device) 
    l = None
    selected_idxs = []
    
    def get_scores() -> torch.Tensor:
        return diag
    
    for _ in range(n_adds):
        scores = get_scores().clone()
        scores[selected_idxs] = -np.Inf
        new_idx = torch.argmax(scores).item()
        
        if scores[new_idx] <= 0.0:
            print(f'Selecting index {len(selected_idxs) + 1}: new diag entry nonpositive')
            break
        
        selected_idxs.append(new_idx)
        
        l = torch.zeros(len(kernel), len(selected_idxs), device=kernel.device, dtype=diag.dtype) if l is None else l
        mat_col = kernel[new_idx].clone()
                    
        update = (1.0 / torch.sqrt(diag[new_idx])) * (mat_col - (l[:, :len(selected_idxs)-1] @ l[new_idx, :len(selected_idxs)-1]))
        
        diag -= update ** 2
        l[:, len(selected_idxs)-1] = update
        diag[new_idx] = -np.Inf
        
    return selected_idxs
