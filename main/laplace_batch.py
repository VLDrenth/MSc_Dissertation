import torch
import numpy as np
from tqdm.auto import tqdm
from batchbald_redux.batchbald import CandidateBatch, compute_conditional_entropy, compute_entropy

import torch

def rank_reduce(matrix):
    '''
    Receives a CxC matrix of rank C-1 and returns a (C-1)x(C-1) matrix of rank C-1.
    Using singular value decomposition.
    NOT USEFUL, JUST DROP LAST 
    '''
    C = matrix.shape[0]
    U, S, V = torch.svd(matrix)
    
    # Take the first C-1 columns of U and V, and the first C-1 singular values
    U_reduced = U[:C-1, :C-1]
    S_reduced = S[:C-1]
    V_reduced = V[:C-1, :C-1]
    
    # Reconstruct the (C-1)x(C-1) matrix
    reduced_matrix = torch.mm(U_reduced, torch.mm(torch.diag(S_reduced), V_reduced.t()))
    
    return reduced_matrix

def get_laplace_batch(model, pool_loader, acquisition_batch_size, method, device=None):
    '''
    model: Laplace model
    batch_size: how many observations to return
    device: device to run on
    dtype: data type
    method: method to use for batch selection
        Options: 'logit_entropy', 'probit_entropy', 'entropy', 'BALD'

    Returns: batch of observations (CandiateBatch object)
    '''
    scores = torch.zeros(len(pool_loader.dataset), 1)

    if method == 'logit_entropy':

        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing logit determinants", leave=False): 
            data = data.to(device=device)
            _, f_vars = model._glm_predictive_distribution(data, diagonal_output=False)
            scoring_batch_size = data.shape[0]

            # compute log determinant of each element of f_var
            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = torch.tensor([(torch.det(f_vars[i])) for i in range(len(f_vars))]).unsqueeze(-1)

    elif method == 'probit_entropy':

        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing probit determinants", leave=False): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]
        
            # returns softmax of samples of f
            f_samples = model.predictive_samples(data, n_samples=10000)

            # compute covariance of each element of f_samples, and drop the last class  
            covs = torch.stack([torch.cov(f_samples[:, n, :].T)[:-1, :-1] for n in range(f_samples.shape[1])])
            
            # compute determinant of each element of f_var
            determinants = torch.tensor([(torch.det(covs[i])) for i in range(len(covs))])
            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = determinants.unsqueeze(-1)
    
    elif method == 'entropy':

        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing entropies", leave=False): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            # obtain samples
            f_samples = model.predictive_samples(data, n_samples=1000)
            p_given_f = torch.nn.functional.softmax(f_samples, dim=2)

            # compute entropy
            log_probs_N_K_C = torch.log(p_given_f).swapaxes(0, 1)
            ent = compute_entropy(log_probs_N_K_C).unsqueeze(-1)

            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = ent

    elif method == 'BALD':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing BALD scores", leave=False): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            # obtain samples
            f_samples = model.predictive_samples(data, n_samples=1000)
            p_given_f = torch.nn.functional.softmax(f_samples, dim=2)

            # compute entropy
            log_probs_N_K_C = torch.log(p_given_f).swapaxes(0, 1)
            ent = compute_entropy(log_probs_N_K_C).unsqueeze(-1)

            # compute conditional entropy
            cond_ent = compute_conditional_entropy(log_probs_N_K_C).unsqueeze(-1)

            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = ent - cond_ent
    else:
        raise ValueError('Invalid method')
    
    
    # Compute top k scores
    values, indices = torch.topk(scores, acquisition_batch_size, largest=True, sorted=False, dim=0)
    return CandidateBatch(indices=indices.squeeze().tolist(), scores=values.squeeze().tolist())

        
    
