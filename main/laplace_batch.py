import torch
import numpy as np
import gpytorch as gpt
from scipy.stats import dirichlet
from .badge import BADGE, TrainedBayesianModel
from tqdm.auto import tqdm
from batchbald_redux.batchbald import CandidateBatch
from .bald_sampling import compute_bald, compute_entropy, compute_conditional_entropy, compute_emp_cov, max_joint_eig
from .approximation import compute_S
import math

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

    Returns: batch of observations (CandiateBatch object)
    '''
    scores = torch.empty(len(pool_loader.dataset), 1).to(device=device)
    scores.fill_(float('-inf'))

    if method == 'random':
        indices = torch.randperm(len(pool_loader.dataset))[:acquisition_batch_size]
        return CandidateBatch(indices=indices.tolist(), scores=[0]*acquisition_batch_size)
    elif method == 'logit_entropy':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing logit determinants", leave=True): 
            data = data.to(device=device)
            _, f_vars = model._glm_predictive_distribution(data, diagonal_output=False)
            scoring_batch_size = data.shape[0]

            # compute log determinant of each element of f_var
            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = torch.tensor([(0.5*torch.logdet(f_vars[i])) for i in range(len(f_vars))]).unsqueeze(-1)

    elif method == 'softmax_entropy':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing probit determinants", leave=True): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            # get logits
            logits = model(data, pred_type='glm', link_approx='probit')
                    
            # for each sample, compute entropy using categorical distribution
            entropies = torch.tensor([dirichlet(alpha=logits[i]).entropy() for i in range(len(logits))])

            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = entropies.unsqueeze(-1)
    
    elif method == 'entropy':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing entropies", leave=True): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            ent = compute_entropy(model, data)
            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = ent.unsqueeze(-1)

    elif method == 'bald':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing BALD scores", leave=True): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            # compute BALD
            bald = compute_bald(model, data, train_loader=None, refit=False, n_samples=10).unsqueeze(-1)

            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = bald

    elif method == 'max_diag_S':
        for i, (data, _) in tqdm(enumerate(pool_loader), desc="Computing Jacobian eigenvalues", leave=True): 
            data = data.to(device=device)
            scoring_batch_size = data.shape[0]

            S = compute_S(model, data)

            scores[i*scoring_batch_size:(i+1)*scoring_batch_size] = torch.diag(S).unsqueeze(-1)
    elif method == 'max_logdet_S':
        # extract data from the pool
        pool_data = torch.cat([data for data, _ in pool_loader], dim=0).to(device=device)

        S = compute_S(model, pool_data)
        # add identity matrix to S
        mat = S + torch.eye(S.shape[0]).to(device=device)
        indices, log_det, _ = stochastic_greedy_maxlogdet(mat, acquisition_batch_size)
        return CandidateBatch(indices=indices, scores=[log_det]*acquisition_batch_size)
    elif method == 'badge':
        trained_model = TrainedBayesianModel(model)
        badge_selector = BADGE(acquisition_size=acquisition_batch_size)
        return badge_selector.compute_candidate_batch(trained_model, pool_loader, device)
        
    elif method == 'empirical_covariance':
         # extract data from the pool
        pool_data = torch.cat([data for data, _ in pool_loader], dim=0).to(device=device)

        # compute the empirical covariance matrix
        cov = compute_emp_cov(model, pool_data)

        # add identity matrix to S
        mat = cov + torch.eye(cov.shape[0]).to(device=device)

        indices, log_det, _ = stochastic_greedy_maxlogdet(mat, acquisition_batch_size)
        return CandidateBatch(indices=indices, scores=[log_det]*acquisition_batch_size)
    elif method == 'joint_eig':
        # extract data from the pool
        pool_data = torch.cat([data for data, _ in pool_loader], dim=0).to(device=device)

        indices, score = max_joint_eig(model=model, data=pool_data, K=10, batch_size=acquisition_batch_size)

        return CandidateBatch(indices=indices, scores=[score]*acquisition_batch_size)
    else:
        raise ValueError('Invalid method')
    
    # Compute top k scores
    values, indices = torch.topk(scores, acquisition_batch_size, largest=True, sorted=False, dim=0)
    return CandidateBatch(indices=indices.squeeze().tolist(), scores=values.squeeze().tolist())


        
def greedy_max_logdet(matrix, k):
    """
    Greedily selects k rows and columns from the input matrix to maximize the determinant.
    
    Args:
    matrix (torch.Tensor): NxN input matrix
    k (int): Size of the submatrix to select
    
    Returns:
    tuple: (selected_indices, max_determinant)
    """
    N = matrix.shape[0]
    if k > N:
        raise ValueError("k cannot be larger than the matrix size")
    
    # Initialize the list of selected indices
    selected_indices = []
    
    for _ in range(k):
        max_det = float('-inf')
        best_index = -1
        
        # Try adding each remaining index and calculate the determinant
        for i in range(N):
            if i not in selected_indices:
                current_indices = selected_indices + [i]
                submatrix = matrix[current_indices][:, current_indices]
                det = torch.det(submatrix).item()
                
                # Update if we found a better determinant
                if det > max_det:
                    max_det = det
                    best_index = i
        
        # Add the best index found in this iteration
        selected_indices.append(best_index)
    
    # Calculate the final determinant
    final_submatrix = matrix[selected_indices][:, selected_indices]
    max_determinant = torch.logdet(final_submatrix).item()/2
    
    return selected_indices, max_determinant, final_submatrix

def stochastic_greedy_maxlogdet(matrix, k, eps=0.2, lancoz=False):
    """
    Stochastically selects k rows and columns from the input matrix to maximize the determinant.
    
    Args:
    matrix (torch.Tensor): NxN input matrix
    k (int): Size of the submatrix to select
    eps (int): Measure of how close to the optimal solution we want to be
    
    Returns:
    tuple: (selected_indices, max_determinant)
    """
    N = matrix.shape[0]
    if k > N:
        raise ValueError("k cannot be larger than the matrix size")
    
    # using formula from Lazier than Greedy
    n_samples = int((N / k) * math.log(1/eps))

    # Initialize the list of selected indices
    selected_indices = []
    
    for _ in range(k):
        max_det = float('-inf')
        best_index = -1

        # take random subsample (of n_samples) from the remaining indices
        remaining_indices = [i for i in range(N) if i not in selected_indices]
        subsample = np.random.choice(remaining_indices, n_samples, replace=False)

        # Try adding each index from subsample and calculate the determinant
        for i in subsample:
            current_indices = selected_indices + [i]
            submatrix = matrix[current_indices][:, current_indices]
            
            if lancoz:
                det = gpt.inv_quad_logdet(submatrix, logdet=True, reduce_inv_quad=False)[1].item()
            else:
                det = torch.logdet(submatrix).item()

            # Update if we found a better determinant
            if det > max_det:
                max_det = det
                best_index = i
        
        # Add the best index found in this iteration
        selected_indices.append(best_index)
    
    # Calculate the final determinant
    final_submatrix = matrix[selected_indices][:, selected_indices]
    if lancoz:
        max_determinant = gpt.inv_quad_logdet(final_submatrix, logdet=True, reduce_inv_quad=False)[1].item()/2
    else:
        max_determinant = torch.logdet(final_submatrix).item()/2

    return selected_indices, max_determinant, final_submatrix