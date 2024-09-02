import torch

def softmax_jacobian(jacobian_logit, pi):
    '''
    Function to compute the jacobian of the softmaxed likelihood of the model
    '''
    jacobian_softmax = torch.diag(pi) - torch.outer(pi, pi)
    res = jacobian_softmax @ jacobian_logit
    return res

def H_p(model, x):
    '''
    Function to compute the Hessian of the log likelihood of the model
    '''
    pi = model(x)
    J, _ = model.backend.last_layer_jacobians(x)
    H_p = torch.zeros(J.shape[0], J.shape[1], J.shape[2])
    for j in range(J.shape[0]):
        H_p[j] = -(1/pi[j].unsqueeze(-1)) * softmax_jacobian(J[j], pi[j])
    return H_p

def compute_S(model, x, pseudo_label='soft_label'):
    '''
    Function to compute the S matrix
    '''
    hp = H_p(model, x)
    probs = model(x)
    if pseudo_label == 'hard_label':
        # pick the class with the highest probability
        label = probs.argmax(dim=1)
    elif pseudo_label == 'soft_label':
        # sample from the predicted probabilities
        label = torch.distributions.Categorical(probs).sample().to('cpu')

    device = hp.device
    print('HP DEVICE:', device)
    print('Precision DEVICE:', model.posterior_precision.to_matrix().device)
    precision = model.posterior_precision.to_matrix().to(device)
    indices = torch.arange(hp.shape[0], device=device)

    hp_hat = hp[indices, label].view(hp.shape[0], -1)  # (batch, num_parameters)
    S = hp_hat @ precision.inverse() @ hp_hat.T
    return S  