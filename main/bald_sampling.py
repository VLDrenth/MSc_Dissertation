import torch

def compute_entropy(la_model, data):
    # for each datapoint compute the probabilties of each class
    p = la_model(data, pred_type='glm', link_approx='probit')  # (n_data, n_classes)
    entropy = _h(p)  

    return entropy

def _h(p):
    # p is a tensor of shape (n_data, n_classes)
    return -torch.sum(p * torch.log(p + 1e-12), dim=1)

def compute_entropy_weights(la_model, data, n_samples=50):
    # Sample from the posterior
    posterior_weights = la_model.sample(n_samples=n_samples)
    probs = torch.zeros(n_samples, data.shape[0], 10)

    # Compute the entropy for each sample
    for i, weights in enumerate(posterior_weights):
        # Set the weights in the model
        if la_model.backend.last_layer:
            set_last_linear_layer_combined(la_model.model, weights)
        else:
            set_full_parameters(la_model.model, weights)

        # Compute the predictive distribution
        probs[i] = la_model(data, pred_type='glm', link_approx='probit')
    
    # Compute the entropy
    entropies = _h(probs.mean(dim=0))
    return entropies

def set_full_parameters(model, new_weights):
    new_weights = new_weights.flatten()
    total_params = sum(p.numel() for p in model.parameters())
    
    if total_params != len(new_weights):
        raise ValueError(f"Number of weights ({len(new_weights)}) does not match the model's parameter count ({total_params})")
    
    i = 0
    for param in model.parameters():
        n = param.numel()
        param.data.copy_(new_weights[i:i+n].reshape(param.shape))
        i += n
    
    return model  # Return the updated model    


def set_last_linear_layer_combined(model, new_weights_and_bias):
    # Find the last linear layer
    last_linear_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module
    
    if last_linear_layer is None:
        raise ValueError("No linear layer found in the model")

    # Get the shapes
    out_features, in_features = last_linear_layer.weight.shape
    
    # Check if the input tensor has the correct shape
    expected_shape = (out_features * in_features + out_features,)
    if new_weights_and_bias.shape != expected_shape:
        raise ValueError(f"Input tensor shape {new_weights_and_bias.shape} doesn't match the expected shape {expected_shape}")

    # Split the input tensor into weights and bias
    new_weights = new_weights_and_bias[:out_features * in_features].reshape(out_features, in_features)
    new_bias = new_weights_and_bias[out_features * in_features:]

    # Set new weights and bias
    last_linear_layer.weight.data = new_weights
    last_linear_layer.bias.data = new_bias

    return last_linear_layer

def compute_conditional_entropy(la_model, data, train_loader, refit=True, n_samples=50):
    # Sample from the posterior
    posterior_weights = la_model.sample(n_samples=n_samples)
    entropies = torch.zeros(posterior_weights.shape[0], data.shape[0])

    # Compute the entropy for each sample
    for i, weights in enumerate(posterior_weights):
        # Set the weights in the model
        if la_model.backend.last_layer:
            set_last_linear_layer_combined(la_model.model, weights)
        else:
            raise NotImplemented 
            set_full_parameters(la_model.model, weights)

        if refit:
            # fit the model
            la_model.fit(train_loader)

            # Optimise the prior precision
            la_model.optimize_prior_precision(pred_type='glm', method='marglik', link_approx='probit', verbose=False)

        # Compute the predictive distribution
        probs = la_model(data, pred_type='glm', link_approx='probit')

        # Compute the entropy
        entropies[i] = _h(probs)

    return entropies.mean(dim=0)

def compute_bald(la_model, data, train_loader, refit=True, n_samples=50):
    # Compute the entropy
    entropy = compute_entropy(la_model, data)

    # Compute the conditional entropy
    cond_entropy = compute_conditional_entropy(la_model, data, train_loader, n_samples=n_samples, refit=refit)

    bald = entropy - cond_entropy

    return bald


if __name__ == '__main__':
    pass


