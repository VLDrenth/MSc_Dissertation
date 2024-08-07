import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

def badge_selection(model, pool_dataset, batch_size):
    """
    Implement the BADGE selection strategy for a single iteration.
    
    Args:
    - model: PyTorch model
    - pool_dataset: Dataset containing unlabeled examples (U \ S)
    - batch_size: Number of examples to select (B)
    
    Returns:
    - indices: Indices of selected examples to be added to S
    """
    
    model.eval()
    gradient_embeddings = []
    
    # Compute gradient embeddings for all examples in U \ S
    for idx, (x, _) in enumerate(pool_dataset):
        x = x.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        output = model(x)
        
        # Compute hypothetical label
        y_hat = output.argmax(dim=1)
        
        # Compute gradient embedding
        loss = F.cross_entropy(output, y_hat)
        
        # Compute gradients w.r.t. the last layer parameters
        last_layer = list(model.parameters())[-2:]
        last_layer = (y for y in last_layer)
        grad_embedding = torch.autograd.grad(loss, 
                                             last_layer, create_graph=False)[0]
        
        gradient_embeddings.append(grad_embedding.cpu().detach().numpy().flatten())
    
    # Convert to numpy array
    gradient_embeddings = np.array(gradient_embeddings)
    
    # Use k-MEANS++ to select diverse samples
    kmeans = KMeans(n_clusters=batch_size, init='k-means++', n_init=1, max_iter=1)
    kmeans.fit(gradient_embeddings)
    
    # Get the indices closest to the centroids
    distances = kmeans.transform(gradient_embeddings)
    selected_indices = np.argmin(distances, axis=0)
    
    return selected_indices