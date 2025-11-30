import numpy as np
from sklearn.metrics import silhouette_score
from typing import List

def calculate_silhouette_score(embeddings: np.ndarray, labels: List[str], sample_size: int = 10000) -> float:
    """
    Calculates the Silhouette Coefficient for the given embeddings and labels.
    
    Args:
        embeddings: Numpy array of embeddings.
        labels: List of cluster labels (Issue names).
        sample_size: Maximum number of samples to use for calculation (Silhouette is O(N^2)).
        
    Returns:
        float: The mean Silhouette Coefficient. Returns -1.0 if error or not enough classes.
    """
    if len(embeddings) < 2:
        print("Not enough samples to calculate Silhouette Score.")
        return -1.0
        
    # Filter out samples where label is None or empty
    valid_indices = [i for i, label in enumerate(labels) if label is not None and str(label).strip() != '']
    
    if len(valid_indices) < 2:
        print("Not enough valid labeled samples to calculate Silhouette Score.")
        return -1.0
        
    X = embeddings[valid_indices]
    y = [labels[i] for i in valid_indices]
    
    # Check if we have at least 2 unique labels
    if len(set(y)) < 2:
        print("Need at least 2 clusters (issues) to calculate Silhouette Score.")
        return -1.0
        
    # Downsample if too large
    if len(X) > sample_size:
        print(f"Downsampling for Silhouette Score calculation (max {sample_size})...")
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X[indices]
        y = [y[i] for i in indices]
        
    try:
        score = silhouette_score(X, y)
        return score
    except Exception as e:
        print(f"Error calculating Silhouette Score: {e}")
        return -1.0
