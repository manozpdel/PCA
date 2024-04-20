import numpy as np
import pandas as pd

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Convert DataFrame to numpy array if input is DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # 1. Data Preprocessing
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # 2. Compute the Covariance Matrix
        covariance_matrix = np.cov(X.T)
        
        # 3. Compute the Eigenvectors and Eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # 4. Select Principal Components
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]
        
    def transform(self, X):
        # Convert DataFrame to numpy array if input is DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Project data onto the new feature space
        X = X - self.mean
        return np.dot(X, self.components)
