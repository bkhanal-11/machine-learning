import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self, num_pc):
        """
        Calculates principal components of 
        a multidimensional input matrix.
        """
        self.num_pc = num_pc

    def normalize_data(self, df):
        """
        Normalizes the dataframe,
        returns centered training data.
        """
        return (df - df.mean()) / df.std()
    
    def covariance_matrix(self, X):
        """
        Computes covariance matrix of 
        centered training data matrix.
        """
        features = X.T
        cov_matrix = np.cov(features)
        
        return cov_matrix

    def svd(self, A):
        """
        Performs Singular Value Decomposition (SVD) to
        find eigen values and vectors of a covariance matrix. 
        """
        self.eig_values, self.eig_vectors = np.linalg.eig(A)
    
    def compress(self, X):
        """
        Calculates principal components of a matrix.
        """
        principal_components = []

        for i in range(0, self.num_pc):
            princ_comp = X.dot(self.eig_vectors.T[i])
            principal_components.append(princ_comp)

        return principal_components
    