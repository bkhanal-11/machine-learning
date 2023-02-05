# Import dependecies
import numpy as np

class LinearRegression:
    def __init__(self):
        self.theta_1 = None
        self.theta_2 = None
    
    def initialize_theta(self):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        """
        theta_1 = 0.0
        theta_2 = 0.0
        
        return theta_1, theta_2
    
    def propagate(self, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above
        """
        
        m = X.shape[0]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        Z = self.theta_2 * X + self.theta_1
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dtheta_1 = np.sum(Z - Y) / m
        dtheta_2 = np.dot(X , (Z - Y)) / m

        grads = {"dtheta_1": dtheta_1,
                "dtheta_2": dtheta_2}
        
        return grads
    
    def fit(self, X, Y, num_iterations=1000, learning_rate=0.0005):
        """
        This function optimizes w and b by running a gradient descent algorithm
        """
        self.theta_1, self.theta_2 = self.initialize_theta()
        
        for i in range(num_iterations):
            # Cost and gradient calculation 
            grads = self.propagate(X, Y)
            
            # Retrieve derivatives from grads
            dtheta_1 = grads["dtheta_1"]
            dtheta_2 = grads["dtheta_2"]
            
            # update rule (gradient descent)
            self.theta_1 = self.theta_1 - learning_rate * dtheta_1
            self.theta_2 = self.theta_2 - learning_rate * dtheta_2
            
    
    def predict(self, X):
        '''
        Predict the output given theta_1 and theta_2
        '''        
        Z = self.theta_2 * X + self.theta_1
        
        return Z
