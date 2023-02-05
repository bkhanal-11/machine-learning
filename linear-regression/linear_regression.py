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
        
        m = X.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        Z = self.theta_2 * X + self.theta_1
        cost = (1 / m) * np.sum(Z - Y) 
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dtheta_1 = np.sum(Z - Y) / m
        dtheta_2 = 1 / m
        
        cost = np.squeeze(np.array(cost))

        grads = {"dtheta_1": dtheta_1,
                "dtheta_2": dtheta_2}
        
        return grads, cost
    
    def fit(self, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        """
        costs = []
        self.theta_1, self.theta_2 = self.initialize_theta()
        
        for i in range(num_iterations):
            # Cost and gradient calculation 
            grads, cost = self.propagate(X, Y)
            
            # Retrieve derivatives from grads
            dtheta_1 = grads["dtheta_1"]
            dtheta_2 = grads["dtheta_2"]
            
            # update rule (gradient descent)
            self.theta_1 = self.theta_1 - learning_rate * dtheta_1
            self.theta_2 = self.theta_2 - learning_rate * dtheta_2
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
                # Print the cost every 100 training iterations
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"theta_1": self.theta_1,
                  "theta_2": self.theta_2}
        
        grads = {"dtheta_1": dtheta_1,
                "dtheta_2": dtheta_2}
        
        return params, grads, costs
    
    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        Z = self.theta_2 * X + self.theta_1
        
        return Z

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to True to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
        """  
        # initialize parameters with zeros 
        self.theta_1, self.theta_2 = self.initialize_theta()
        
        # Gradient descent 
        params, _, costs = self.fit(X_train, Y_train, num_iterations, learning_rate, print_cost)
        
        # Retrieve parameters w and b from dictionary "params"
        self.theta_1 = params["theta_1"]
        self.theta_2 = params["theta_2"]
        
        # Predict test/train set examples
        Y_prediction_test = self.predict(X_test)
        Y_prediction_train = self.predict(X_train)

        # Print train/test Errors
        if print_cost:
            print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        
        d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "theta_1" : self.theta_1, 
            "theta_2" : self.theta_2,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}
        
        return d