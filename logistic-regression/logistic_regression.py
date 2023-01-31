import numpy as np
import copy

class LogisticRegression:
    def __inti__(self):
        self.w = None
        self.b = None

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        """
        return 1 / (1 + np.exp(-z))

    def initialize_w_b(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        """
        w = np.zeros((dim,1))
        b = 0.0

        return w, b

    def propagate(self, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above
        """
        
        m = X.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = np.dot(X , (A - Y).T) / m
        db = np.sum(A - Y) / m
        
        cost = np.squeeze(np.array(cost))

        grads = {"dw": dw,
                "db": db}
        
        return grads, cost

    def optimize(self, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        """
        
        # w = copy.deepcopy(self.w)
        # b = copy.deepcopy(self.b)
        
        costs = []
        
        for i in range(num_iterations):
            # Cost and gradient calculation 
            grads, cost = self.propagate(X, Y)
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # update rule (gradient descent)
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
                # Print the cost every 100 training iterations
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": self.w,
                "b": self.b}
        
        grads = {"dw": dw,
                "db": db}
        
        return params, grads, costs

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        self.w = self.w.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        
        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0, i] > 0.5:
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0
        
        return Y_prediction

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
        self.w, self.b = self.initialize_w_b(X_train.shape[0])
        
        # Gradient descent 
        params, grads, costs = self.optimize(X_train, Y_train, num_iterations, learning_rate, print_cost)
        
        # Retrieve parameters w and b from dictionary "params"
        self.w = params["w"]
        self.b = params["b"]
        
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
            "w" : self.w, 
            "b" : self.b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}
        
        return d
