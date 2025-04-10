import numpy as np

# define a class
class linearRegression:
    def __init__(self,learning_rate = 0.001,n_iterations = 1000):
        
        # initialise learning rate , iterations , weights , bias
        self.learning_rate = learning_rate
        self.n_iteration = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X , y):
        # the arguments are supposed to be the training sets prepared

        # get number of samples and features from x using shape

        n_samples , n_features = X.shape

        # initialise weights and biases to zeros

        self.weights = np.zeros(n_features)
        self.bias = 0

        # run the gradient descent updation in a for loop for the number of iterations

        for _ in range(self.n_iteration):

            # y = wX + b 
            y_pred = np.dot(X,self.weights) + self.bias

            # dw = (1/n) * summation(x * (yhat - y))
            dw = (1/n_samples ) * np.dot(X.T , (y_pred-y))
            # db = (1/n) * summation((yhat - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # updating the weights and biases
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self,X):

        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred
    




