# Implementing Linear regression and Logistic regression from scratch

## Linear Regression
1. The input is usually a p dimensional vector suppose X
2. Here the p dimensions can be thought of as the features of the input 
3. The output as well as the input are both drawn from a real number space
4. GOAL 1: learn a function f(x) that maps inputs to outputs
5. In linear regression we assume the data has a linear pattern
6. GOAL 2: find a function f(x) such that loss function is minimized
7. A good loss function is MSE (Mean Squared Error) 
        E(Yi - f(Xi))**2
8. The best solution for this is given as E[y|x] [ when mse is minimised]
9. We approximate E[y|x] with a linear function
10. In the implementation , gradient descent is used to update the weights and bias
11. we are optimizing mse here
12. The input was taken in the form of a vector 


## Logistic Regression


