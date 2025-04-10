import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import linearRegression

# generation of training set and testing set
X , y = datasets.make_regression(n_samples=100,n_features=1,noise = 20 , random_state=4)
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state= 1234)

# initializing reg as the class linear regression
reg = linearRegression(learning_rate=0.01)

# using the function fit to train the data
reg.fit(X_train,y_train)

#using the function predict to predict using the test data
predictions = reg.predict(X_test)

#finding mse (mean squared error)
def mse(y_test , predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test , predictions)
print(mse)


#plotting the graph for the above regression
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,8))
m1 = plt.scatter(X_train,y_train,color = cmap(0.9), s =10)
m1 = plt.scatter(X_test,y_test,color = cmap(0.5), s =10)

plt.plot(X , y_pred_line , color = "black", linewidth=1 ,label = 'predictions')
plt.show()