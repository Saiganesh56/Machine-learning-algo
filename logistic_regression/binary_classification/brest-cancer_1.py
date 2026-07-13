import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import os 

# Download latest version
path = kagglehub.dataset_download("ahmedesso/brest-cancer")

df = pd.read_csv(os.path.join(path,'data.csv'))
df["diagnosis"] = df["diagnosis"].replace(["M","B"],[1,0]).astype(int)
# basic data preprossecing
df.isna().sum()
df.drop(columns=["Unnamed: 32" , "id"],inplace=True)
df.sample(n=10)

# target columns and features  splitting
X = df.drop(columns=["diagnosis"],inplace=False)
y = df["diagnosis"].values

# normalising of features to overcome overfitting 
scalar  = StandardScaler()
X_scaled = scalar.fit_transform(X)

# train_test data splitting 
x_train , x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20)

# activation funciton for logistic regressing 
def sigmoid(z):
        return 1/(1 + np.exp(-z))

# fingding the cost , loss 
def cost_function(X, y, w, b):
        m = X.shape[0]
        cost_sum = 0
        for i in range(m):
                z = np.dot(w, X[i]) + b # finding linearity 
                g = sigmoid(z) # fingding the probability 
                cost_sum += -y[i]*np.log(g) - (1 - y[i]) *np.log(1 - g) # calculating the cost, loss 
                
#  cost = (-1 / m) * np.sum(y * np.log(g + 1e-15) + (1 - y) * np.log(1 - g + 1e-15)) return cost
        
        return (1/m)*cost_sum # cross entropy loss funcion
# finding gradiets or partial derivatives of cost with respect to  parameters + b
def gradient_function(X, y, w, b): # 
    m, n = X.shape
    z = np.dot(X, w) + b
    g = sigmoid(z)
    
    
    err = g - y # Error vector
    
    
    grad_b = (1 / m) * np.sum(err) # Vectorized gradient calculation
    grad_w = (1 / m) * np.dot(X.T, err)
    
    return grad_b, grad_w

# descending that graadients or optimising parameteres 
def gradient_descent_function(X, y, learning_rate, epochs):
        m,n = X.shape 
        w = np.zeros(n)
        b = 0
        
        for i in range(epochs):
                grad_b, grad_w = gradient_function(X, y, w, b)
                w = w - learning_rate * grad_w
                b = b - learning_rate * grad_b
                if i % 100 == 0 :
                        print(f" iteration {i} cost {cost_function(X, y, w, b )}")
        return w , b
# prediction of the data || final resulting boolean values 
def predict(X, w, b):
        m,n = X.shape
        preds = np.zeros(m)
        z = np.dot(X,w) + b
        g = sigmoid(z)
        return (g > 0.5).astype(int)


if __name__ == "__main__":
        epochs = 1000
        learning_rate = 0.01
        final_w, final_b = gradient_descent_function(x_train , y_train, learning_rate, epochs) 
        # trainig accuracy 
        predictions = predict(x_train, final_w, final_b)
        # evaluation
        training_accuracy = np.mean(predictions == y_train)*100
        print(f" training accuracy {training_accuracy}")
        # testing accuracy
        predictions = predict(x_test, final_w, final_b)
        # evaluation
        testing_accuracy = np.mean(predictions == y_test)*100
        print(f" testing accuracy {testing_accuracy}")
        
        
        
        
                