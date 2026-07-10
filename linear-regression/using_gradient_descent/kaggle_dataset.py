import kagglehub 
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
import os

# importing dataset
path = kagglehub.dataset_download("jemishdonda/headbrain")
df = pd.read_csv(os.path.join(path,"headbrain.csv"))

df.columns = df.columns.str.replace(" ","_").str.lower()

# feature selection
df = df.drop(columns=["gender","age_range"])

# dataset splitting in X (features) , Y (target)
X = df["head_size(cm^3)"].values
Y = df["brain_weight(grams)"].values

# train test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20) 

# standardisation 
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train.reshape(-1, 1)).flatten()
x_test = scalar.transform(x_test.reshape(-1, 1)).flatten()

# random values initialisation for weights(m) and bias(b)
m= torch.tensor([0.9],requires_grad=True)
b= torch.tensor([0.1],requires_grad=True)

# y= m*x+b
# hyper parameters 
learning_rate = 0.01
epochs = 1000

n = x_train.shape[0]

# step 1 forward pass
def regression(x, m, b):
        return x*m+b
# step 2 calc loss or cost 
def mse(y, y_hat):
        loss = torch.sum((y-y_hat)**2)
        return loss / n
#step -3 mse
# Auto diff of C cost function
# step -4 optimization of the model
optimiser = torch.optim.SGD([m,b], lr=learning_rate)
# training of a model 

# r2 = 1 - ss_res / ss_tot
def r2_eval(y, y_hat):
       with torch.no_grad():
        r2_01 = 1- (torch.sum((y - y_hat)**2) / torch.sum((y-torch.mean(y))**2))
        r2_1 = r2_score(y, y_hat)
        print(f"r2_1: {r2_1}, r2_01: {r2_01}")
        
        
def training_gradient_model(x, y, m, b, epochs):
        for epoch in range(epochs):
                optimiser.zero_grad()
                #step -1 
                y_hat = regression(x, m, b)
                #step -2 
                C = mse(y, y_hat)
                #step -3
                C.backward()
                #step -4 
                optimiser.step()
                if epoch == 0 or epoch == 499 or epoch ==999 :
                        print(f"cost or loss: {C.item()} , epoch: {epoch+1} , wieghts: {m.item()} , bias: {b.item()}")
                        # plotting(x, y, y_hat, epoch, C, m, b)
        return m , b

def plotting(x, y, y_hat, epoch, C, m, b):
        plt.clf()
        plt.title(f"'Gradient descent' epoch: {epoch}, loss: {C}, weights: {m}, bias:{b}")
        plt.xlabel(" x label")
        plt.ylabel("y label")
        plt.scatter(x, y, c= "blue")
        plt.plot(x, y_hat.detach().numpy())
        plt.grid(alpha= 0.30)
        plt.show()
        

def predict(x_test, m, b):
        y_pred = m * x_test + b
        return y_pred.detach().numpy()

if __name__ == "__main__":
        x_train = torch.tensor(x_train,dtype=torch.float)
        y_train = torch.tensor(y_train,dtype=torch.float)
        x_test = torch.tensor(x_test,dtype=torch.float)
        y_test = torch.tensor(y_test,dtype=torch.float)
        m , b = training_gradient_model(x_train,y_train , m, b, epochs)
        r2_eval(y_test,predict(x_test,m ,b))
        
        


