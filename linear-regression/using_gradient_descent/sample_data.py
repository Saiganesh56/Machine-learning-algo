from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch

# BATCH GRADIENT DESCENT
# y = mx+b
# Example data 
x1 = [1, 2, 3, 4, 5]
y1 = [2, 4, 5, 4, 5]

n = len(x1)
x = torch.tensor(x1,dtype=torch.float)
y = torch.tensor(y1,dtype=torch.float)

# assigning random values to m and b 
#                      where,
#                               m = slope 
#                               b = y-intercept 

m = torch.tensor([0.9],requires_grad=True)
b = torch.tensor([0.1],requires_grad=True)

# there are majorly four step involved in finding the gradient 
# 1. forward pass 
# 2. c = y-y_pred ( finding cost or error )
# 3. finding gradients of c using autodiff 
# 4. gradient descent

def regression(x,m, b):
        return x * m + b

def mse(y, y_hat): # this is mse mean squared error 
        loss = torch.sum((y-y_hat)**2)
        return loss/n

# hyperparameters 
learning_rate = 0.01
epochs = 1000

# 1 forword pass
y_hat = regression(x, m, b)

# 2 finding cost or loss
C = mse(y, y_hat)

# 3 finding of gradients 
C.backward()

# 4 gradient descent  updating parameters
optimiser = torch.optim.SGD([m,b],lr=learning_rate)


def training_of_model(x, y, y_hat, m, b, epochs):
        
        for epoch in range(epochs):
                
                optimiser.zero_grad()
                
                # step -1 forward-pass
                y_hat = regression(x, m, b)
                
                #step - 2
                C = mse(y,y_hat)
                
                #step - 3 
                C.backward() 
                
                if epoch % 100==0:
                        plotting_of_gradients(x, y, y_hat, C, epoch)
                        print(
                                f"Epoch {epoch:4d} | "
                                f"Loss={C.item():.4f} | "
                                f"m={m.item():.4f} | "
                                f"b={b.item():.4f}" )
                 
                # step - 4
                optimiser.step()
        r2_eval(y,regression(x, m, b))
        print("\nTraining Complete")
        print(f"Final Weight(m) : {m.item():.4f}")
        print(f"Final Bias(b)  : {b.item():.4f}")
                
# r2 = 1 - ss_res / ss_tot
def r2_eval(y, y_hat):
        r2_01 = 1- (torch.sum((y - y_hat)**2) / torch.sum((y-torch.mean(y))**2))
        r2_1 = r2_score(y.detach().numpy(), y_hat.detach().numpy())
        print(f"r2_1: {r2_1}, r2_01: {r2_01}")

# plotting in graph
def plotting_of_gradients(x, y, y_hat, C, epoch):
        plt.clf()
        plt.title(f"'gradient descent', 'Cost: '{C}, 'epoch: '{epoch}")
        
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.scatter(x.numpy(),y.numpy(),c="blue")
        plt.plot(x,y_hat.detach().numpy(), color="red")
        plt.show()
        
        
          
        
if __name__ == "__main__":
        training_of_model(x, y, y_hat, m, b, epochs)


        