import numpy as np 
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report 

df = pd.read_csv("C:/Users/Admin/Desktop/(Copy) diabetes.csv")

df.head()

X = df.drop(columns=["Outcome"]).to_numpy()
Y = df["Outcome"].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.20, random_state= 42, stratify= Y) # type: ignore

print(x_train.shape , x_test.shape, y_train.shape, y_test.shape) 

class KNearest_Nieghbor():
        def __init__(self) :
                pass
        
        def get_distance(self,x_train, x_test):
                
                distance = np.sum((x_train - x_test)**2)
                
                return np.sqrt(distance)
        
        def get_nearest_nieghbor(self, x_train, x_test, y_train, K):
                
                nieghbores = []
                
                for idx , row in enumerate(x_train) :
                        distance = self.get_distance(row,x_test)
                        nieghbores.append((y_train[idx], distance))
                
                nieghbores.sort(key=lambda x : x[1])
                
                nearest = []
                
                for i in range(K):
                        nearest.append(nieghbores[i][0])
                return nearest
                   
        def predict(self, x_train, x_test, y_train, K):
                
                nieghbors = self.get_nearest_nieghbor(x_train, x_test, y_train, K)
                
                pred = statistics.mode(nieghbors)
                
                return pred
if __name__ == "__main__":
        classifier = KNearest_Nieghbor()
        prediction = np.array([])
        for idx , row in enumerate(x_test):
                prediction=np.append(prediction , [classifier.predict(x_train, x_test[idx], y_train, 9)])
        print("Accurasy of the model is : ", np.mean(np.array(prediction) == y_test))
        print(" confusion matrics : \n", confusion_matrix(np.array(prediction),y_test))
        print(" classification report : \n", classification_report(np.array(prediction), y_test))
                