from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error , root_mean_squared_error
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np  
import kagglehub 
import os

# importing synthetic datasets of  
path=kagglehub.dataset_download("jemishdonda/headbrain")
# print(os.listdir(path))
df = pd.read_csv(os.path.join(path,"headbrain.csv"))

df.columns = df.columns.str.replace(" ","_").str.lower()
df.head()

# these to columns are correlated rest are all outer scope 
plt.scatter(df["brain_weight(grams)"],df["head_size(cm^3)"])

# extracting features form dataset  [head_size(cm^3)]
X = df.drop(columns=["gender","age_range" ,"brain_weight(grams)"]).values
y = df.drop(columns=["gender","age_range","head_size(cm^3)"]).values

# test ,  train splipts
x_train , x_test , y_train , y_test = train_test_split(X , y, test_size=0.2 , random_state=42)

# training of a model
lr=LinearRegression()
lr.fit(x_train , y_train)

y_predictions = lr.predict(x_test)

# give an score bt 0 to 1 based on how good the model is [ colser ot 1 ]
r2_score(y_test , y_predictions) 

# absolute residual cumulative sum bt actual y and predicted y 
mean_absolute_error(y_test , y_predictions)

# squared the distance even the small error makes larger value
mean_squared_error(y_test , y_predictions)
# actual error like square root for MSE
root_mean_squared_error(y_test , y_predictions)

# predicting random wieght over random sixe
y_random_pred = lr.predict([[4177]])  # type: ignore






