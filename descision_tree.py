import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score , classification_report , f1_score, confusion_matrix


df = pd.read_csv("C:/Users/Admin/Desktop/snippets/Play Tennis.csv")

df.shape
df.columns = df.columns.str.replace(" ","_").str.lower().str.strip()

columns = df.columns

for i in columns :
        if df[i].dtype == "O":
                le = LabelEncoder()
                df[i] = le.fit_transform(df[i]) # type: ignore



X = df.drop(columns=["play_tennis"]) 
y = df["play_tennis"] 

             
x_train, x_test, y_train, y_test  = train_test_split(X, y ,test_size=0.20, random_state=42)

dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
plt.figure(figsize=(15, 10))
plot_tree(
    dt,
    filled=True,
    rounded=True,
    fontsize=12,
)

plt.show()




