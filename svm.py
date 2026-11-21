import numpy as np 
import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.datasets import load_iris

iris_df = load_iris(as_frame=True)

# Access the full dataframe (features + target)
df = iris_df.frame

df.sample(n=9)
df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()
# df.info()

df["target"] = df["target"].astype("bool")

X = df.drop(columns=["target"])
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.20)

Lsvm = LinearSVC(max_iter=1000, random_state=42)

Lsvm.fit(x_train, y_train)

# on training data
y_pred_train = Lsvm.predict(x_train)
print(accuracy_score(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))
print(confusion_matrix(y_train, y_pred_train))

# on testing data 
y_pred = Lsvm.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))




