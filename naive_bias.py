import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , accuracy_score, classification_report


df = load_breast_cancer(as_frame = True)
df = df.frame

df.head()
df.columns
df.describe()
# df.info()
df["target"] = df["target"].astype("bool")


X = df.drop(columns = ["target"])
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state= 42, test_size=0.20)

mn_NB = MultinomialNB()

mn_NB.fit(x_train, y_train)

y_tain_pred = mn_NB.predict(x_train)

print(accuracy_score(y_tain_pred, y_train ))
print(classification_report(y_tain_pred, y_train ))
print(confusion_matrix(y_tain_pred, y_train ))

