from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import kagglehub
import os
import pandas as pd 

# Download latest version
path = kagglehub.dataset_download("ahmedesso/brest-cancer")

df = pd.read_csv(os.path.join(path,'data.csv'))
df["diagnosis"] = df["diagnosis"].replace(["M","B"],[1,0]).astype(int)

df.isna().sum()
df.drop(columns=["Unnamed: 32" , "id"],inplace=True)
df.sample(n=10)

# target feature split
X = df.drop(columns=["diagnosis"],inplace=False)
y = df.diagnosis


# normalisation || scalling

scalar = StandardScaler()

X_scaled = scalar.fit_transform(X)

# test_train_split

x_train ,  x_test , y_train , y_test = train_test_split(X_scaled , y, test_size=0.20 , stratify=y, random_state= 42)

# model biulding
cancer_detection = LogisticRegression(random_state=42)

cancer_detection.fit(x_train , y_train)
y_pred = cancer_detection.predict(x_test)

accuracy = accuracy_score(y_test , y_pred)

print(accuracy)

report = classification_report(y_test , y_pred)

print( report )






