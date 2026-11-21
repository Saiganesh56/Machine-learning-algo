import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("C:/Users/Admin/Downloads/car+evaluation/car.data")
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safty", "class"]

df.drop("maint", axis=1, inplace= True)
df.describe(include="all").T
# for col in df.columns:
#         print(df[col].value_counts())

df1 = df[~df.duplicated()]
df1.shape
df1.columns

X = df1.drop(columns=["class"])
y = df1["class"]

oe = OrdinalEncoder()
le = LabelEncoder()
X = oe.fit_transform(X)
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)  # type: ignore

rf = RandomForestClassifier(n_estimators=20, random_state=42)

rf.fit(x_train, y_train)
y_pred = rf.predict(x_train)

def evaluation(ground_truth, prediction):
        ac_score = accuracy_score(ground_truth, prediction)
        cl_re = classification_report(ground_truth, prediction)
        con_fu = confusion_matrix(ground_truth, prediction)
        return {"accuracy_score":ac_score, "classification_report":cl_re, "confusion_matrix":con_fu}       

evaluation(y_train, y_pred)



        





