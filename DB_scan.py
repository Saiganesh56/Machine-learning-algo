import pandas as pd 
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score



df1 = load_wine(as_frame=True)
X = df1.data
y =  df1.target 
df = pd.DataFrame(X, columns=df1.feature_names)
df["wine_classes"] = y

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
X.shape

model_db = DBSCAN(eps=3, min_samples = 3, metric="euclidean", algorithm="auto" )
lables_db = model_db.fit_predict(X_scaled)

# basic math statistics 

no_of_clusters = len(set(lables_db)) - 1 if -1 in lables_db else 0
no_noise = (lables_db == -1 ).sum()

print(" numbers of clusters :",no_of_clusters)
print("noise in the datasets : ",no_noise)
print("clusters distribution : ", np.bincount(lables_db[lables_db >= 0]))

score = silhouette_score(X_scaled[lables_db != -1],lables_db[lables_db != -1 ])
print(score)