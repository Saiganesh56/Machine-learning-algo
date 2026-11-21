import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
 
data = load_wine()
X = data.data
y = data.target
df = pd.DataFrame(X, columns = data.feature_names)
df["wine_class"] = y
df.head()
df.isna().sum()

scalar = StandardScaler()
X = scalar.fit_transform(X)

wss = []
for i in range(1, 11):
        centroid = KMeans(n_clusters=i,init="k-means++", random_state=42)
        centroid.fit(X)
        wss.append(centroid.inertia_)


fig,  ax= plt.subplots(figsize = (8,6))
ax.plot(range(1,11,1),wss)
plt.title(" elbow method ")
plt.xlabel(" the sample number of k values ")
plt.ylabel("wss within sum of square errors ")
plt.show()

#  obtainig of labels 
# k = 3 
k_means = KMeans(n_clusters=3,init="k-means++",random_state=42)
k_means.fit(X)
labels = k_means.labels_

#  evalution of the model in two different ways 
rand_score = adjusted_rand_score(y, labels) 
print(rand_score)

silhouette = silhouette_score(X, labels)
print(silhouette)





