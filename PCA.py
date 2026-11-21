from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = load_wine(as_frame=True) 
X = df.data
y = df.target

scalar = StandardScaler()

X_scaled = scalar.fit_transform(X)

pca = PCA(n_components=2)

x_dim= pca.fit_transform(X_scaled)

