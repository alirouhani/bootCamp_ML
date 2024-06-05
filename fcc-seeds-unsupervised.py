import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ============== Formatting Model : Start  ===================
cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("seeds_dataset.txt", names = cols, sep="\s+")
# for i in range(len(cols)-1):
#     for j in range(i+1, len(cols)-1):
#         x_label = cols[i]
#         y_label = cols[j]
#         sns.scatterplot(x = x_label, y=y_label, data=df, hue="class")
#         plt.show()
# ============== Formatting Model : End  ===================
# ============== K-means Clustering  ===================
# How do it? Expectation-Maximization
# 1 => choose k
# 2 => choose k random points to be centroid (maybe the points are not from our dataset)
# 3 => distance between each point and those centroid points
# 4 => assigning ponits to the closest centroid (The point is maybe the final clusters are less than what we set as the K)
# 5 => compute new centroid (are not from the dataset)
# 6 => do the process until we get the same cluster => we reach to a stable state
from sklearn.cluster import KMeans
x = "perimeter" # we can change it to more messy one like compactness
y = "asymmetry"
X = df[[x,y]].values
kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.labels_ # the ouput of clustering
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1,1))), columns = [x, y, "class"]) #kmeans classes
# sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
# plt.show()
# sns.scatterplot(x=x, y=y, hue='class', data=df)
# plt.show()
# Higer dimension
X = df[cols[:-1]].values
kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.labels_ # the ouput of clustering
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=df.columns)
# sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
# plt.show()
# sns.scatterplot(x=x, y=y, hue='class', data=df)
# plt.show()
# ============== Principal Component Analysis (dimentionally reduction)  ===================
# Data are unlabeled to extract more dimension
# Let's say we want to plot our data, and we don't have too axis to show it on.
# How we can diplay the data without deleting information?
# find the direction in space with largest variance
# we want to project our data to less dimension => for example: map 2D data into 1D point 
# largest variance: give us the most discreemention between the points => minimum the residuals (2d point - mppped 1d point)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
transformed_x = pca.fit_transform(X)
# print(transformed_x.shape)
# plt.scatter(transformed_x[:,0], transformed_x[:,1])
# plt.show()
kmeans_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1, 1))), columns=["pca1", "pca2", "class"])
truth_pca_df = pd.DataFrame(np.hstack((transformed_x, df["class"].values.reshape(-1, 1))), columns=["pca1", "pca2", "class"])
sns.scatterplot(x="pca1", y="pca2", hue='class', data=kmeans_pca_df)
plt.show()
sns.scatterplot(x="pca1", y="pca2", hue='class', data=truth_pca_df)
plt.show()

