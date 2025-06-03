import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


print("Setup complete")

#Linking files
df = pd.read_csv("Mall_Customers (1).csv")

#Checking for missing data
print(df.head())
print(df.isnull().sum())

#Making xx use annual income and spending score columns
xx = df.iloc[:, [3,4]].values
print(xx)

#Finding optimal number of clusters based on ELBOW METHOD
wwcss = []
for i in range(1,11):
    kmean = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmean.fit(xx)
    
    wwcss.append(kmean.inertia_)

# plt.plot(range(1,11), wcss)
# plt.title("Elbow Graph")
# plt.xlabel("Centroids")
# plt.ylabel("wcss")
#plt.show()

#Last elbow on the graph landed on numer 5, so 5 is the optiman number of clusters. Making the clustering algorithm.
#When initializing centroids, make sure you assign them to the same name as in the elbow method to override their original number which is 10. If not you will have 10 centroids in your final output.
kmean = KMeans(n_clusters=5, init='k-means++', random_state=0)
ymodel = kmean.fit_predict(xx)
print(ymodel)

#Using matpilotlib to plot the algorithm's output.
plt.scatter(xx[ymodel == 0,0], xx[ymodel == 0,1], s=50, c="orange", label="Cluster 1")
plt.scatter(xx[ymodel == 1,0], xx[ymodel == 1,1], s=50, c="green", label="Cluster 2")
plt.scatter(xx[ymodel == 2,0], xx[ymodel == 2,1], s=50, c="yellow", label="Cluster 3")
plt.scatter(xx[ymodel == 3,0], xx[ymodel == 3,1], s=50, c="cyan", label="Cluster 4")
plt.scatter(xx[ymodel == 4,0], xx[ymodel == 4,1], s=50, c="blue", label="Cluster 5")
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s=75, c="red", label="centroids")

plt.title("Customer segmentation")
plt.ylabel("Spending Score")
plt.xlabel("Annual Income")
plt.legend()
plt.show()

#On the output you can see cluster 2 has high income and a high spending score. From here we would begin sending out emails to these customers for promos and events so they are more likely to come.