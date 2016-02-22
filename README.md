# Hierarchical-clustering-KNN


+ In the *First step*,Agglomerative hierarchical clustering is used on the UCI seed dataset to make clusters.
+ User is asked to enter the number of clusters and one of three agglomerative clustering algorithms to use (Single, Complete or Average) + The data has been divided into training and test datasets by sampling without replacement.
+ Then in *Second part*, the clusters are considered as nodes (the average of all data points in that cluster represent that node).
+ Using KNN for test-data, the cluster to which this data belongs is predicted.
