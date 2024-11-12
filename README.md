K-means Clustering on the Iris Dataset
This repository demonstrates the application of the K-means clustering algorithm on the famous Iris dataset, one of the most commonly used datasets in machine learning. By utilizing K-means, the project clusters the dataset into groups based on petal and sepal measurements, aiming to distinguish between the three species of iris flowers (Setosa, Versicolor, and Virginica).

Features
Data Loading and Preprocessing: Loads the Iris dataset and performs any necessary preprocessing, such as scaling or normalization, to prepare the data for clustering.
K-means Clustering: Implements K-means clustering to segment the dataset into three clusters, with k=3 corresponding to the three flower species. The algorithm iteratively adjusts cluster centroids and assigns data points to the nearest centroid to minimize within-cluster variance.
Cluster Boundary Visualization: After clustering, the repository includes visualization functions that plot the data points in two-dimensional space, using colors to indicate cluster assignments and displaying the boundaries around each cluster. It provides a clear visual of how the K-means model differentiates between species.

How to Use
Clone the repository and install the necessary dependencies.
Edit output.py with different, unmcommenting different lines to graph and visualize the cluster boundaries.
Experiment with different numbers of clusters or model parameters to explore the clustering behavior.
