# K-means Clustering on the Iris Dataset

This repository demonstrates the application of the **K-means clustering algorithm** on the famous Iris dataset, one of the most commonly used datasets in machine learning. Using K-means, the project clusters the dataset into groups based on petal and sepal measurements, aiming to distinguish between the three species of iris flowers: **Setosa**, **Versicolor**, and **Virginica**.

## Features

1. **Data Loading and Preprocessing**  
   - Loads the Iris dataset and performs necessary preprocessing steps, such as scaling or normalization, to prepare the data for clustering.

2. **K-means Clustering**  
   - Implements K-means clustering to segment the dataset into three clusters (`k=3`), corresponding to the three flower species.  
   - The algorithm iteratively adjusts cluster centroids and assigns data points to the nearest centroid to minimize within-cluster variance.

3. **Cluster Boundary Visualization**  
   - Visualizes the data points in two-dimensional space, using colors to indicate cluster assignments and displaying boundaries around each cluster.  
   - Provides a clear depiction of how the K-means model differentiates between species.

## Output: 

![Cluster Boundaries](https://github.com/user-attachments/assets/0a56d78a-053b-4eb5-b1b6-c26539132517)
