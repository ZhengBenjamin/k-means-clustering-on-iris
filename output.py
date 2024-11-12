import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import math

from data_generator import DataGenerator as data
from clustering import Clustering as cluster

class Output:
  
  def __init__(self, k: int) -> None:
    data.generate(k, 'irisdata.csv')
    self.cluster = cluster()

    # Change the number of clusters at the bottom of the file (k)

    # Run one at at time. Comment out the rest

    # Plots the total distortion value as a function of iterations
    # self.plot_objective()

    # Plots final cluster means and data points
    # self.plot_clusters(data.means)

    # Plots the progress of the clusters over time 
    # self.progress_over_time()

    # Plots the boundaries of the clusters using a contour plot
    self.plot_boundaries()

  def plot_objective(self) -> None:
    """
    Plots the total distortion as a function of iterations
    """
    distortion_over_time = self.cluster.get_distortion_over_time()

    x = []
    y = []

    for iteration in distortion_over_time:
      x.append(iteration[0])
      y.append(iteration[1])

    plt.plot(x, y)
    plt.title("Total Distortion as a Function of Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Total Distortion")
    plt.xticks(x)
    plt.savefig(f"Distortion over time {data.k} clusters.png")
    plt.close()

  def progress_over_time(self) -> None:
    """
    Plots the progress of the clusters over time
    """
    means_over_time = self.cluster.get_means_over_time()
    means = []
    means.append((means_over_time[0], "Initial Means Cluster"))
    means.append((means_over_time[math.floor(len(means_over_time) / 3)], "Intermediate Means Cluster"))
    means.append((means_over_time[-1], "Final Means Cluster"))

    print(means)

    for mean in means:
      self.plot_clusters(mean[0], mean[1])

  def plot_clusters(self, means, title="Iris Data Cluster Means") -> None:

    values = data.data_vectors_with_values
    x = []
    y = []
    flower = []

    figure, axes = plt.subplots()

    for value in values:
      x.append(value[2])
      y.append(value[3])
      flower.append("green" if value[4] == "versicolor" else "blue" if value[4] == "virginica" else "red")

    for mean in means:
      axes.add_artist(plt.Circle((mean[2], mean[3]), 0.5, color='black', fill=False))
    
    axes.scatter(x, y, c=flower)
    
    plt.title(title)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')

    plt.savefig(f"{title}.png")

  def plot_boundaries(self): 
    """
    Plots the boundaries of the clusters using a contour plot
    """

    # Values for scatterplot 
    x = [data.data_vectors[i][2] for i in range(len(data.data_vectors))]
    y = [data.data_vectors[i][3] for i in range(len(data.data_vectors))]

    # Boundaries for boundaries plot
    x_min, x_max = min(x) - 0.5, max(x) + 0.5
    y_min, y_max = min(y) - 0.5, max(y) + 0.5
    color = []

    # Color the points based on the flower type
    for value in data.data_vectors_with_values:
      if value[4] == 'setosa':
        color.append('red')
      elif value[4] == 'versicolor':
        color.append('blue')
      else:
        color.append('green')

    plt.scatter(x, y, c=color)

    # Create meshgrid for contour plot
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    grid = np.zeros(xx.shape)

    # Assign each point to a cluster
    for x in range(grid.shape[0]):
      for y in range(grid.shape[1]):
        grid[x][y] = -1
        min_dist = float('inf')
        for k in range(data.k): # For each cluster
          dist = np.linalg.norm([xx[x][y], yy[x][y]] - data.means[k][2:4] + 0.75) # Calculate distance
          if dist < min_dist: # If distance is less than the minimum distance
            min_dist = dist 
            grid[x][y] = k # Assign the point to the cluster
    
    plt.contourf(xx, yy, grid, alpha=0.1)

    plt.title('Cluster Boundaries')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.savefig('Cluster Boundaries.png')

  def plot_boundaries_old(self):
    """
    Old implementation of plotting boundaries using orthogonal lines
    """
    
    # Get the 2D means of the clusters
    means = [mean[2:4] for mean in data.means]
    
    # Vector pairs of closest means 
    mean_pairs = self.closest_mean_pairs(means, 2)

    # Slope of the perpendicular line between the two closest means
    slopes = []
    for mean1, mean2 in mean_pairs:
      slope = (- (mean2[0] - mean1[0]) / (mean2[1] - mean1[1]))
      slopes.append(slope.item())

    # Get midpoint between the two closest means
    midpoints = []
    
    for mean1, mean2 in mean_pairs:
      midpoint = ((mean1[0] + mean2[0]) / 2, (mean1[1] + mean2[1]) / 2)
      midpoints.append(midpoint)

    # Get the y-intercept of the line between the two closest means
    y_intercepts = []
    for slope, midpoint in zip(slopes, midpoints):
      y_intercept = midpoint[1] - slope * midpoint[0]
      y_intercepts.append(np.squeeze(np.asarray(y_intercept)))
    
    # Plot the clusters

    values = data.data_vectors_with_values
    x = []
    y = []
    flower = []

    figure, axes = plt.subplots()

    for value in values:
      x.append(value[2])
      y.append(value[3])
      flower.append("green" if value[4] == "versicolor" else "blue" if value[4] == "virginica" else "red")
    
    axes.scatter(x, y, c=flower)
    
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')

    # Plot the boundaries

    x = np.array(x)

    for slope, y_intercept in zip(slopes, y_intercepts):
      y = slope * x + y_intercept
      axes.plot(x, y)

    plt.xlim(data.minmax[2][0], data.minmax[2][1])
    plt.ylim(data.minmax[3][0], data.minmax[3][1])
    plt.show()

  def closest_mean_pairs(self, means, num_pairs):

    pairs = [(mean1, mean2) for mean1, mean2 in combinations(means, 2)]
    pairs.sort(key=lambda pair: np.linalg.norm(np.array(pair[0]) - np.array(pair[1])))
    
    return pairs[:num_pairs]


if __name__ == '__main__':
  output = Output(3) # Change the number of clusters here