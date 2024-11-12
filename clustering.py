import pandas as pd
import numpy as np
import random
import sys

from data_generator import DataGenerator as data

class Clustering:

  def __init__(self):
    self.distortion_over_time = []
    self.means_over_time = []

    # Initialize cluster means to random points:
    for i in range(data.k):
      random_point = []
      for dim in range(len(data.data_vectors[0])):
        random_point.append(random.randrange(int(data.minmax[dim][0]*100), int(data.minmax[dim][1])*100)/100)
      data.means[i] = np.r_['c', random_point]

    self.k_means()

  def get_distortion_over_time(self) -> list:
    return self.distortion_over_time
  
  def get_means_over_time(self) -> list:
    return self.means_over_time

  def k_means(self) -> None:

    last = sys.maxsize
    iteration = 1

    while True:
      # Resets cluster points for each iteration
      data.clusters = [[] for i in range(data.k)]

      # Assign clusters based on closest mean:
      for index, data_point in enumerate(data.data_vectors):
        data.clusters[self.closest_mean(data_point)].append(index)

      # Update means
      self.means_over_time.append(data.means.copy())
      self.update_means()

      distortion = self.objective_function()

      self.distortion_over_time.append((iteration, distortion))

      # Break if objective function converges
      if last - distortion < 0.001:
        break

      last = distortion
      iteration += 1

  def closest_mean(self, data_vector) -> int:
    """
    Finds the closest cluster mean of data vector by square distance
    """
  
    square_distances = {}

    for k in range(data.k):
      square_distances[k] = np.linalg.norm(data_vector - data.means[k]) ** 2

    return min(square_distances, key=square_distances.get)


  def update_means(self) -> None:
    """
    Updates the means of each cluster using the learning function
    """
    for k in range(data.k):
      sum = np.zeros(data.data_vectors[0].shape) # Initialize zero vector for mean
      for n in range(len(data.data_vectors)):
        if n in data.clusters[k]:
          sum = np.add(sum, data.data_vectors[n]) # Sum of all data vectors in cluster
      
      if len(data.clusters[k]) != 0:
        data.means[k] = np.divide(sum, len(data.clusters[k])) # Update mean
      else:
        data.means[k] = np.zeros(data.data_vectors[0].shape)

  def objective_function(self) -> float:
    """
    Calcualtes distortion based on objective function
    """

    distortion = 0
    
    for n in range(len(data.data_vectors)):
      for k in range(data.k):
        if n in data.clusters[k]:
          distortion += np.linalg.norm(data.data_vectors[n] - data.means[k]) ** 2
    
    return distortion
