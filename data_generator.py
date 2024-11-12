import pandas as pd
import numpy as np

class DataGenerator:
  
  k = 0
  data_vectors = []
  data_vectors_with_values = []
  means = []
  clusters = []
  minmax = []

  def generate(k: int, data) -> None:
    DataGenerator.k = k
    DataGenerator.gen_data_vectors(data)
    DataGenerator.set_init_clusters()
    
  def gen_data_vectors(data) -> None:
    """
    Reads data from CSV and generates list of data vectors
    """
    df = pd.read_csv(data)
    DataGenerator.data_vectors_with_values = df.values.tolist()
    df_elements = df.iloc[:, :-1]

    DataGenerator.minmax = [[np.inf, -np.inf] for i in range(len(df_elements.values[0]))]
    
    for value in df_elements.values:
      for i in range(len(value)):
        if value[i] < DataGenerator.minmax[i][0]:
          DataGenerator.minmax[i][0] = value[i]
        if value[i] > DataGenerator.minmax[i][1]:
          DataGenerator.minmax[i][1] = value[i]
      DataGenerator.data_vectors.append(np.r_['c', value])

  def set_init_clusters():
    """
    Initializes the means and clusters
    Means are set to the first k data vectors
    Clusters are set to empty lists
    """

    DataGenerator.means = [] # Reset means / clusters
    DataGenerator.clusters = []

    for i in range(DataGenerator.k):
      DataGenerator.means.append([])
    
    for i in range(DataGenerator.k):
      DataGenerator.clusters.append([])

