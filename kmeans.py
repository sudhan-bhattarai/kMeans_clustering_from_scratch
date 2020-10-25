'''
Data retrieved from:
Cooper, L. (1964). Heuristic methods for location-allocation problems. SIAM review, 6(1), 37-53.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
df = pd.read_excel(r'data.xlsx')
df = df.set_index('Customer', drop = True)
grid = df.to_numpy()

class kMeans_from_scratch:
  def __init__(self, coordinates, tolerance, iter, nof):
    self.grid = coordinates
    self.N = coordinates.shape[0]
    self.t = tolerance
    self.nof = nof
    self.f = None
    self.fac = None
    self.assigned_nodes = []

  def initial_facility(self):  ## randomly create facility locations inside the grid
    self.fac = np.empty([self.nof,2])
    for i in range(self.nof):
      self.fac[i,0] = np.random.randint(np.min(self.grid[:,0]), np.max(self.grid[:,0]))
      self.fac[i,1] = np.random.randint(np.min(self.grid[:,1]), np.max(self.grid[:,1]))
    return self.fac

  def fac_assignment(self, fac):  ## assign each facility to the node with shortest distance
    assign = [0]*self.N
    for n in range(self.N):
      distance = [0]*self.nof
      for f in range(self.nof):
        distance[f] = math.sqrt((self.grid[n,0]-fac[f,0])**2 + (self.grid[n,1]-fac[f,1])**2)
      assign[n] = np.argmin(distance)
    return assign

  def nodes_assignment(self, N, fa, assigned):  ## all assigned nodes to a facility
    n_assign = []
    for asg in range(N):
      if fa == assigned[asg]:
        n_assign.append(asg)
    return n_assign

  def kMeans(self, nof):
    self.initial_facility()
    for itr in range(iter):
      assign = self.fac_assignment(self.fac)
      for fa in range(nof):
        x, y = 0, 0
        n_assign = self.nodes_assignment(self.N, fa, assign)
        x, y = np.mean(self.grid[n_assign,0]), np.mean(self.grid[n_assign,1])
        self.assigned_nodes.append(self.grid[n_assign].tolist())
        if abs(x-self.fac[fa,0]) >= self.t:
          self.fac[fa,0] = x
        if abs(y-self.fac[fa,1]) >= self.t:
          self.fac[fa,1] = y
    self.f = self.fac
    return self.fac

  def plot(self):
    self.kMeans(self.nof)
    plt.scatter(df['X'], df['Y'], marker = 'x', color = 'b', label = 'coordinates')
    plt.scatter(self.fac[:,0], self.fac[:,1], marker = 'o', color = 'g', label = 'facility')
    plt.title('Kmeans clustering'), plt.xlabel('X'), plt.ylabel('Y')
    plt.legend(loc = 1)
    plt.show()

tol, iter, f = 0.01, 10000, 3
kM = kMeans_from_scratch(grid, tol, iter, f)
kM.kMeans(3)
kM.plot()
print('final facilities:\n', kM.fac)
for i in range(len(list(kM.assigned_nodes[-kM.nof:]))):
  print('\n Assigned nodes for facility {}:'.format(i),'\n', list(kM.assigned_nodes[-kM.nof:][i]))
