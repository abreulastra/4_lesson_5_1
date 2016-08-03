# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:31:49 2016

@author: AbreuLastra_Work
"""

import pandas as pd
import matplotlib.pyplot as plt
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

df= pd.read_csv('https://raw.githubusercontent.com/Thinkful-Ed/curric-data-001-data-sets/master/un/un.csv', header=0)
print(df.shape)
print(df.head())
print(df.count())

#defining the number of clusters we are going to try

k_neighbors = range(1,11)
df = df[df['GDPperCapita']>0]
data = df['GDPperCapita']

analysis = df['infantMortality']

distances =[]

# calculating the average distance to centroid for different numbers of clustes
for k in k_neighbors:
    cluster_dist = kmeans(data,k)[-1]
    distances.append(cluster_dist)

# plotting the distance vs the mumber of clustesr
plt.figure(1)
plt.plot(k_neighbors, distances)
plt.show()

# by inspection we determine that having three clusters reduces the distance enough. More clusters do not reduce the distance marginally

df = df[df['GDPperCapita']>0]
centroids,_ = kmeans(data,3)
idx,_= vq(data,centroids)

plt.figure()
plt.plot(data[idx==0], df['infantMortality'][idx==0], 'ob')
plt.plot(data[idx==1], df['infantMortality'][idx==1], 'or')
plt.plot(data[idx==2], df['infantMortality'][idx==2], 'og')
plt.legend()
plt.show()

plt.figure()
plt.plot(data[idx==0], df['lifeMale'][idx==0], 'ob')
plt.plot(data[idx==1], df['lifeMale'][idx==1], 'or')
plt.plot(data[idx==2], df['lifeMale'][idx==2], 'og')
plt.legend()
plt.show()

plt.figure()
plt.plot(data[idx==0], df['lifeFemale'][idx==0], 'ob')
plt.plot(data[idx==1], df['lifeFemale'][idx==1], 'or')
plt.plot(data[idx==2], df['lifeFemale'][idx==2], 'og')
plt.legend()
plt.show()

"""Need to ask how to pass series to a loop"""