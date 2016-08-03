# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:13:00 2016

@author: AbreuLastra_Work
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix



column_name = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'class']

df = pd.read_csv('https://raw.githubusercontent.com/Thinkful-Ed/curric-data-001-data-sets/master/iris/iris.data.csv', names=column_name)
class_names = set(df['class'])

#Scatterplott, sepal width by length

x,y = df['sepal_l'],  df['sepal_w']

for name in class_names:
    cond = df['class'] == name
    plt.plot(x[cond], y[cond], linestyle='none', marker='o', label=name)

plt.legend(numpoints=1)
plt.show()

