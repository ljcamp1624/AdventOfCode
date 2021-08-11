#%% Advent of code 2019
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
from matplotlib.animation import FuncAnimation

#%%
x = pd.read_csv(r"C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019_py3\q14.csv", header=None)
x = pd.DataFrame(x[0].str.split('=>', expand=True))

reac = x[0].str.split(' ', expand=True).replace(',', '', regex=True).replace(' ', '', regex=True);
prod = x[1].str.split(' ', expand=True).replace(',', '', regex=True).replace(' ', '', regex=True);

items = [];
for i in range(1, reac.shape[1], 2):
    items = np.append(items, reac[i].values);
items = np.unique(items[items!=None]);
key = pd.DataFrame({'name':items, 'index':range(len(items))});


mat = np.zeros([reac.shape[0], key.shape[0]]);
for i in range(reac.shape[0]):
    for j in range(0, reac.shape[1], 2):
        
        mat.iloc[i, j] = 
        

# reac = reac.replace('', 0)#.replace(None, 0);
# for i in range(0, reac.shape[1], 2):
#     reac[i] = reac[i].astype(int);