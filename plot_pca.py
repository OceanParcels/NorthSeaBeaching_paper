#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:36:42 2021

@author: kaandorp
"""
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import pickle
import math
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
# filename = 'pickle_files/regression_table_180_321_20210517.pickle'
filename = 'pickle_files/regression_table_180_468_20211213.pickle'
# filename = '../NorthSeaBeaching/pickle_files/regression_table_180_452_20210629.pickle'
with open(filename, 'rb') as f:
    regression_table = pickle.load(f)
df = regression_table.dropna(axis=0,how='any').copy()

def load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# vars_ = ['tide_std_030','tide_max_003','dot_mesh_coast','coastal_length_050','tide_tour_max']
# # vars_ = ['tide_std_030','tide_max_003','beaching_f_tau25_050_009','dot_mesh_coast','coastal_length_050']

# vars_ = ['tide_std_030','tide_max_003','beaching_f_tau25_050_009','dot_mesh_coast','coastal_length_050',
#          'beaching_p_tau150_050_030','tide_tour_max','in_tide_max_003','in_currents_min_000_030','in_Stokes_min_000_009']


# data_noLag = load('01_figure_data/fig2_391_202112141146.pickle')

# labels_noLag = data_noLag['labels']

# features_use_noLag = labels_noLag[::-1][:20]
data_Lag = load('01_figure_data/fig2_391_202112141146.pickle')

labels_Lag = data_Lag['labels']
vars_ = labels_Lag[::-1][:8]


# X = df.iloc[:,4:]#.loc[:,vars_]
X = df.loc[:,vars_]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

pca = PCA(n_components=2)
pca.fit(X)
Y = pca.transform(X)

cmap = plt.cm.viridis
bounds = np.linspace(51,53.5,6)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig,ax = plt.subplots(1,figsize=(7,5))

cplt = ax.scatter(Y[:,0],Y[:,1],c=df['lat'],cmap=cmap,norm=norm)
cbar = plt.colorbar(cplt)
cbar.set_label('Latitude')

ax.set_xlabel('Principal component 1',fontsize=13)
ax.set_ylabel('Principal component 2',fontsize=13)

print('loadings')
print(pca.components_.T)

cmap = plt.cm.viridis
bounds = np.logspace(np.log10(2),np.log10(500),5)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig,ax = plt.subplots(1,figsize=(7,5))

cplt = ax.scatter(Y[:,0],Y[:,1],c=df['kg/m'],cmap=cmap,norm=norm)
cbar = plt.colorbar(cplt)
cbar.set_label('kg/km')

ax.set_xlabel('Principal component 1',fontsize=13)
ax.set_ylabel('Principal component 2',fontsize=13)