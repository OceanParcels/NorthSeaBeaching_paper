#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:14:23 2021

@author: kaandorp
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
# import cmocean.cm as cmo
import pickle
import math


def load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

#%%
cmap = plt.cm.tab10
colors_tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959',
                 '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

# data_fig1 = load('01_figure_data/fig1_370_202106301124.pickle')
data_fig1 = load('01_figure_data/fig1_391_202112131728.pickle')

fig,ax = plt.subplots(1,figsize=(5,7))

for i1 in range(5):
    ax.plot(data_fig1['y_test'][i1],data_fig1['y_pred'][i1],'o',color=colors_tableau[i1],label='test fold %i, R: %2.2f' %(i1+1,data_fig1['R_test'][i1]))

y_max = 2.9
ax.set_xlabel(r'Observed value [kg km$^{-1}$]',fontsize=13)
ax.set_ylabel(r'Predicted value [kg km$^{-1}$]',fontsize=13)
ax.axis('equal')
ax.plot([0,y_max],[0,y_max],'k--',label='1:1')

ax.set_xticks(np.arange(0,3,1))
ax.set_yticks(np.arange(0,4,1))

ax.set_xticklabels(10**np.arange(0,3,1))
ax.set_yticklabels(10**np.arange(0,4,1))

n_std = 2
estim_var = 0.08
log10_std_y = np.sqrt(estim_var)

dy = n_std*log10_std_y
minval = 0
maxval = y_max     
y1u = minval+dy
y2u = maxval+dy

y1l = minval-dy
y2l = maxval-dy

ax.plot([minval,maxval],[y1u,y2u],'r--',label='2x std from variogram',zorder=0)
ax.plot([minval,maxval],[y1l,y2l],'r--',zorder=0)

ax.legend()   
ax.set_title('Pearson R: %2.2f +- %2.2f' % (data_fig1['array_pearsonR'].mean(),data_fig1['array_pearsonR'].std())) 
fig.subplots_adjust(left=0.17)

#%% All features
# data_fig2 = load('01_figure_data/fig2_370_202106301209.pickle')
data_fig2 = load('01_figure_data/fig2_391_202112141146.pickle')

labels = data_fig2['labels']
labels2 = []
for i1,label_ in enumerate(labels):
    if 'in_' in label_:
        label_ = label_.replace('in_','dot_')
    if 'mdot' in label_: #stupid..
        label_ = label_.replace('mdot','min')
        
    labels2.append( '%i) %s' % (len(labels)-i1,label_) )
    
fig,ax = plt.subplots(1,figsize=(12,12))
ax.boxplot(data_fig2['feature_importance_score_mat'] , vert=False, labels=labels2)
ax.set_xlabel('Gini importance')
fig.tight_layout()

for tick in ax.yaxis.get_major_ticks():
    str_tick = str(tick.label1)
    color_ = 'darkblue'
    if 'beaching' in str_tick:
        color_ = 'darkorange'
    elif 'dot' in str_tick:
        color_ = 'firebrick'
    tick.label1.set_color(color_)

# cluster_names_top10 = np.array([r'$h_{tide}$, std. (t = 30d.)',
#                        r'$h_{tide}$, max. (t = 3d.)',
#                        r'$\mathbf{n}_{grid} \cdot \mathbf{n}$',
#                        r'$l_{coast}$  (r = 50km)',
#                        r'$h_{tide}$, max. (during tour)',
#                        r'$F_{beach.,riv.}$ (r = 50km, t = 1d., $\tau_{beach}=75d.$)',
#                        r'$\mathbf{U_{curr.} \cdot n}$, min. (r = 0km, t = 30d.)',
#                        r'$F_{beach.,fis.}$ (r = 100km, t = 3d., $\tau_{beach}=75d.$)',
#                        r'$\mathbf{U_{curr.} \cdot n}$, max. (r = 100km, t = 3d.)',
#                        r'$F_{beach.,pop.}$ (r = 50km, t = 9d., $\tau_{beach}=25d.$)'])
cluster_names_top10 = np.array([r'$h_{tide}$, std. (t = 30d.)',
                        r'$h_{tide}$, max. (t = 3d.)',
                        r'$F_{beach.,fis.}$ (r = 50km, t = 9d., $\tau_{beach}=25d.$)',
                        r'$l_{coast}$  (r = 50km)',
                        r'$\mathbf{n}_{grid} \cdot \mathbf{n}$',
                        r'$h_{tide}$, max. (during tour)',
                        r'$F_{beach.,pop.}$ (r = 50km, t = 30d., $\tau_{beach}=150d.$)',
                        r'$\mathbf{U_{tide} \cdot n}$, max. (r = 0km, t = 3d.)',
                        r'$\mathbf{U_{curr.} \cdot n}$, min. (r = 0km, t = 30d.)',
                        r'$n_{fis.}$ (r = 0km)'])

fig,ax = plt.subplots(1,figsize=(10,5))
ax.boxplot(data_fig2['feature_importance_score_mat'][:,-10:], vert=False)
ax.set_yticklabels(cluster_names_top10[::-1],fontsize=13)
ax.set_xlabel('Gini importance',fontsize=13)
fig.tight_layout()

colors = ['darkblue','darkblue','darkorange','darkblue','firebrick',
          'darkblue','darkorange','firebrick','firebrick','darkblue'][::-1]
for color,tick in zip(colors,ax.yaxis.get_major_ticks()):
    tick.label1.set_color(color) #set the color property

#%%  Poster plot
data_fig2 = load('01_figure_data/fig2_370_202106301209.pickle')

labels = data_fig2['labels']
labels2 = []
for i1,label_ in enumerate(labels):
   labels2.append( '%i) %s' % (len(labels)-i1,label_) )
    
fig,ax = plt.subplots(1,figsize=(12,12))
ax.boxplot(data_fig2['feature_importance_score_mat'] , vert=False, labels=labels2)
ax.set_xlabel('Gini importance')
fig.tight_layout()

for tick in ax.yaxis.get_major_ticks():
    str_tick = str(tick.label1)
    color_ = 'darkblue'
    if 'beaching' in str_tick:
        color_ = 'darkorange'
    elif 'dot' in str_tick:
        color_ = 'firebrick'
    tick.label1.set_color(color_)

cluster_names_top10 = np.array([r'$h_{tide}$, std. (t = 30d.)',
                       r'$h_{tide}$, max. (t = 3d.)',
                       r'$\mathbf{n}_{grid} \cdot \mathbf{n}$',
                       r'$l_{coast}$  (r = 50km)',
                       r'$h_{tide}$, max. (during tour)',
                       r'$F_{beach.,riv.}$ (r = 50km, t = 1d., $\tau_{beach}=75d.$)',
                       r'$\mathbf{U_{curr.} \cdot n}$, min. (r = 0km, t = 30d.)',
                       r'$F_{beach.,fis.}$ (r = 100km, t = 3d., $\tau_{beach}=75d.$)',
                       r'$\mathbf{U_{curr.} \cdot n}$, max. (r = 100km, t = 3d.)',
                       r'$F_{beach.,pop.}$ (r = 50km, t = 9d., $\tau_{beach}=25d.$)'])

fig,ax = plt.subplots(1,figsize=(10,5))
fig.patch.set_alpha(0.)

ax.boxplot(data_fig2['feature_importance_score_mat'][:,-10:], vert=False)
ax.set_yticklabels(cluster_names_top10[::-1],fontsize=13)
ax.set_xlabel('Gini importance',fontsize=13)
fig.tight_layout()

colors = ['darkblue','darkblue','firebrick','firebrick','darkblue',
          'darkorange','darkorange','darkorange','darkorange','darkorange'][::-1]
for color,tick in zip(colors,ax.yaxis.get_major_ticks()):
    tick.label1.set_color(color) #set the color property

#%% no model features
data_fig2 = load('01_figure_data/fig2_247_202112131816.pickle')

labels = data_fig2['labels']
labels2 = []
for i1,label_ in enumerate(labels):
    if 'in_' in label_:
        label_ = label_.replace('in_','dot_')
    if 'mdot' in label_: #stupid..
        label_ = label_.replace('mdot','min')
        
    labels2.append( '%i) %s' % (len(labels)-i1,label_) )
    
fig,ax = plt.subplots(1,figsize=(12,12))
ax.boxplot(data_fig2['feature_importance_score_mat'] , vert=False, labels=labels2)
ax.set_xlabel('Gini importance')
fig.tight_layout()

for tick in ax.yaxis.get_major_ticks():
    str_tick = str(tick.label1)
    color_ = 'darkblue'
    if 'beaching' in str_tick:
        color_ = 'darkorange'
    elif 'dot' in str_tick:
        color_ = 'firebrick'
    tick.label1.set_color(color_)
#%% no model features, correlation


cmap = plt.cm.tab10

data_fig1 = load('01_figure_data/fig1_226_202106301238.pickle')

fig,ax = plt.subplots(1,figsize=(5,7))

for i1 in range(5):
    ax.plot(data_fig1['y_test'][i1],data_fig1['y_pred'][i1],'o',label='test fold %i, R: %2.2f' %(i1+1,data_fig1['R_test'][i1]))

y_max = 2.9
ax.set_xlabel(r'True value [log$_{10}$(kg km$^{-1}$)]',fontsize=13)
ax.set_ylabel(r'Predicted value [log$_{10}$(kg km$^{-1}$)]',fontsize=13)
ax.axis('equal')
ax.plot([0,y_max],[0,y_max],'k--',label='1:1')

n_std = 2
estim_var = 0.08
log10_std_y = np.sqrt(estim_var)

dy = n_std*log10_std_y
minval = 0
maxval = y_max     
y1u = minval+dy
y2u = maxval+dy

y1l = minval-dy
y2l = maxval-dy

ax.plot([minval,maxval],[y1u,y2u],'r--',label='2x std from variogram',zorder=0)
ax.plot([minval,maxval],[y1l,y2l],'r--',zorder=0)

ax.legend()   
ax.set_title('Pearson R: %2.2f +- %2.2f' % (data_fig1['array_pearsonR'].mean(),data_fig1['array_pearsonR'].std())) 
fig.subplots_adjust(left=0.17)

#%% no population density
data_fig2 = load('01_figure_data/fig2_388_202112141432_noPopDen.pickle')

labels = data_fig2['labels']
labels2 = []
for i1,label_ in enumerate(labels):
    if 'in_' in label_:
        label_ = label_.replace('in_','dot_')
    if 'mdot' in label_: #stupid..
        label_ = label_.replace('mdot','min')
        
    labels2.append( '%i) %s' % (len(labels)-i1,label_) )
    
i_use = np.arange(10)
i_use = np.append(i_use,42)
i_use = len(labels) - i_use - 1
i_use = i_use[::-1]

fig,ax = plt.subplots(1,figsize=(12,12))
ax.boxplot(data_fig2['feature_importance_score_mat'][:,i_use] , vert=False, labels=np.array(labels2)[i_use])
ax.set_xlabel('Gini importance')
fig.tight_layout()

for tick in ax.yaxis.get_major_ticks():
    str_tick = str(tick.label1)
    color_ = 'darkblue'
    if 'beaching' in str_tick:
        color_ = 'darkorange'
    elif 'dot' in str_tick:
        color_ = 'firebrick'
    tick.label1.set_color(color_)