#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:24:20 2021
Assess how many features are really necessary
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

import shapely.geometry
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from shapely.ops import split

from shapely.geometry import (box, LineString, MultiLineString, MultiPoint,
    Point, Polygon, MultiPolygon, shape)

from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance
from collections import defaultdict

#%% Part 1: verify that only using the top 10 features works
def load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def find_features_containing(list_features,string_):
    return np.array([True if string_ in feature_ else False for feature_ in list_features])


def normalize_beaching_variables(regression_table_):
    i_100 = np.where(find_features_containing(regression_table_.keys(),'beaching') & find_features_containing(regression_table_.keys(),'_100_'))[0]
    i_50 = np.where(find_features_containing(regression_table_.keys(),'beaching') & find_features_containing(regression_table_.keys(),'_050_'))[0]
    i_20 = np.where(find_features_containing(regression_table_.keys(),'beaching') & find_features_containing(regression_table_.keys(),'_020_'))[0]

    for i_100_ in i_100:
        regression_table_.iloc[:,i_100_] /= (regression_table_.loc[:,'coastal_length_100'].values/1000)
    for i_50_ in i_50:
        regression_table_.iloc[:,i_50_] /= (regression_table_.loc[:,'coastal_length_050'].values/1000)
    for i_20_ in i_20:
        regression_table_.iloc[:,i_20_] /= (regression_table_.loc[:,'coastal_length_020'].values/1000)

    return regression_table_


def datetime64_to_datetime(datetime64):
    if type(datetime64) == np.ndarray:
        array_datetime = np.array([])
        for dt64 in datetime64:
            
            ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            array_datetime = np.append(array_datetime, datetime.utcfromtimestamp(ts))
        return array_datetime
    else:
        ts = (datetime64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        return datetime.utcfromtimestamp(ts)


def impute_participant_numbers(regression_table_):
    array_years = np.array([2014,2015,2016,2017,2018,2019])
    array_participants = np.array([1479,2015,2320,2748,2764,2568])

    data_available = regression_table_.dropna(axis=0,how='any',subset=['participants']).copy()
    data_time = datetime64_to_datetime(data_available['time'].values)

    for index, time_, lat_ in zip(regression_table_.index,regression_table_['time'],regression_table_['lat']):
        
        if np.isnan(regression_table_.loc[index,'participants']):
            
            data_at_location = data_available[data_available['lat'] == lat_]
            data_at_location_year = np.array([d_.year for d_ in datetime64_to_datetime(data_at_location['time'].values) ])

            data_at_location_participants = np.array([])

            for i1,year_ in enumerate(data_at_location_year):
                i_year = np.where(array_years == year_)[0]
                data_at_location_participants = np.append(data_at_location_participants,array_participants[i_year])

            current_year = datetime64_to_datetime(time_).year
            participant_fractions = array_participants[np.where(array_years == current_year)[0]] / data_at_location_participants
            

            if len(participant_fractions) == 1:
                res_ = data_at_location['participants'].values * participant_fractions
            elif len(participant_fractions) > 1:
                res_ = (data_at_location['participants'].values * participant_fractions).mean()
            else:
                i_closest_loc = np.argmin(data_available['lat'] - lat_)
                closest_year = data_time[i_closest_loc].year
                
                res_ = data_available.loc[:,'participants'].values[i_closest_loc] * (array_participants[np.where(array_years == current_year)[0]] / 
                                                                         array_participants[np.where(array_years == closest_year)[0]])
            
            regression_table_.loc[index,'participants'] = res_
            

filename = 'pickle_files/regression_table_180_468_20211213.pickle'

with open(filename, 'rb') as f:
    regression_table = pickle.load(f)

regression_table_ = regression_table.dropna(axis=0,how='any',subset=['kg/m']).copy()
regression_table_ = normalize_beaching_variables(regression_table_)

output_table = pd.DataFrame(regression_table_.iloc[:,0:4])

impute_participant_numbers(regression_table_)


regressor = 'RFR' #RFR, LR, GLM
use_scaling = True
use_PCA = False

if regressor == 'RFR':
    reg1 = RandomForestRegressor(oob_score=True,max_features=.33)
elif regressor == 'LR':
    reg1 = linear_model.LinearRegression()
elif regressor == 'GLM':
    reg1 = linear_model.TweedieRegressor(power=2,link='log',alpha=0.,fit_intercept=True)   
elif regressor == 'GPR':
    reg1 = GaussianProcessRegressor()    

# features_use = ['tide_std_030','tide_max_003','dot_mesh_coast','coastal_length_050','tide_tour_max','beaching_r_tau75_050_001',
#                 'dot_currents_min_000_030','beaching_f_tau75_100_003','dot_currents_max_100_003','beaching_p_tau25_050_009']

data_Lag = load('01_figure_data/fig2_391_202112141146.pickle')

labels_Lag = data_Lag['labels']

data_noLag = load('01_figure_data/fig2_247_202112131816.pickle')

labels_noLag = data_noLag['labels']

features_use_noLag = labels_noLag[::-1][:20]
features_use = labels_Lag[::-1][:20]
# features_use_noLag = ['tide_std_030','tide_max_003','dot_mesh_coast','coastal_length_050','tide_tour_max',
#                 'dot_currents_min_000_030','mag_currents_mean_000_009','dot_Stokes_min_000_009','dot_currents_max_100_003','dot_Stokes_mean_100_030',
#                 'pop_density_050','dot_currents_max_050_009','dot_currents_mean_100_030','coastal_length_000','tide_tour_min',
#                 'mag_wind_mean_000_003','VHM0_mean_050_030','dot_wind_min_100_003','mag_currents_max_000_003','dot_currents_mean_050_003']

# features_use = ['tide_std_030','tide_max_003','dot_mesh_coast','coastal_length_050','tide_tour_max','beaching_r_tau75_050_001',
#                 'dot_currents_min_000_030','beaching_f_tau75_100_003','dot_currents_max_100_003','beaching_p_tau25_050_009',
#                 'mag_currents_mean_000_009','dot_Stokes_min_000_009','pop_density_050','dot_currents_max_050_009','tide_tour_min',
#                 'dot_currents_mean_100_030','dot_Stokes_mean_100_030','coastal_length_000','mag_wind_mean_000_003','VHM0_mean_050_030']

#%% 
n_repeat = 10
n_splits = 5
pearson_R_scores = np.zeros([len(features_use),n_repeat*n_splits])

for i_outer in range(len(features_use)):
    
    features_use_ = features_use[0:i_outer+1]
    
    x = regression_table_.loc[:,features_use_]
    y = np.log10(regression_table_.loc[:,'kg/m'])
    print(x.shape)
    
    for i_inner in range(n_repeat):
        
        print('inner loop %i/%i' % (i_inner+1,n_repeat))
        
        kf = KFold(n_splits=n_splits,shuffle=True)
        
        
        for i1, (i_train, i_test) in enumerate(kf.split(x)):
        
            x_train = x.iloc[i_train,:]
            y_train = y.iloc[i_train]
            x_test = x.iloc[i_test,:]
            y_test = y.iloc[i_test]
                
            
            #----------------apply scaling: all variables to mean=0, std=1--------------
            if use_scaling:
                scaler = StandardScaler()
                scaler.fit(x_train)
                
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            
                x_all = scaler.transform(x)
            #----------------apply PCA: retain 95% of the variance--------------------
            if use_PCA:
                pca = PCA(.95)
                pca.fit(x_train)
                
                x_train = pca.transform(x_train)
                x_test = pca.transform(x_test)
            
                x_all = pca.transform(x_all)
     
            
            reg1.fit(x_train,y_train)
            
            y_pred_train = reg1.predict(x_train)
            y_pred_test = reg1.predict(x_test)
            
            pearsonR = pearsonr(y_pred_test,y_test)[0]
            pearson_R_scores[i_outer,i_inner*n_splits:(i_inner+1)*n_splits] = pearsonR

#%%
pearson_R_scores_noLag = np.zeros([len(features_use_noLag),n_repeat*n_splits])

for i_outer in range(len(features_use_noLag)):
    
    features_use_ = features_use_noLag[0:i_outer+1]
    
    x = regression_table_.loc[:,features_use_]
    y = np.log10(regression_table_.loc[:,'kg/m'])
    print(x.shape)
    
    for i_inner in range(n_repeat):
        
        print('inner loop %i/%i' % (i_inner+1,n_repeat))
        
        kf = KFold(n_splits=n_splits,shuffle=True)
        
        
        for i1, (i_train, i_test) in enumerate(kf.split(x)):
        
            x_train = x.iloc[i_train,:]
            y_train = y.iloc[i_train]
            x_test = x.iloc[i_test,:]
            y_test = y.iloc[i_test]
                
            
            #----------------apply scaling: all variables to mean=0, std=1--------------
            if use_scaling:
                scaler = StandardScaler()
                scaler.fit(x_train)
                
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            
                x_all = scaler.transform(x)
            #----------------apply PCA: retain 95% of the variance--------------------
            if use_PCA:
                pca = PCA(.95)
                pca.fit(x_train)
                
                x_train = pca.transform(x_train)
                x_test = pca.transform(x_test)
            
                x_all = pca.transform(x_all)
     
            
            reg1.fit(x_train,y_train)
            
            y_pred_train = reg1.predict(x_train)
            y_pred_test = reg1.predict(x_test)
            
            pearsonR = pearsonr(y_pred_test,y_test)[0]
            pearson_R_scores_noLag[i_outer,i_inner*n_splits:(i_inner+1)*n_splits] = pearsonR

#%%    
dict_results = {}
dict_results['pearson_R_scores'] = pearson_R_scores
dict_results['pearson_R_scores_noLag'] = pearson_R_scores_noLag

SAVE = True
if SAVE:
    filename_pickle = 'minimum_feature_analysis.pickle'
    outfile = open('datafiles/' + filename_pickle,'wb')
    pickle.dump(dict_results,outfile)
    outfile.close()  
#%% 
LOAD = False
if LOAD:
    filename = 'datafiles/minimum_feature_analysis.pickle'
    with open(filename, 'rb') as f:
        dict_results = pickle.load(f)
        
pearson_R_scores = dict_results['pearson_R_scores'] 
pearson_R_scores_noLag = dict_results['pearson_R_scores_noLag'] 

plt.figure()
for i_outer in range(10):
    plt.plot(i_outer*np.ones(n_repeat*n_splits),pearson_R_scores[i_outer,:],'o')
plt.plot(np.arange(len(features_use)),pearson_R_scores.mean(axis=1),'rx-')

plt.plot(np.arange(len(features_use)),pearson_R_scores_noLag.mean(axis=1),'bx-')

cmap = plt.cm.tab10
fig,ax = plt.subplots(1,figsize=(6,4))

mean = pearson_R_scores.mean(axis=1)
q_1 = np.quantile(pearson_R_scores,.1,axis=1)
q_9 = np.quantile(pearson_R_scores,.9,axis=1)

ax.fill_between(np.arange(1,len(features_use)+1),q_1,q_9,color=cmap(0),alpha=.2)
ax.plot(np.arange(1,len(features_use)+1),mean,'-',color=cmap(0),label='Using Lagrangian model features')


mean = pearson_R_scores_noLag.mean(axis=1)
q_1 = np.quantile(pearson_R_scores_noLag,.1,axis=1)
q_9 = np.quantile(pearson_R_scores_noLag,.9,axis=1)

ax.fill_between(np.arange(1,len(features_use)+1),q_1,q_9,color=cmap(1),alpha=.2)
ax.plot(np.arange(1,len(features_use)+1),mean,'-',color=cmap(1),label='Not using Lagrangian model features')

ax.errorbar(21,0.72,yerr=0.08,fmt='o',color='red',ecolor='red',capsize=4,barsabove=True,zorder=10,label='Using all %i features (Fig.4)' % len(labels_Lag))
# ax.errorbar(20,0.71,yerr=0.11,ecolor='red',capsize=4,barsabove=True)

xlabels = ['%i' % i_ for i_ in np.arange(1,len(features_use)+2)]
xlabels[-1] = len(labels_Lag)
ax.set_xticks(np.arange(1,len(features_use)+2))
ax.set_xticklabels(xlabels)
ax.set_xlabel('Amount of features used')
ax.set_ylabel('Pearson correlation coefficient R')

ax.legend(loc='lower right')

#%%

features_use_ = features_use[0:9]

x = regression_table_.loc[:,features_use_]
y = np.log10(regression_table_.loc[:,'kg/m'])
print(x.shape)

# for i_inner in range(n_repeat):
    
    # print('inner loop %i/%i' % (i_inner+1,n_repeat))
    
kf = KFold(n_splits=n_splits,shuffle=True)


for i1, (i_train, i_test) in enumerate(kf.split(x)):

    x_train = x.iloc[i_train,:]
    y_train = y.iloc[i_train]
    x_test = x.iloc[i_test,:]
    y_test = y.iloc[i_test]
        
    
    #----------------apply scaling: all variables to mean=0, std=1--------------
    if use_scaling:
        scaler = StandardScaler()
        scaler.fit(x_train)
        
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    
        x_all = scaler.transform(x)
    #----------------apply PCA: retain 95% of the variance--------------------
    if use_PCA:
        pca = PCA(.95)
        pca.fit(x_train)
        
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
    
        x_all = pca.transform(x_all)
 
    
    reg1.fit(x_train,y_train)
    
    y_pred_train = reg1.predict(x_train)
    y_pred_test = reg1.predict(x_test)
    
    output_table.loc[y_test.index,'y_test'] = y_test
    output_table.loc[y_test.index,'y_pred'] = y_pred_test
    
output_table.to_csv('datafiles/output_predictions_test_8feat.csv')

#%%
import matplotlib as mpl
from cartopy.feature import ShapelyFeature, LAND

cmap2 = plt.cm.viridis
bounds = np.logspace(np.log10(4),np.log10(256),13)
norm = mpl.colors.BoundaryNorm(bounds, cmap2.N)

fig = plt.figure(figsize=(11,5),dpi=120)  

c = 1
for i1,year in enumerate(np.arange(2014,2020)):
    print(year)

    mask_year = (output_table['time'] > datetime(year,1,1)) & (output_table['time'] < datetime(year+1,1,1))

    int_ = 230+c
    ax = fig.add_subplot(int_, projection=ccrs.PlateCarree())

    kgkm = 10**(output_table[mask_year]['y_pred'].values)
    lons = output_table[mask_year]['lon'].values
    lats = output_table[mask_year]['lat'].values

    # kgkm = data_BBCT[year]['Gewicht']/data_BBCT[year]['kms']

    i_sort = np.argsort(-kgkm)
    # fig = plt.figure(figsize=(10,5),dpi=120)     
    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent((3.2,6.8,51,54))
    ax.add_feature(LAND, zorder=0,edgecolor='black')

    ax.scatter(lons[i_sort],lats[i_sort],s=1*kgkm[i_sort]+30,c='k',
               vmin=2,vmax=500,transform=ccrs.PlateCarree())
    if year == 2016:
        cplot=ax.scatter(lons[i_sort],lats[i_sort],s=1*kgkm[i_sort],c=kgkm[i_sort],
               vmin=2,vmax=300,transform=ccrs.PlateCarree(),cmap=cmap2,norm=norm)
    else:
        ax.scatter(lons[i_sort],lats[i_sort],s=1*kgkm[i_sort],c=kgkm[i_sort],
               vmin=2,vmax=300,transform=ccrs.PlateCarree(),cmap=cmap2,norm=norm)
    ax.set_title('%i' % year)
    
    c+=1
    
cax = fig.add_axes([0.88, 0.1, 0.02, 0.7])
fig.colorbar(cplot,cax=cax,orientation='vertical',label=r'kg km$^{-1}$',extend='both')
fig.tight_layout()
fig.subplots_adjust(wspace=-.5,hspace=.12)



#%% cripple the model
reg1 = RandomForestRegressor(oob_score=True,max_features=.33,min_samples_leaf=5)


n_repeat = 20
n_splits = 5
# features_use_top = features_use[:8]
def string_not_containing(features,list_string_):
    features_use = []
    for feat_ in features:
        add_ = True
        for string_ in list_string_:
            if string_ in feat_:
                add_ = False
        
        if add_:
            features_use.append(feat_)
    return features_use

all_features = labels_Lag[::-1][:10]
# remove_str = [['yomama'],['tide'],['beaching'],['coastal','mesh'],['currents'],['density']]
remove_str = [['yomama'],['tide','beaching'],['beaching','coastal','mesh'],['currents','beaching'],['density','beaching']]

# all_features = labels_Lag[::-1][:10]
# remove_str = [[feat_] for feat_ in all_features]
# remove_str.insert(0,['yomama'])

# n_iter = len(features_use_top)
n_iter = len(remove_str)
pearson_R_scores_cripple = np.zeros([n_iter,n_repeat*n_splits])

for i_outer in range(n_iter):
    
    # features_use_ = features_use[0:i_outer+1]
    # features_use_ = np.delete(features_use_top,i_outer)
    features_use_ = string_not_containing(all_features,remove_str[i_outer])#[:20]
    print(len(features_use_))
    
    x = regression_table_.loc[:,features_use_]
    y = np.log10(regression_table_.loc[:,'kg/m'])
    print(x.shape)
    
    importance_sum = np.zeros(len(features_use_))
    
    for i_inner in range(n_repeat):
        
        print('inner loop %i/%i' % (i_inner+1,n_repeat))
        
        kf = KFold(n_splits=n_splits,shuffle=True)
        
        
        for i1, (i_train, i_test) in enumerate(kf.split(x)):
        
            x_train = x.iloc[i_train,:]
            y_train = y.iloc[i_train]
            x_test = x.iloc[i_test,:]
            y_test = y.iloc[i_test]
                
            
            #----------------apply scaling: all variables to mean=0, std=1--------------
            if use_scaling:
                scaler = StandardScaler()
                scaler.fit(x_train)
                
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            
                x_all = scaler.transform(x)
            #----------------apply PCA: retain 95% of the variance--------------------
            if use_PCA:
                pca = PCA(.95)
                pca.fit(x_train)
                
                x_train = pca.transform(x_train)
                x_test = pca.transform(x_test)
            
                x_all = pca.transform(x_all)
     
            
            reg1.fit(x_train,y_train)
            
            y_pred_train = reg1.predict(x_train)
            y_pred_test = reg1.predict(x_test)
            
            pearsonR = pearsonr(y_pred_test,y_test)[0]
            pearson_R_scores_cripple[i_outer,i_inner*n_splits:(i_inner+1)*n_splits] = pearsonR
            
            importance_sum += reg1.feature_importances_
            
    i_sort = np.argsort(importance_sum)[::-1]
    for i_ in np.arange(len(i_sort)):
        print('%i: %s' % (i_,np.array(features_use_)[i_sort][i_]))
     
mean = pearson_R_scores_cripple.mean(axis=1)
std = pearson_R_scores_cripple.std(axis=1)
# q_1 = np.quantile(pearson_R_scores_cripple,.1,axis=1)
# q_9 = np.quantile(pearson_R_scores_cripple,.9,axis=1)
q_1 = mean-std
q_9 = mean+std


cmap = plt.cm.tab10
fig,ax = plt.subplots(1,figsize=(6,4))

ax.plot([0,n_iter-2],[mean[0],mean[0]],'r--')
ax.fill_between([0,n_iter-2],[mean[0]-std[0],mean[0]-std[0]],[mean[0]+std[0],mean[0]+std[0]],color='r',alpha=.2)

ax.plot(np.arange(0,n_iter)[:-1],mean[1:],'-',color=cmap(0),label='Cripple')
ax.fill_between(np.arange(0,n_iter)[:-1],q_1[1:],q_9[1:],color=cmap(0),alpha=.2)

ax.set_xticks(np.arange(4))
ax.set_xticklabels(['Without tides','Without coastal properties',
                    'Without currents','Without anthropogenic\ndensity information'],rotation=45)
ax.set_ylabel('Pearson R')
fig.tight_layout()

dict_results = {}
dict_results['pearson_R_scores'] = pearson_R_scores_cripple

SAVE = True
if SAVE:
    filename_pickle = 'cripple_analysis_top10_leavecats.pickle'
    outfile = open('datafiles/' + filename_pickle,'wb')
    pickle.dump(dict_results,outfile)
    outfile.close()     