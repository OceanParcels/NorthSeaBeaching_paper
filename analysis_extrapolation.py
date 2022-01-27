#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:42:25 2021

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

# def find_features_not_containing(list_features,string_):
#     return np.array([False if string_ in feature_ else True for feature_ in list_features])

# def cluster_select_features_2(x,threshold=2.89,include_model_feat=True,separate_model_weather=False,exclude=np.array([2,3])):

#     if separate_model_weather:
#         i_nonmodel_features_ = np.where(find_features_not_containing(x.keys(),'beaching'))[0]
#     else:
#         i_nonmodel_features_ = np.where(find_features_not_containing(x.keys(),'xxxxxx'))[0]
    
#     i_nonmodel_features_ = np.setdiff1d(i_nonmodel_features_,exclude)

#     fig1,ax1 = plt.subplots(1,figsize=(20,15))
#     fig2,ax2 = plt.subplots(1,figsize=(20,15))
    
#     corr = spearmanr(x.iloc[:,i_nonmodel_features_]).correlation
#     corr_linkage = hierarchy.ward(corr)
#     dendro = hierarchy.dendrogram(
#         corr_linkage, labels=list(x.iloc[:,i_nonmodel_features_].keys()), ax=ax1, leaf_rotation=90,  leaf_font_size=8,
#     )
#     dendro_idx = np.arange(0, len(dendro['ivl']))
    
#     ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
#     ax2.set_xticks(dendro_idx)
#     ax2.set_yticks(dendro_idx)
#     ax2.set_xticklabels(dendro['ivl'], rotation='vertical',fontsize=8)
#     ax2.set_yticklabels(dendro['ivl'],fontsize=6)
    
#     fig1.tight_layout()
#     fig2.tight_layout()
#     fig2.subplots_adjust(bottom=.2)
#     fig1.subplots_adjust(bottom=.3)
    
#     cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
#     cluster_id_to_feature_ids = defaultdict(list)
#     for idx, cluster_id in enumerate(cluster_ids):
#         cluster_id_to_feature_ids[cluster_id].append(idx)
#     selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
#     selected_features_names = x.iloc[:,i_nonmodel_features_].keys()[selected_features]
    
#     clusters_out = []
#     for cluster_ in cluster_id_to_feature_ids.values():
#         clusters_out.append(list(i_nonmodel_features_[cluster_]))
    
#     if separate_model_weather:
#         if include_model_feat:
#             clusters_out.append(list(np.where(find_features_containing(x.keys(),'beaching_f'))[0]))
#             clusters_out.append(list(np.where(find_features_containing(x.keys(),'beaching_p'))[0]))
#             clusters_out.append(list(np.where(find_features_containing(x.keys(),'beaching_r'))[0]))
            
#     return np.array(selected_features),selected_features_names,clusters_out,fig1,fig2 #cluster_id_to_feature_ids.values()
    

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


# def calculate_most_common_feature(best_feature_in_cluster):
#     best_feature = []
#     for i1 in range(best_feature_in_cluster.shape[1]):
#         a,b, = np.unique(best_feature_in_cluster[:,i1],return_counts=True)
#         index_feat = a[np.argmax(b)]
#         best_feature.append(index_feat)
#     return np.array(best_feature,dtype=int)


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
            

# filename = 'pickle_files/regression_table_180_452_20210629.pickle'
filename = '../NorthSeaBeaching_paper/pickle_files/regression_table_180_468_20211213.pickle'

with open(filename, 'rb') as f:
    regression_table = pickle.load(f)

regression_table_ = regression_table.dropna(axis=0,how='any',subset=['kg/m']).copy()
regression_table_ = normalize_beaching_variables(regression_table_)

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

data_Lag = load('../NorthSeaBeaching_paper/01_figure_data/fig2_391_202112141146.pickle')
labels_Lag = data_Lag['labels']
features_use = labels_Lag[::-1][:8]

# features_use = ['tide_std_030','tide_max_003','dot_mesh_coast','coastal_length_050','tide_tour_max','beaching_r_tau75_050_001',
#                 'dot_currents_min_000_030','beaching_f_tau75_100_003','dot_currents_max_100_003','beaching_p_tau25_050_009']

x = regression_table_.loc[:,features_use]
y = np.log10(regression_table_.loc[:,'kg/m'])

n_repeat = 10
feature_importance_score = np.zeros(len(features_use))
feature_importance_score_mat = np.zeros([n_repeat*5,len(features_use)])

y_test_tot = np.array([])
y_pred_tot = np.array([])

c = 0
array_pearsonR = []

data_fig1 = {}
data_fig1['y_test'] = {}
data_fig1['y_pred'] = {}
data_fig1['R_test'] = {}

for i_outer in range(n_repeat):
    
    print('outer loop %i/%i' % (i_outer+1,n_repeat))
    
    kf = KFold(n_splits=5,shuffle=True)
    
    if i_outer == n_repeat-1:
        fig,ax = plt.subplots(1,figsize=(5,7))
    
    
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
        
        y_test_tot = np.append(y_test_tot,y_test)
        y_pred_tot = np.append(y_pred_tot,y_pred_test)
        
        pearsonR = pearsonr(y_pred_test,y_test)[0]
        array_pearsonR.append(pearsonR)
      
        if regressor == 'RFR':
            feature_importance_score += reg1.feature_importances_
      
        elif regressor == 'GLM' or regressor == 'GPR':
            feature_importance_score += permutation_importance(reg1, x_test, y_test,n_repeats=30,random_state=0).importances_mean

        
        feature_importance_score_mat[c,:] = feature_importance_score

        
        if i_outer == n_repeat-1:
            plt.plot(y_test,y_pred_test,'o',label='test fold %i, R: %f' %(i1,pearsonR))
        
            data_fig1['y_test'][i1] = y_test
            data_fig1['y_pred'][i1] = y_pred_test
            data_fig1['R_test'][i1] = pearsonR 
            
        print(i1)
        c+=1
    

tot_pearsonR = pearsonr(y_test_tot,y_pred_tot)[0]
array_pearsonR = np.array(array_pearsonR)
plt.xlabel('True value [log10(kg/km)]')
plt.ylabel('Predicted value [log10(kg/km)]')
plt.axis('equal')
plt.plot([0,1.1*y.max()],[0,1.1*y.max()],'k--',label='1:1')
PLOT_STD = True
if PLOT_STD:
    n_std = 2
    estim_var = 0.08
    log10_std_y = np.sqrt(estim_var)
    
    dy = n_std*log10_std_y
    minval = 0
    maxval = 1.1*y.max()            
    y1u = minval+dy
    y2u = maxval+dy
    
    y1l = minval-dy
    y2l = maxval-dy
    
    plt.plot([minval,maxval],[y1u,y2u],'r--',label='2x std from variogram',zorder=0)
    plt.plot([minval,maxval],[y1l,y2l],'r--',zorder=0)

    true_values_upper = y_test_tot + dy
    true_values_lower = y_test_tot - dy
    n_within = ((y_pred_tot < true_values_upper) & (y_pred_tot > true_values_lower)).sum() / len(y_pred_tot)
    print('Fraction of values within error bounds: %f' % n_within)

plt.legend()   
plt.title('Pearson R: %f +- %f' % (array_pearsonR.mean(),array_pearsonR.std())) 

print('R +- sigma: %f, %f' % (array_pearsonR.mean(),array_pearsonR.std()))

data_fig1['array_pearsonR'] = array_pearsonR

SAVE = False
if SAVE:
    now_ = datetime.now()
    filename_fig = 'fig1_%i_%4.4i%2.2i%2.2i%2.2i%2.2i.pickle' % (len(features_use),now_.year,now_.month,now_.day,now_.hour,now_.minute)
    outfile = open('01_figure_data/' + filename_fig,'wb')
    pickle.dump(data_fig1,outfile)
    outfile.close()  

final_model = reg1.fit(x_all,y)

#%% Part 2: Calculate normal vectors

import pyproj 
from functools import partial


project = partial(
    pyproj.transform,
    pyproj.Proj(proj='latlong',datum='WGS84'), # the standard lon/lat source coordinate system
    pyproj.Proj('epsg:28992')) # destination coordinate system. Rijksdriehoek


def give_element_within_bounds2(geom_split, box_analyze, tol=1.01):
    tol_lower = 1/tol
    tol_upper = tol
    
    i_return = np.array([],dtype=int)
    c = 0
    for geom_ in geom_split:
        x_ = np.array(geom_.xy[0])
        y_ = np.array(geom_.xy[1])
        
        if (x_.min()>= tol_lower*np.array(box_analyze)[:,0].min()) and (y_.min()>= tol_lower*np.array(box_analyze)[:,1].min()) and (
                x_.max()<= tol_upper*np.array(box_analyze)[:,0].max()) and (y_.max()<= tol_upper*np.array(box_analyze)[:,1].max()):
            i_return = np.append(i_return,c)
        c+=1
    return i_return


def give_element_outside_bounds(geom_split,box_analyze):
  
    i_return = np.array([],dtype=int)
    c = 0
    for geom_ in geom_split:
        if box_analyze.contains(geom_):
            pass
        else:
            i_return = np.append(i_return,c)
              
        c+=1
    return i_return    

#-------------select the right indices to take into account
data_coastal_lengths = xr.open_dataset('datafiles/netcdf_coastal_lengths.nc')
coastal_length = data_coastal_lengths['coastline_length'].values
lons = data_coastal_lengths['lon']
lats = data_coastal_lengths['lat']
dlon = lons[1]-lons[0]
dlat = lats[1]-lats[0]
lons_edges = lons-.5*dlon
lons_edges = np.append(lons_edges,lons_edges[-1]+dlon)#[i_lons]
lats_edges = lats-.5*dlat
lats_edges = np.append(lats_edges,lats_edges[-1]+dlat)#[i_lats]
meshPlotx,meshPloty = np.meshgrid(lons_edges,lats_edges)
X,Y = np.meshgrid(lons,lats)

fig = plt.figure(figsize=(10,8),dpi=120)  
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')

i_lon = [210,236+1]
i_lat = [170,203+1]

ax.pcolormesh(meshPlotx[i_lat[0]:i_lat[1]+1,i_lon[0]:i_lon[1]+1],meshPloty[i_lat[0]:i_lat[1]+1,i_lon[0]:i_lon[1]+1],data_coastal_lengths['coastline_length'][i_lat[0]:i_lat[1],i_lon[0]:i_lon[1]])

indices_analyze = np.where( (X >= lons[i_lon[0]].values) & (X < lons[i_lon[1]].values) & 
                           (Y >= lats[i_lat[0]].values) & (Y < lats[i_lat[1]].values) & (coastal_length > 0) )

ax.plot(lons[indices_analyze[1]],lats[indices_analyze[0]],'kx')


#---------------calculate normal vectors
water_front = np.array([[ 3.5  , 51.   ],
                       [ 3.546, 51.404],
                       [ 3.572, 51.44 ],
                       [ 3.683, 51.6  ],
                       [ 3.722, 51.664],
                       [ 3.828, 51.739],
                       [ 3.876, 51.805],
                       [ 4.043, 51.826],
                       [ 4.11 , 51.844],
                       [ 4.8  , 52.4  ],
                       [ 4.77 , 52.963],
                       [ 4.77 , 52.988],
                       [ 4.858, 53.184],
                       [ 4.91 , 53.23 ],
                       [ 5.038, 53.298],
                       [ 5.098, 53.306],
                       [ 5.158, 53.349],
                       [ 5.547, 53.44 ],
                       [ 5.624, 53.44 ],
                       [ 5.968, 53.463],
                       [ 6.119, 53.465],
                       [ 6.395, 53.522],
                       [ 7.1  , 53.682],
                       [ 7.818, 53.777],
                       [ 7.818, 51.   ],
                       [ 3.5  , 51.   ]])

shpfilename = shpreader.natural_earth(resolution='10m',
                                      category='physical',
                                      name='coastline')
# waterfront: beaches adjacent to the North Sea, wadden sea / inland waters are removed
box_waterfront = Polygon(water_front)
line_waterfront = LineString(water_front)
line_waterfront_offset = line_waterfront.parallel_offset(-0.01) #offset the waterfront line outwards
box_waterfront_offset = Polygon(line_waterfront_offset)


def calculate_coastal_section_length(coast_geom):
    if coast_geom.intersects(box_poly):
        # coast_.geometry
        print('intersect at %i' %c)
        print(lon_left,lat_lower)
        # break
        
        box_linestring = LineString(box_analyze)
        
        #if the cells lies within the land domain defined as the 'box_waterfront', skip
        if box_waterfront.contains(box_linestring):
            pass
        
        #if the waterfront box intersects the cell, take an intersection and calculate length
        #of the section outside the waterfront box
        elif line_waterfront.intersects(box_linestring):
            
            split_waterfront = split(Polygon(box_linestring),line_waterfront)
            i_waterfront = give_element_outside_bounds(split_waterfront,box_waterfront_offset)
           
            for i_coast2_ in i_waterfront:
                geom_waterfront = split_waterfront[i_coast2_]
                coastline_split = split(coast_geom,geom_waterfront)
    
                indices_in_box = give_element_within_bounds2(coastline_split,box_analyze,tol=1.00)
                
                for index_in_box in indices_in_box:
                    coastline_in_box = coastline_split[index_in_box]
                    
                    coastline_lons = np.array(coastline_in_box.xy[0])
                    coastline_lats = np.array(coastline_in_box.xy[1])
                        
                    ax.plot(coastline_lons,coastline_lats,'ro',transform=ccrs.PlateCarree())
                    
                    dict_coastlines['lons'][c].append(coastline_lons)
                    dict_coastlines['lats'][c].append(coastline_lats)
                    
                    # if np.isnan(coastline_length[i2,i1]):
                    #     coastline_length[i2,i1] = transform(project,coastline_in_box).length
                    # else:
                    #     coastline_length[i2,i1] += transform(project,coastline_in_box).length                        

        #otherwise, the box is outside of the waterfront box, just take the length of the sections
        else:
            coastline_split = split(coast_geom,box_linestring)

            indices_in_box = give_element_within_bounds2(coastline_split,box_analyze,tol=1.00)
            
            for index_in_box in indices_in_box:
                coastline_in_box = coastline_split[index_in_box]
                
                coastline_lons = np.array(coastline_in_box.xy[0])
                coastline_lats = np.array(coastline_in_box.xy[1])
                    
                ax.plot(coastline_lons,coastline_lats,'ro',transform=ccrs.PlateCarree())
                
                dict_coastlines['lons'][c].append(coastline_lons)
                dict_coastlines['lats'][c].append(coastline_lats)
                
                # if np.isnan(coastline_length[i2,i1]):
                #     coastline_length[i2,i1] = transform(project,coastline_in_box).length
                # else:
                #     coastline_length[i2,i1] += transform(project,coastline_in_box).length
          
    if box_poly.contains(coast_geom):
        print('coastline contained in box')
        # coastline_length[i2,i1] += transform(project,coast_geom).length

dict_coastlines = {}
dict_coastlines['lons'] = [[] for i in range(len(indices_analyze[0]))]
dict_coastlines['lats'] = [[] for i in range(len(indices_analyze[0]))]
dict_coastlines['lon'] = [[] for i in range(len(indices_analyze[0]))]
dict_coastlines['lat'] = [[] for i in range(len(indices_analyze[0]))]
dict_coastlines['n'] = [[] for i in range(len(indices_analyze[0]))]

fig = plt.figure(figsize=(7,7),dpi=120)  
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')

c = 0
for i1,i2,lon_left,lon_right,lat_lower,lat_upper in zip (indices_analyze[1],indices_analyze[0],lons_edges[indices_analyze[1]],lons_edges[indices_analyze[1]+1],
                                                         lats_edges[indices_analyze[0]],lats_edges[indices_analyze[0]+1]):
    
    dict_coastlines['lon'][c] = lons[i1].values
    dict_coastlines['lat'][c] = lats[i2].values
    
    reader = shpreader.Reader(shpfilename)
    coastlines = reader.records()
   
    box_analyze = [[lon_left, lat_lower], [lon_left, lat_upper], 
                   [lon_right, lat_upper], [lon_right, lat_lower], [lon_left, lat_lower]]
    box_poly = shapely.geometry.Polygon(box_analyze)
     
    for coast_ in coastlines:

        calculate_coastal_section_length(coast_.geometry)
    
    c += 1
    print(i1)


fig = plt.figure(figsize=(7,7),dpi=120)  
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.pcolormesh(meshPlotx[i_lat[0]:i_lat[1]+1,i_lon[0]:i_lon[1]+1],meshPloty[i_lat[0]:i_lat[1]+1,i_lon[0]:i_lon[1]+1],data_coastal_lengths['coastline_length'][i_lat[0]:i_lat[1],i_lon[0]:i_lon[1]])

flip_nr = [9,12,50,59,60,64]

cut_nr = [41]
cut_i_discard = [1]

for i1 in range(len(dict_coastlines['lon'])):
    
    lons_ = np.array([])
    lats_ = np.array([])
    
    lon_ = dict_coastlines['lon'][i1]
    lat_ = dict_coastlines['lat'][i1]
    
    for i2 in range(len(dict_coastlines['lons'][i1])):
        if i1 in cut_nr and i2 in cut_i_discard:
            print('pass...')
            pass
        else:
            lons_ = np.append(lons_, np.array(dict_coastlines['lons'][i1][i2]) )
            lats_ = np.append(lats_, np.array(dict_coastlines['lats'][i1][i2]) )
            

    x_lons = lons_ * np.cos(lat_*(np.pi/180)) #convert to meters, i.e. compress longitude
    y_lats = lats_
    
    x_ = x_lons - x_lons.mean()
    y_ = y_lats - y_lats.mean()
    svd_ = np.linalg.svd(np.array([x_,y_]))
    
    # print('----')
    # print(x_,y_)
    # print(svd_[0][:,1])
    
    normal_vec = svd_[0][:,1]
    
    if normal_vec[0] < 0: #all vectors point right, don't copy this for other domains than the Netherlands..
        normal_vec = -normal_vec
    if i1 in flip_nr:
        normal_vec = -normal_vec
        
    scale_ = 0.001
    normal_vec_lon = np.array([lon_,lon_+scale_*(normal_vec[0]*(1.11e2 * np.cos(lat_*(np.pi/180))))])
    normal_vec_lat = np.array([lat_,lat_+scale_*(normal_vec[1]*1.11e2)])

    ax.plot(lons_,lats_,'ro')
    ax.plot(lon_,lat_,'kx')
    ax.plot(normal_vec_lon,normal_vec_lat,'k-',zorder=1000)
    
    ax.text(lon_,lat_,'%i'%i1)
    
    dict_coastlines['n'][i1] = normal_vec


def find_normal_vector_landborder(lon_unique,lat_unique,fieldMesh_x,fieldMesh_y,landBorder,radius=30):


    plt.figure()
    plt.pcolormesh(meshPlotx,meshPloty,landBorder,alpha=.8)
    
    normal_vecs = np.zeros([2,len(lon_unique)])
    
    for i1, (lon_, lat_) in enumerate(zip(lon_unique,lat_unique)):
        
        d_x = (fieldMesh_x - lon_)*np.cos((np.pi/180)*lat_) * 1.11e2
        d_y = (fieldMesh_y - lat_) * 1.11e2
    
        i_close = (np.sqrt(d_x**2 + d_y**2) < radius)
    
        lons_close = fieldMesh_x[(landBorder == 2) & (i_close)]
        lats_close = fieldMesh_y[(landBorder == 2) & (i_close)]
    
        x_lons = lons_close * np.cos(lat_*(np.pi/180)) #convert to meters, i.e. compress longitude
        y_lats = lats_close
        x_ = x_lons - x_lons.mean()
        y_ = y_lats - y_lats.mean()
        svd_ = np.linalg.svd(np.array([x_,y_]))
        
        normal_vec = svd_[0][:,1]
        
        if normal_vec[1] > 0: #all vectors point downwards, don't copy this for other domains than the Netherlands..
            normal_vec = -normal_vec

        normal_vecs[:,i1] = normal_vec
    
        scale_ = 0.005
        normal_vec_lon = np.array([lon_,lon_+scale_*(normal_vec[0]*(1.11e2 * np.cos(lat_*(np.pi/180))))])
        normal_vec_lat = np.array([lat_,lat_+scale_*(normal_vec[1]*1.11e2)])

        plt.plot(lon_,lat_,'kx')
        plt.plot(normal_vec_lon,normal_vec_lat,'k-',zorder=1000)
    
        if i1 == 8 or i1 == 32:
    
            plot_close = i_close.copy() * 1.
            plot_close[plot_close == 0] = np.nan
            plt.pcolormesh(meshPlotx,meshPloty,plot_close,alpha=.4,cmap=plt.cm.tab10,vmin=0,vmax=2)
            plt.plot(lons_close,lats_close,'rx')
            
    return normal_vecs

def getLandBorder(landMask,val_add):
    """
    Function to obtain a mask of the land which borders ocrean, uses the landmask and searches for 
    boundaries with the sea (horiz./vert. adjacent cells only)
    TODO: check for cyclic boundaries
    TODO: check diagonal cells as well?
    """
    
    n_lat = landMask.shape[0]
    n_lon = landMask.shape[1]
    
#    borderMask = np.zeros([n_lat,n_lon])

    for i1 in range(n_lat):
        for i2 in range(n_lon):
            
            check_bot = True
            check_top = True
            check_left = True
            check_right = True
            
            # check whether land is located at boundary
            if i1 == 0:
                check_top = False
            if i1 == n_lat-1:
                check_bot = False
            if i2 == 0:
                check_left = False
            if i2 == n_lon-1:
                check_right = False
                
            # check whether cell is land, if so look for coast
            if landMask[i1,i2] == 1:
                
                if check_top:
                    if (landMask[i1-1,i2] == 0) or (landMask[i1-1,i2] >= 2):
                        landMask[i1,i2] = -1
                if check_bot:
                    if (landMask[i1+1,i2] == 0) or (landMask[i1+1,i2] >= 2):
                        landMask[i1,i2] = -1
                if check_left:
                    if (landMask[i1,i2-1] == 0) or (landMask[i1,i2-1] >= 2):
                        landMask[i1,i2] = -1
                if check_right:
                    if (landMask[i1,i2+1] == 0) or (landMask[i1,i2+1] >= 2):
                        landMask[i1,i2] = -1
    landMask[landMask == -1] = val_add
            
    return landMask

file_landMask = './datafiles/datafile_trueLandMask_%ix_%iy' % (len(lons),len(lats))
landMask = np.loadtxt(file_landMask)

landBorder = landMask.copy()
landBorder = getLandBorder(landBorder,2)

normal_vecs_mesh = find_normal_vector_landborder( np.array(dict_coastlines['lon']), np.array(dict_coastlines['lat']),X,Y, landBorder)
dict_coastlines['n_mesh'] = normal_vecs_mesh
dict_coastlines['n'] = np.array(dict_coastlines['n']).T

dict_coastlines['dot_n'] = np.array([np.dot(dict_coastlines['n'][:,i1], dict_coastlines['n_mesh'][:,i1]) for i1 in range(len(dict_coastlines['lon']))])

ax.scatter(dict_coastlines['lon'],dict_coastlines['lat'],c=dict_coastlines['dot_n'])

SAVE = False
if SAVE:
    filename_pickle = 'material_flow_coastlines.pickle'
    outfile = open('datafiles/' + filename_pickle,'wb')
    pickle.dump(dict_coastlines,outfile)
    outfile.close()  
    
    
    
    
#%% Part 3: create regression table for the artificial data
def find_closest_date(arr,date):
    diff = np.array([arr[i]-date for i in range(len(arr))])
    return np.argmin(np.abs(diff))

def calculate_variables(lon,lat,date,data,dist_select,time_lag,variable_name,tmp_X,tmp_Y,quantity='all',use_land_mask=False,land_mask=None):

    i_date_closest = None
    if 'time' in data.keys():
        tmp_time = data['time']
        
        i_date_closest = find_closest_date(tmp_time.data,date)
        i_date_start = find_closest_date(tmp_time.data,date-timedelta(days=time_lag))
        
        i_date_start = max(0,i_date_start)
        i_date_closest = min(len(tmp_time.data),i_date_closest)
        if i_date_start == 0:
            print('warning: starting index 0')
        if i_date_closest == len(tmp_time.data):
            print('warning: date at the end of the data array')    
    
    
    dist_mat = np.sqrt(((tmp_X - lon)*1.11e2*np.cos(lat*(np.pi/180)))**2 + ((tmp_Y - lat)*1.11e2)**2)
    
    
    def closest_point(dist_mat,use_land_mask,land_mask):
        dist_mat_min = None
        if use_land_mask==True:
            dist_mat_min = np.min(dist_mat[~land_mask])
            i_select = (dist_mat == dist_mat_min)
        elif type(use_land_mask) == float:
            dist_mat_min = np.min(dist_mat[land_mask < use_land_mask])
            i_select = (dist_mat == dist_mat_min)
        else:
            i_select = (dist_mat == dist_mat.min())       
        return i_select,dist_mat_min
    
    if dist_select == 0: #closest point
        i_select,_ = closest_point(dist_mat,use_land_mask,land_mask)
        assert(i_select.sum() == 1)
    else: # look in a radius
        if use_land_mask==True:
            i_select = (dist_mat < dist_select) & ~land_mask
        elif type(use_land_mask) == float:
            i_select = (dist_mat < dist_select) & (land_mask < use_land_mask)
        else:
            i_select = (dist_mat < dist_select)
        # fall back to closest distance if necessary
        if i_select.sum() == 0:
            i_select,dist_mat_min = closest_point(dist_mat,use_land_mask,land_mask)
            print('no cells within %2.2f km, falling back on closest distance (%2.2f)' % (dist_select,dist_mat_min))
    
    
    if quantity == 'mean':
        fn_quantity = lambda x: np.nanmean(x)
    elif quantity == 'max':
        fn_quantity = lambda x: np.nanmax(x)
    elif quantity == 'all':
        fn_quantity = lambda x: (np.nanmean(x),np.nanmax(x),np.nanmin(x),np.nanstd(x))
    elif quantity == 'sum':
        fn_quantity = lambda x: np.nansum(x)
    else:
        raise RuntimeError('not implemented')

    try:
        if 'time' in data.keys():       
            if i_date_start != i_date_closest: #calculate quantity over a range of times (i.e. lag time)
                if isinstance(variable_name,str): #scalar
                    result = fn_quantity(data[variable_name][i_date_start:i_date_closest+1].data[:,i_select])
                elif isinstance(variable_name,list): #vector -> convert to magnitude (scalar)
                    magnitude = np.sqrt(data[variable_name[0]]**2 + data[variable_name[1]]**2)
                    result = fn_quantity(magnitude[i_date_start:i_date_closest+1].data[:,i_select])
                else:
                    raise RuntimeError('not implemented')            
        
            else: #calculate quantity for a single time
                if isinstance(variable_name,str):
                    result = fn_quantity(data[variable_name][i_date_closest].data[i_select])
                elif isinstance(variable_name,list):
                    magnitude = np.sqrt(data[variable_name[0]]**2 + data[variable_name[1]]**2)
                    result = fn_quantity(magnitude[i_date_closest].data[:,i_select])
                else:
                    raise RuntimeError('not implemented')            
        
        else: #the netcdf does not contain time, space only
            if isinstance(variable_name,str):
                result = fn_quantity(data[variable_name].data[i_select])
            elif isinstance(variable_name,list):
                magnitude = np.sqrt(data[variable_name[0]]**2 + data[variable_name[1]]**2)
                result = fn_quantity(magnitude.data[:,i_select])
            else:
                raise RuntimeError('not implemented')              
            
    except:
        result = np.nan
        print('returning nan')
    
    return result   


def calculate_inproduct(lon,lat,date,data,dist_select,time_lag,variable_name,tmp_X,tmp_Y,quantity='all',PLOT=False,use_land_mask=False,land_mask=None):
    
    tmp_time = data['time']

    i_date_closest = find_closest_date(tmp_time.data,date)
    i_date_start = find_closest_date(tmp_time.data,date-timedelta(days=time_lag))
    
    i_date_start = max(0,i_date_start)
    i_date_closest = min(len(tmp_time.data),i_date_closest)
    
    dist_mat = np.sqrt(((tmp_X - lon)*1.11e2*np.cos(lat*(np.pi/180)))**2 + ((tmp_Y - lat)*1.11e2)**2)


    def closest_point(dist_mat,use_land_mask,land_mask):
        dist_mat_min = None
        if use_land_mask==True:
            dist_mat_min = np.min(dist_mat[~land_mask])
            i_select = (dist_mat == dist_mat_min) & (~land_mask)
        elif type(use_land_mask) == float:
            dist_mat_min = np.min(dist_mat[land_mask < use_land_mask])
            i_select = (dist_mat == dist_mat_min)
        else:
            i_select = (dist_mat == dist_mat.min())     
            
        # if multiple points are equally close, just select the first one
        if i_select.sum() > 1:
            where_ = np.where(i_select)
            i_select[where_[0][1:],where_[1][1:]] = False
            
        return i_select,dist_mat_min
    
    if dist_select == 0: #closest point
        i_select,_ = closest_point(dist_mat,use_land_mask,land_mask)
        assert(i_select.sum() == 1)
    else: # look in a radius
        if use_land_mask==True:
            i_select = (dist_mat < dist_select) & ~land_mask
        elif type(use_land_mask) == float:
            i_select = (dist_mat < dist_select) & (land_mask < use_land_mask)
        else:
            i_select = (dist_mat < dist_select)
        # fall back to closest distance if necessary
        if i_select.sum() == 0:
            i_select,dist_mat_min = closest_point(dist_mat,use_land_mask,land_mask)
            print('no cells within %2.2f km, falling back on closest distance (%2.2f)' % (dist_select,dist_mat_min))
    
    
    i_which_coast = np.where( np.isclose(lon_,dict_coastlines['lon']) )[0]
    # i_which_coast = np.where(lon_ == beach_orientations['lon'])[0]
    if i_which_coast.size != 0:
        i_coastal_segment = i_which_coast[0]
    else:
        raise RuntimeError('coastal segment not found')
        
    
    if quantity == 'mean':
        fn_quantity = lambda x: np.nanmean(x)
    elif quantity == 'max':
        fn_quantity = lambda x: np.nanmax(x)
    elif quantity == 'all':
        fn_quantity = lambda x: (np.nanmean(x),np.nanmax(x),np.nanmin(x))
    else:
        raise RuntimeError('not implemented')    
    
    if i_date_start != i_date_closest: #calculate quantity over a range of times (i.e. lag time)
        vec_u = data[variable_name[0]][i_date_start:i_date_closest+1].data[:,i_select]
        vec_v = data[variable_name[1]][i_date_start:i_date_closest+1].data[:,i_select]
        vec_ = np.array([vec_u[~np.isnan(vec_u)],vec_v[~np.isnan(vec_v)]]).T
        normal_vec = dict_coastlines['n'][:,i_coastal_segment]    
        dot_prod = np.array([np.dot(vec_[i,:], normal_vec) for i in range(len(vec_))])
    else:
        print('calculating in product for single time')
        vec_u = data[variable_name[0]][i_date_closest].data[i_select]
        vec_v = data[variable_name[1]][i_date_closest].data[i_select]
        vec_ = np.array([vec_u[~np.isnan(vec_u)],vec_v[~np.isnan(vec_v)]]).T
        normal_vec = dict_coastlines['n'][:,i_coastal_segment]    
        dot_prod = np.array([np.dot(vec_[i,:], normal_vec) for i in range(len(vec_))])        
        
    
    
    if PLOT: #validation plots
        lons_plot = tmp_X[0,:]
        lats_plot = tmp_Y[:,0]
        lons_spacing = lons_plot[1] - lons_plot[0]
        lats_spacing = lats_plot[1] - lats_plot[0]
        lons_mesh = np.append(lons_plot -.5*lons_spacing, lons_plot[-1]+.5*lons_spacing)
        lats_mesh = np.append(lats_plot -.5*lats_spacing, lats_plot[-1]+.5*lats_spacing)
        X_plot,Y_plot = np.meshgrid(lons_mesh,lats_mesh)
        
        scale_ = 0.001
        normal_vec_lon = np.array([lon,lon+scale_*(normal_vec[0]*(1.11e2 * np.cos(lat*(np.pi/180))))])
        normal_vec_lat = np.array([lat,lat+scale_*(normal_vec[1]*1.11e2)])
        
        #set 1
        vec_u = data[variable_name[0]][i_date_start].data[i_select]
        vec_v = data[variable_name[1]][i_date_start].data[i_select]
        vec_ = np.array([vec_u,vec_v]).T
        dot_prod_1 = np.array([np.dot(vec_[i,:], normal_vec) for i in range(len(vec_))])
        dot_field = np.zeros(tmp_X.shape)
        dot_field[i_select] = dot_prod_1
        dot_field[dot_field==0] = np.nan
        
        level_max = np.nanmax(np.abs(dot_field))
        # levels = np.linpspace(-levels_max,levels_max,50)
        fig = plt.figure(figsize=(7,5),dpi=120)     
        ax = plt.axes(projection=ccrs.PlateCarree())    
        cmesh = plt.pcolormesh(X_plot,Y_plot,dot_field,cmap=plt.cm.coolwarm,vmin=-level_max,vmax=level_max)
        plt.quiver(tmp_X,tmp_Y,data[variable_name[0]][i_date_start],data[variable_name[1]][i_date_start],scale=300)
        ax.plot(normal_vec_lon,normal_vec_lat,'g-')
        ax.set_extent((3.2,6.8,51,54))
        ax.coastlines(resolution='10m')
        plt.colorbar(cmesh)
        plt.title('In-product of wind normal to coastline\nRadius = %ikm' % dist_select)

        #set 2
        vec_u = data[variable_name[0]][i_date_closest].data[i_select]
        vec_v = data[variable_name[1]][i_date_closest].data[i_select]
        vec_ = np.array([vec_u,vec_v]).T
        dot_prod_1 = np.array([np.dot(vec_[i,:], normal_vec) for i in range(len(vec_))])
        dot_field = np.zeros(tmp_X.shape)
        dot_field[i_select] = dot_prod_1
        dot_field[dot_field==0] = np.nan
        
        level_max = np.nanmax(np.abs(dot_field))        
        fig = plt.figure(figsize=(7,5),dpi=120)     
        ax = plt.axes(projection=ccrs.PlateCarree())    
        cmesh = plt.pcolormesh(X_plot,Y_plot,dot_field,cmap=plt.cm.coolwarm,vmin=-level_max,vmax=level_max)
        plt.quiver(tmp_X,tmp_Y,data[variable_name[0]][i_date_closest],data[variable_name[1]][i_date_closest],scale=100)
        ax.plot(normal_vec_lon,normal_vec_lat,'g-')
        ax.set_extent((3.2,6.8,51,54))
        ax.coastlines(resolution='10m')
        plt.colorbar(cmesh)

        if type(use_land_mask) == float:
            fig = plt.figure(figsize=(7,5),dpi=120)     
            ax = plt.axes(projection=ccrs.PlateCarree())    
            plt.pcolormesh(X_plot,Y_plot,(land_mask < use_land_mask))
            ax.plot(normal_vec_lon,normal_vec_lat,'g-')
            ax.set_extent((3.2,6.8,51,54))
            ax.coastlines(resolution='10m')            
    
    if dot_prod.size == 0:
        result = np.nan
        print('returning nan')
    else:
        result = fn_quantity(dot_prod)
    
    return result
    

def load_data_currents(year):
    file_july = '/Users/kaandorp/Data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/%4.4i07.nc'%year
    file_aug = '/Users/kaandorp/Data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/%4.4i08.nc'%year
    
    data_currents = xr.load_dataset(file_july)
    data_currents = xr.concat((data_currents,xr.load_dataset(file_aug)), 'time')
    
    return data_currents.squeeze()
    

with open('datafiles/material_flow_coastlines.pickle', 'rb') as f:
    dict_coastlines = pickle.load(f)

features_use = ['tide_std_030', 'tide_max_003', 'beaching_f_tau25_050_009',
       'coastal_length_050', 'dot_mesh_coast', 'tide_tour_max',
       'beaching_p_tau150_050_030', 'in_tide_max_003']

regression_table_material_flow = pd.DataFrame(columns=['lon','lat','time']+features_use)
    
years = np.arange(2014,2020)
# years = [2019]
# dates = np.array([],dtype=np.datetime64)
c=0
for yr in years:
#     dates = np.append(dates,np.array(pd.date_range(datetime(year_,8,1),datetime(year_,8,31))))

# for date in dates:    
    # date_ = pd.Timestamp( datetime(2019, 8, 1, 11) )
    # date_ = pd.Timestamp(date)
    # yr = pd.Timestamp(date_).year
    data_tides = xr.load_dataset('datafiles/tides_%4i.nc' % yr)
    X_tides,Y_tides = np.meshgrid(data_tides['lon'].data,data_tides['lat'].data)
    tides_land_mask = data_tides['mask_land'].values

    # data_tides,X_tides,Y_tides,tides_land_mask = initialize_tides(yr)
    data_currents = load_data_currents(yr)
    X_curr,Y_curr = np.meshgrid(data_currents['longitude'].data,data_currents['latitude'].data)
    data_beaching_f = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/histograms/beaching_hist_11_f.nc')
    data_beaching_p = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/histograms/beaching_hist_11_p.nc')
    data_beaching_r = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/histograms/beaching_hist_11_r.nc')
    X_beaching,Y_beaching = np.meshgrid(data_beaching_f['lon'].data,data_beaching_f['lat'].data)   
    data_coastal_length = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/datafiles/netcdf_coastal_lengths.nc')
    X_coast, Y_coast = np.meshgrid(data_coastal_length['lon'],data_coastal_length['lat'])
    currents_landmask = np.loadtxt('datafiles/datafile_trueLandMask_297x_375y').astype(bool)
    
    # dates = pd.date_range(datetime(yr,8,1),datetime(yr,8,31))
    dates = pd.date_range(datetime(yr,8,1),datetime(yr,8,31))
    for date in dates:
        date_ = pd.Timestamp(date)
        print(date_)
        
        for i1,(lon_,lat_,dot_) in enumerate(zip(dict_coastlines['lon'],dict_coastlines['lat'],dict_coastlines['dot_n'])):
            
            _, _, _, tide_std_030 = calculate_variables(lon_, lat_, date_, data_tides, 0, 30, 'tide', X_tides, Y_tides, quantity='all',use_land_mask=True,land_mask=tides_land_mask)
        
            _, tide_max_003, _, _ = calculate_variables(lon_, lat_, date_, data_tides, 0, 3, 'tide', X_tides, Y_tides, quantity='all',use_land_mask=True,land_mask=tides_land_mask)
            
            beaching_f_tau25 = calculate_variables(lon_,lat_,date_,data_beaching_f,
                                                         50,9,'beaching_tau25',X_beaching,Y_beaching,quantity='sum')

            coastal_length_050 = calculate_variables(lon_, lat_, date_, data_coastal_length, 50, 0, 'coastline_length', X_coast, Y_coast, quantity='sum',use_land_mask=False)

            _, tide_tour_max, _, _ = calculate_variables(lon_, lat_, date_+timedelta(hours=6), data_tides, 0, 0.25, 'tide', X_tides, Y_tides, quantity='all',use_land_mask=True,land_mask=tides_land_mask)
            
        
            beaching_p_tau150 = calculate_variables(lon_,lat_,date_,data_beaching_p,
                                                         50,30,'beaching_tau150',X_beaching,Y_beaching,quantity='sum')
            
            _, in_tide_max_003, _ = calculate_inproduct(lon_,lat_,date_,data_tides,
                                                                 0,3,['tide_U','tide_V'],X_tides,Y_tides,quantity='all',
                                                                 use_land_mask=True,land_mask=tides_land_mask)
                 
            # beaching_r_tau75 = calculate_variables(lon_,lat_,date_,data_beaching_r,
            #                                              50,1,'beaching_tau75',X_beaching,Y_beaching,quantity='sum')    
            
            # _, in_currents_max, _ = calculate_inproduct(lon_,lat_,date_,data_currents,
            #                                             100,3,['uo','vo'],X_curr,Y_curr,quantity='all',use_land_mask=True,land_mask=currents_landmask)
            
            # _, _, in_currents_min = calculate_inproduct(lon_,lat_,date_,data_currents,
            #                                             0,30,['uo','vo'],X_curr,Y_curr,quantity='all',use_land_mask=True,land_mask=currents_landmask)
        
            # ['tide_std_030', 'tide_max_003', 'beaching_f_tau25_050_009',
            # 'coastal_length_050', 'dot_mesh_coast', 'tide_tour_max',
            # 'beaching_p_tau150_050_030', 'in_tide_max_003']                  
            regression_table_material_flow.loc[c] = [lon_,lat_,date_,tide_std_030,tide_max_003,beaching_f_tau25,coastal_length_050,dot_,tide_tour_max,
                                                     beaching_p_tau150,in_tide_max_003]
            
            # print(c)
            c+=1
    
    
filename_rt = 'regression_table_matflow_%3.3i_%3.3i_%4.4i%2.2i%2.2i.pickle' % (regression_table_material_flow.shape[0],regression_table_material_flow.shape[1],datetime.today().year,datetime.today().month,datetime.today().day)
outfile = open(os.path.join('./pickle_files/',filename_rt),'wb')
pickle.dump(regression_table_material_flow,outfile)
outfile.close()   


#%% 
from cartopy import feature

file_train = 'pickle_files/regression_table_180_468_20211213.pickle'
# file_test = 'pickle_files/regression_table_matflow_065_013_20210726.pickle'
file_test = 'pickle_files/regression_table_matflow_12090_011_20211214.pickle'
file_coastlines = 'datafiles/material_flow_coastlines_v2.pickle'


features_use = ['tide_std_030', 'tide_max_003', 'beaching_f_tau25_050_009',
       'coastal_length_050', 'dot_mesh_coast', 'tide_tour_max',
       'beaching_p_tau150_050_030', 'in_tide_max_003']

with open(file_train, 'rb') as f:
    reg_tab_train = pickle.load(f)
with open(file_test, 'rb') as f:
    reg_tab_test = pickle.load(f)
with open(file_coastlines, 'rb') as f:
    dict_coastlines = pickle.load(f)
    
reg_tab_train = reg_tab_train.dropna(axis=0,how='any')

use_scaling = True    
reg1 = RandomForestRegressor(oob_score=True,max_features=.33)

x = reg_tab_train.loc[:,features_use]
y = np.log10(reg_tab_train.loc[:,'kg/m'])

x_test = reg_tab_test.loc[:,features_use]


#----------------apply scaling: all variables to mean=0, std=1--------------
if use_scaling:
    scaler = StandardScaler()
    scaler.fit(x)

    x_train = scaler.transform(x)
    y_train = y.copy()

    x_test = scaler.transform(x_test)
    
reg1.fit(x_train,y_train)

y_pred_test = reg1.predict(x_test)


data_coastal_length = xr.open_dataset('datafiles/netcdf_coastal_lengths.nc')
coastal_length = data_coastal_length['coastline_length'].values
lons = data_coastal_length['lon']
lats = data_coastal_length['lat']

dates_unique = np.unique(reg_tab_test['time'])
n_dates = len(dates_unique)
var = 0.08
sigma = np.sqrt(var)
n_pert = 50

array_y_kg = []
array_litter_dist = np.zeros([65,len(dates_unique)])
array_y_kg_pert = np.zeros([n_pert,len(dates_unique)])


for i3,date_ in enumerate(dates_unique):
    i_date_select = (reg_tab_test['time'] == date_)
    
    y_coastal_length = np.array([])
    lons_list = []
    lats_list = []
    for i1,(lon_,lat_) in enumerate(zip(reg_tab_test['lon'][i_date_select].values,reg_tab_test['lat'][i_date_select].values)):
        i_lon = np.where(lons == lon_)[0][0]
        i_lat = np.where(lats == lat_)[0][0]
        
        y_coastal_length = np.append(y_coastal_length,coastal_length[i_lat,i_lon])
        
        i_dict = np.where( (dict_coastlines['lon'] == lon_) & (dict_coastlines['lat'] == lat_))[0][0]
        # lons_coast = np.array([])
        # lats_coast = np.array([])
        
        lons_list.append([])
        lats_list.append([])
        for lon_,lat_ in zip(dict_coastlines['lons'][i_dict],dict_coastlines['lats'][i_dict]):
            # lons_coast = np.append(lons_coast,lon_ )
            # lats_coast = np.append(lats_coast,lat_ )
            lons_list[i1].append(lon_)
            lats_list[i1].append(lat_)
      
    n_per_date = i_date_select.sum()
    y_pred_select = y_pred_test[i_date_select]
    
    for i2 in range(n_pert):
        noise = np.random.normal(scale=sigma,size=n_per_date)
        y_kg_km_pert = 10**(y_pred_select + noise)
        y_kg_pert = y_kg_km_pert * (y_coastal_length/1000)
    
        array_y_kg_pert[i2,i3] = (y_kg_pert.sum())
        
    y_kg_km = 10**(y_pred_select)
    
    array_litter_dist[:,i3] = y_kg_km
    y_kg = y_kg_km * (y_coastal_length/1000)
    
    array_y_kg.append(y_kg.sum())
    
    print(date_,y_kg.sum())

mean_litter_kgkm = array_litter_dist.mean(axis=1)

print('95 perc. quantile: %f, %f' % (np.quantile(array_y_kg_pert.ravel(),.025),np.quantile(array_y_kg_pert.ravel(),.975)))

print(y_kg.sum())


def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

cmap = plt.cm.viridis
colors = get_colors(mean_litter_kgkm,cmap)

fig_,ax_ = plt.subplots(1)
cplot=ax_.scatter(np.random.random(len(colors)),np.random.random(len(colors)),c=mean_litter_kgkm)


fig = plt.figure(figsize=(5,3),dpi=120)  
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.add_feature(feature.NaturalEarthFeature('physical','land','10m'),facecolor='grey',zorder=1000)
for i1,(lons_,lats_) in enumerate(zip(lons_list,lats_list)):
    for lon_,lat_ in zip(lons_,lats_):
        ax.plot(lon_,lat_,color='r',transform=ccrs.PlateCarree(),linewidth=7) 
for i1,(lons_,lats_) in enumerate(zip(lons_list,lats_list)):
    for lon_,lat_ in zip(lons_,lats_):
        ax.plot(lon_,lat_,color=colors[i1],transform=ccrs.PlateCarree(),linewidth=6)
cbar=fig.colorbar(cplot)
cbar.set_label(r'Mean predicted litter [kg km$^{-1}$]')
# ax.set_extent((min(lons_list),max(lons_list),min(lats_list),max(lats_list)))
ax.set_xticks(np.linspace(3,6.5,8), crs=ccrs.PlateCarree())
ax.set_yticks(np.linspace(51,54,7), crs=ccrs.PlateCarree())
ax.set_ylabel('Latitude',fontsize=9)
ax.set_xlabel('Longitude',fontsize=9)


# #plot some daily changes in litter distribution
# for i2,date_ in enumerate(dates_unique[0:30]):
#     colors = get_colors(array_litter_dist[:,i2],cmap,vmin=mean_litter_kgkm.min(),vmax=mean_litter_kgkm.max())
#     fig = plt.figure(figsize=(5,3),dpi=120)  
#     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#     ax.add_feature(feature.NaturalEarthFeature('physical','land','10m'),facecolor='grey',zorder=1000)
#     for i1,(lons_,lats_) in enumerate(zip(lons_list,lats_list)):
#         for lon_,lat_ in zip(lons_,lats_):
#             ax.plot(lon_,lat_,color='r',transform=ccrs.PlateCarree(),linewidth=7) 
#     for i1,(lons_,lats_) in enumerate(zip(lons_list,lats_list)):
#         for lon_,lat_ in zip(lons_,lats_):
#             ax.plot(lon_,lat_,color=colors[i1],transform=ccrs.PlateCarree(),linewidth=6)
#     cbar=fig.colorbar(cplot)
#     cbar.set_label(r'Predicted litter [kg km$^{-1}$]')
#     # ax.set_extent((min(lons_list),max(lons_list),min(lats_list),max(lats_list)))
#     ax.set_xticks(np.linspace(3,6.5,8), crs=ccrs.PlateCarree())
#     ax.set_yticks(np.linspace(51,54,7), crs=ccrs.PlateCarree())
#     ax.set_ylabel('Latitude',fontsize=9)
#     ax.set_xlabel('Longitude',fontsize=9)    
#     ax.set_title(str(date_)[0:10])
#     fig.savefig('01_figures_matAnalysis/daily_litter/%3.3i.png' %i2)
#     plt.close(fig)

    
# fig,axes = plt.subplots(6,1,figsize=(13,13))
# years = [2014,2015,2016,2017,2018,2019]

# fig2,axes2 = plt.subplots(6,1,figsize=(13,13))

# fluxes_tot = np.array([])
# fluxes_positive = np.array([])
# for i1,year_ in enumerate(years):
#     mask = (dates_unique > np.datetime64('%4.4i-01-01'%year_) ) & (dates_unique < np.datetime64('%4.4i-12-31'%year_) )

#     axes[i1].plot(dates_unique[mask],array_y_kg_pert[:,mask].mean(axis=0))

#     pdy = np.quantile(array_y_kg_pert[:,mask],.95,axis=0) - array_y_kg_pert[:,mask].mean(axis=0)
#     mdy = array_y_kg_pert[:,mask].mean(axis=0) - np.quantile(array_y_kg_pert[:,mask],.05,axis=0)
    
#     axes[i1].errorbar(dates_unique[mask],array_y_kg_pert[:,mask].mean(axis=0),yerr=(mdy,pdy))

#     fluxes = (array_y_kg_pert[:,mask].mean(axis=0)[1:] - array_y_kg_pert[:,mask].mean(axis=0)[:-1]) / 365
    
#     axes2[i1].plot(dates_unique[mask][1:], fluxes,label='min.: %2.2f, max.: %2.2f' % (fluxes.min(),fluxes.max()))
#     axes2[i1].grid()
    
#     fluxes_positive = np.append(fluxes_positive, fluxes[fluxes>0])
    
#     print(fluxes.min(),fluxes.mean(),fluxes.max())
    
#     fluxes_tot = np.append(fluxes_tot,fluxes)
# axes[0].set_title('Total beached litter (90% C.I.)',fontsize=14)
# axes[-1].set_xlabel('Date',fontsize=14)
# axes[2].set_ylabel('Total amount of litter along the Dutch coastline',fontsize=14)

# axes2[0].set_title('Fluxes per day based on mean total litter')
# axes2[-1].set_xlabel('Date',fontsize=14)
# axes2[2].set_ylabel('Total modelled flux along the Dutch coastline [kg/km/day]',fontsize=14)

# plt.figure()
# plt.hist(fluxes_positive,20)
# plt.xlabel('Positive beaching flux [kg/km/day]')
# plt.ylabel('Frequency')
#%%
data_ospar = pd.read_excel('datafiles/BBCT_data_10062021.xlsx',sheet_name=4,header=4)

beach_IDs = ['NL001','NL002','NL003','NL004']

fig,ax = plt.subplots(1,figsize=(10,5))

for ID_ in beach_IDs:
    
    data_ = data_ospar[data_ospar['Survey beach'] == ID_]

    data_date = data_['Year and date']

    dates = np.array([pd.Timestamp(data_['Year and date'].values[i]) for i in range(len(data_date))])
    weights = data_['Weight per survey in kg '].values * 10

    ax.plot(dates,weights,'o-',label='OSPAR beach: %s'%ID_)

    dt = np.array([(dates[i] - dates[i-1]).days for i in range(1,len(dates))])
    fluxes_per_day = weights[1:] / dt

    print(weights)
    
    print('Mean net flux per day for %s: %f [kg/km]' %(ID_, fluxes_per_day.mean()))
    
    
ax.set_xlabel('Date',fontsize=14)
ax.set_ylabel('Total weight [kg/km]',fontsize=14)
ax.set_title('OSPAR data')
ax.legend()
