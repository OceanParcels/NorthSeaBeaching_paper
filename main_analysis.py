#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:21:35 2020
Main script used to run the machine learning analysis.
First coastline properties have to be calculated (see 'calculate_....py')
This script then:
    1) crunches the data from the BBCT cleanup tour (see folder datafiles)
    2) Calculates a big regression table with all features: for every measurement, 
    hydrodynamic conditions in the neighborhood are calculated
    3) The machine learning model is trained and tested using this regression table
    4) Model properties are calculated, and the variogram is plotted

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
from shapely.ops import split

from shapely.geometry import (box, LineString, MultiLineString, MultiPoint,
    Point, Polygon, MultiPolygon, shape)


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    
    
    i_which_coast = np.where( np.isclose(lon_,data_coastal_orientations['lon']) )[0]
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
        normal_vec = data_coastal_orientations['normal_vec_cartopy'][:,i_coastal_segment]    
        dot_prod = np.array([np.dot(vec_[i,:], normal_vec) for i in range(len(vec_))])
    else:
        print('calculating in product for single time')
        vec_u = data[variable_name[0]][i_date_closest].data[i_select]
        vec_v = data[variable_name[1]][i_date_closest].data[i_select]
        vec_ = np.array([vec_u[~np.isnan(vec_u)],vec_v[~np.isnan(vec_v)]]).T
        normal_vec = data_coastal_orientations['normal_vec_cartopy'][:,i_coastal_segment]    
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
    

def calculate_tide_derivative(lon,lat,date,data,dist_select,time_lag,variable_name,tmp_X,tmp_Y,use_land_mask=False,land_mask=None):
    """
    Calculate variable indicating in which part of the spring-neap cycle we are
    """
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
            i_select = (dist_mat == dist_mat_min)
        elif type(use_land_mask) == float:
            dist_mat_min = np.min(dist_mat[land_mask < use_land_mask])
            i_select = (dist_mat == dist_mat_min)
        else:
            i_select = (dist_mat == dist_mat.min())       
        return i_select,dist_mat_min
    
    if dist_select == 0: #closest point
        i_select,_ = closest_point(dist_mat,use_land_mask,land_mask)    
    else:
        raise RuntimeError('tide derivatives only defined for closest point')
    
    tide_max_start = data_tides[variable_name][i_date_start:i_date_start+24].values[:,i_select].max()
    tide_max_end = data_tides[variable_name][i_date_closest:i_date_closest+24].values[:,i_select].max()
    dtide_dt = (tide_max_end - tide_max_start) / time_lag
    
    return dtide_dt
    

def load_data_currents(year):
    file_july = '/Users/kaandorp/Data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/%4.4i07.nc'%year
    file_aug = '/Users/kaandorp/Data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/%4.4i08.nc'%year
    
    data_currents = xr.load_dataset(file_july)
    data_currents = xr.concat((data_currents,xr.load_dataset(file_aug)), 'time')
    
    return data_currents.squeeze()
    

def give_element_within_bounds(geom_split, box_analyze, tol=1.01):
    tol_lower = 1/tol
    tol_upper = tol
    c = 0
    for geom_ in geom_split:
        x_ = np.array(geom_.xy[0])
        y_ = np.array(geom_.xy[1])
        
        if (x_.min()>= tol_lower*np.array(box_analyze)[:,0].min()) and (y_.min()>= tol_lower*np.array(box_analyze)[:,1].min()) and (
                x_.max()<= tol_upper*np.array(box_analyze)[:,0].max()) and (y_.max()<= tol_upper*np.array(box_analyze)[:,1].max()):
            break
        c+=1
    return c


def find_normal_vector(lon_center,lat_center,ax,shpfilename,radius=5):
    # dkm = 5
    reader = shpreader.Reader(shpfilename)
    coastlines = reader.records()

    dlon = radius / (1.11e2 * np.cos(lat_center*(np.pi/180)))
    dlat = radius / 1.11e2
    
    box_analyze = [[lon_center-dlon, lat_center-dlat], [lon_center-dlon, lat_center+dlat], 
                   [lon_center+dlon, lat_center+dlat], [lon_center+dlon, lat_center-dlat], [lon_center-dlon, lat_center-dlat]]
    box_poly = shapely.geometry.Polygon(box_analyze)
    
    coastline_lons = np.array([])
    coastline_lats = np.array([])
    normal_vec = np.array([])
    
    any_intersect = False
    
    c = 0
    for coast_ in coastlines:
        check_ = False
        
        if radius > 10:
            check_ = True
        else:           
            if coast_.bounds[0] > -10 and coast_.bounds[1] > 30 and coast_.bounds[2] < 30 and coast_.bounds[3] < 60: #only use coastlines in the neighborhood
                check_ = True
        
        if check_:
            if coast_.geometry.intersects(box_poly):
                any_intersect = True
                print('intersect at %i' %c)
                                
                box_linestring = LineString(box_analyze)
                coastline_split = split(coast_.geometry,box_linestring)
    
                index_in_box = give_element_within_bounds(coastline_split,box_analyze,tol=1.00)
                coastline_in_box = coastline_split[index_in_box]
                
                coastline_lons = np.array(coastline_in_box.xy[0])
                coastline_lats = np.array(coastline_in_box.xy[1])
                
                x_lons = coastline_lons * np.cos(lat_center*(np.pi/180)) #convert to meters, i.e. compress longitude
                y_lats = coastline_lats
                
                x_ = x_lons - x_lons.mean()
                y_ = y_lats - y_lats.mean()
                svd_ = np.linalg.svd(np.array([x_,y_]))
                
                normal_vec = svd_[0][:,1]
                
                break
        c += 1
   
    if not any_intersect:
        print('no intersections found')
        normal_vec = np.array([0,0])
    
    if normal_vec[0] < 0: #all vectors point to the right, don't copy this for other domains than the Netherlands..
        normal_vec = -normal_vec
            
    ax.plot(*box_poly.exterior.xy,'k',transform=ccrs.PlateCarree())
    ax.plot(coastline_lons,coastline_lats,'ro-')
    
    scale_ = 0.001
    normal_vec_lon = np.array([lon_center,lon_center+scale_*(normal_vec[0]*(1.11e2 * np.cos(lat_center*(np.pi/180))))])
    normal_vec_lat = np.array([lat_center,lat_center+scale_*(normal_vec[1]*1.11e2)])
    ax.plot(normal_vec_lon,normal_vec_lat,'g-')
    
    
    return normal_vec


def normal_vector_2_points(lons,lats,ax):
    dx = (lons[1]-lons[0])*1.11e2 * np.cos(lats.mean()*(np.pi/180))
    dy = (lats[1]-lats[0])*1.11e2
    
    n1 = np.array([-dy, dx]) / np.sqrt(dx**2+dy**2)
    n2 = np.array([dy, -dx]) / np.sqrt(dx**2+dy**2)

    if n1[0] < 0: #all vectors point to the right (onto the land), don't copy this for other domains than the Netherlands..
        normal_vec = n2
    else:
        normal_vec = n1
    
    lon_center = lons.mean()
    lat_center = lats.mean()
    scale_ = 0.001
    normal_vec_lon = np.array([lon_center,lon_center+scale_*(normal_vec[0]*(1.11e2 * np.cos(lat_center*(np.pi/180))))])
    normal_vec_lat = np.array([lat_center,lat_center+scale_*(normal_vec[1]*1.11e2)])
    ax.plot(normal_vec_lon,normal_vec_lat,'c-')
    
    return normal_vec

#%%    get the locations, and corresponding beach orientations

data_BBCT = {}
startcols = [1, 7, 13, 19, 26, 38, 51]
for s in startcols[1:]:
    if s <= 13:
        usecols = [s+i for i in range(5)]
        colnames = ["Datum", "Etappes", "kms", "Gewicht", "Opmerkingen"]
    elif s == 19:
        usecols = [s + i for i in [0, 1, 2, 3, 5]]
        ["Datum", "Etappes", "kms", "Gewicht", "Opmerkingen"]
    elif s == 26: #2017
        usecols = [s+i for i in range(5)]
        colnames = ["Datum", "Etappes", "kms", "Gewicht", "Deelnemers"]
    elif s == 38: #2018
        usecols = [s + i for i in [0,1,3,4,5,7]]
        colnames = ["Datum", "Etappes", "kms", "Gewicht", "Deelnemers", "Opmerkingen"]
    elif s == 51:
        usecols = [s + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        colnames = ["Datum", "Etappes", "Startlocatie", "CoordinatenStart", "Eindlocatie", "CoordinatenEind", "Opmerkingen", "kms", "Gewicht", "Deelnemers"]
    else:
        raise RuntimeError('not implemented')
        
    df = pd.read_excel('datafiles/BBCT_data_10062021.xlsx', header=1, skiprows=[0, 1, 2, 3, 4], usecols=usecols)

    df.columns = colnames

    df.Datum = pd.to_datetime(df.Datum, errors='coerce')
    df = df.drop(df[np.isnat(df.Datum)].index)
    yr = df.iloc[0,0].year
    print(yr)
    data_BBCT[yr] = df
  
GPSlocs = {}
for yr in data_BBCT:
    df = data_BBCT[yr]
    if 'CoordinatenStart' in df.columns:
        print('Found coordinates data for %s' % yr)
        for i, e in enumerate(df.Etappes):
            if '.' in e:
                e = e.split('. ')[1]
            e = e.replace('–', '-')
            CoordStart = df.CoordinatenStart[i]
            lat1, lon1 = df.CoordinatenStart[i].split(',')
            if ',' in df.CoordinatenEind[i]:
                lat2, lon2 = df.CoordinatenEind[i].split(',')
            else:
                lat2, lon2 = lat1, lon1
            lat = (float(lat1) + float(lat2)) /2
            lon = (float(lon1) + float(lon2)) /2
            GPSlocs[e] = (lat, lon)

# adding additional coordinates not in xls file
GPSlocs['Vrouwenpolder'] = (51.59, 3.57)
GPSlocs['Camperduin'] = (52.73, 4.64)
GPSlocs['Callantsoog'] = (52.84, 4.69)
GPSlocs['Julianadorp'] = (52.90, 4.71)
GPSlocs['Katwijk'] = (52.20, 4.39)

 
lon_unique = np.array([])
lat_unique = np.array([])



for i1,yr in enumerate(data_BBCT):
    df = data_BBCT[yr]
    lat = np.nan*np.zeros(len(df.Etappes))
    lon = np.nan*np.zeros(len(df.Etappes))
    exactlocmatch = np.zeros(len(df.Etappes), dtype=bool)

    for i, e in enumerate(df.Etappes):
        # split off stage number
        e = e.split('. ')[-1]
        # some replacing
        e = e.replace('–', '-')
        e = e.replace(' naar ', '-')
        if e in GPSlocs:
            lat[i] = GPSlocs[e][0]
            lon[i] = GPSlocs[e][1]
            exactlocmatch[i] = True
        else:
            # separate start and end
            e = e.split('-')
            # remove province
            for j in range(len(e)):
                e[j] = e[j].split('(')[0]
                e[j] = e[j].replace(')', '')
                # remove colons:
                e[j] = e[j].split(':')[0]
                # remove numbers:
                for n in ['1', '2']:
                    e[j] = e[j].replace(n, '')
            for v in GPSlocs.keys():
                if e[0].strip() in v or (len(e) > 1 and e[1].strip() in v):
                    lat[i] = GPSlocs[v][0]
                    lon[i] = GPSlocs[v][1]
                    exactlocmatch[i] = False
        if np.isnan(lon[i]):
            print('Not found:', yr, e)
                    
                        
    df['lat'] = lat
    df['lon'] = lon
    df['exactlocmatch'] = exactlocmatch
    
    for lon_,lat_ in zip(df['lon'],df['lat']):
        if ~np.isin(lon_,lon_unique) and ~np.isin(lat_,lat_unique):
            lon_unique=np.append(lon_unique,lon_)
            lat_unique=np.append(lat_unique,lat_)
    
i_sort = np.argsort(lon_unique)
lon_unique = lon_unique[i_sort]
lat_unique = lat_unique[i_sort]    
    


#%% import the calculated coastline properties (see calculate_coast_mesh_properties.py and calculate_coastal_orientations.py)

data_coastal_length = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/datafiles/netcdf_coastal_lengths.nc')
X_coast, Y_coast = np.meshgrid(data_coastal_length['lon'],data_coastal_length['lat'])
currents_landmask = np.loadtxt('datafiles/datafile_trueLandMask_297x_375y').astype(bool)

def plot_normal_vec(normal_vec,lon_,lat_,ax,scale=0.005,style='k-'):
    normal_vec_lon = np.array([lon_,lon_+scale*(normal_vec[0]*(1.11e2 * np.cos(lat_*(np.pi/180))))])
    normal_vec_lat = np.array([lat_,lat_+scale*(normal_vec[1]*1.11e2)])
    ax.plot(normal_vec_lon,normal_vec_lat,style,zorder=1000)    


fig = plt.figure(figsize=(10,5),dpi=120)     
ax = plt.axes(projection=ccrs.PlateCarree())    
ax.pcolormesh(X_coast,Y_coast,currents_landmask,alpha=.6)
ax.plot(lon_unique,lat_unique,'o',transform=ccrs.PlateCarree())
ax.set_extent((3.2,6.8,51,53.7))
ax.coastlines(resolution='10m')


data_coastal_orientations = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/datafiles/netcdf_coastal_orientations.nc')

array_dot_mesh_coast = []
for i1,yr in enumerate(data_BBCT):
    df = data_BBCT[yr]
    for lon_,lat_ in zip(df['lon'],df['lat']):
        i_which_coast = np.where( np.isclose(lon_,data_coastal_orientations['lon']) )[0]
        if i_which_coast.size == 1:
            i_coastal_segment = i_which_coast[0]
        else:
            raise RuntimeError('coastal segment not found, or multiple segments correspond to location')
        
        n_mesh = data_coastal_orientations['normal_vec_mesh'][:,i_coastal_segment]
        n_cartopy = data_coastal_orientations['normal_vec_cartopy'][:,i_coastal_segment]
        
        plot_normal_vec(n_mesh,lon_,lat_,ax,style='k-')
        plot_normal_vec(n_cartopy,lon_,lat_,ax,style='r-')
        
        
        array_dot_mesh_coast.append(np.dot(n_mesh,n_cartopy))
array_dot_mesh_coast = np.array(array_dot_mesh_coast)    


#%%    Calculate the regression table which is used to select the machine learning features from

data_land_sea = xr.open_dataset('/Users/kaandorp/Data/ERA5/Wind_NorthSea_old/land_sea_mask.nc')
mask_land_sea = data_land_sea['lsm'].data[0,:,:]

data_beaching_f = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/histograms/beaching_hist_0405_f.nc')
data_beaching_p = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/histograms/beaching_hist_0405_p.nc')
data_beaching_r = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/histograms/beaching_hist_0405_r.nc')

X_beaching,Y_beaching = np.meshgrid(data_beaching_f['lon'].data,data_beaching_f['lat'].data)

data_popden = xr.open_dataset('/Users/kaandorp/Git_repositories/NorthSeaBeaching/datafiles/netcdf_popdensity.nc')
X_pop, Y_pop = np.meshgrid(data_popden['lon'],data_popden['lat'])
land_mask_pop = ~np.isnan(data_popden['popdensity'][0,:,:].data)



#-----------tides------------
tides_folder = '/Users/kaandorp/Data/FES2014Data/ocean_tide/'
file_K1 = os.path.join(tides_folder,'conv_k1.nc')
file_M2 = os.path.join(tides_folder,'conv_m2.nc')
file_O1 = os.path.join(tides_folder,'conv_o1.nc')
file_S2 = os.path.join(tides_folder,'conv_s2.nc')

data_K1 = xr.open_dataset(file_K1)
data_M2 = xr.open_dataset(file_M2)
data_O1 = xr.open_dataset(file_O1)
data_S2 = xr.open_dataset(file_S2)


def initialize_tides(year):
    print('Calculating tides for %i' % year)
    lons = data_K1['lon']
    lats = data_K1['lat']
    
    
    i_lon_min = np.where(lons >= 2 )[0][0]
    i_lon_max = np.where(lons <= 7 )[0][-1]
    i_lat_min = np.where(lats >= 50 )[0][0]
    i_lat_max = np.where(lats <= 54 )[0][-1]

    X_tides,Y_tides = np.meshgrid(lons[i_lon_min:i_lon_max],lats[i_lat_min:i_lat_max])

    day_start = datetime(year,7,1,00)
    day_end = datetime(year,9,1,00)

    time_array = pd.date_range(day_start,day_end,freq='H')
    
    t0 = datetime(1900,1,1,0,0) # origin of time = 1 January 1900, 00:00:00 UTC
    t0rel =  (day_start - t0).total_seconds() # number of seconds elapsed between t0 and starttime
    deg2rad = math.pi/180.0 # factor to convert degrees to radians
    
    omega_M2 = (28.9841042 * deg2rad) / 3600.0 # angular frequency of M2 in radians per second
    omega_S2 = (30.0000000 * deg2rad) / 3600.0 # angular frequency of S2 in radians per second
    omega_K1 = (15.0410686 * deg2rad) / 3600.0 # angular frequency of K1 in radians per second
    omega_O1 = (13.9430356 * deg2rad) / 3600.0 # angular frequency of O1 in radians per second
    
    # Define constants to compute astronomical variables T, h, s, N (all in degrees) (source: FES2014 code)
    cT0 = 180.0
    ch0 = 280.1895
    cs0 = 277.0248
    cN0 = 259.1568; cN1 = -1934.1420
    deg2rad = math.pi/180.0
    
    # Calculation of factors T, h, s at t0 (source: Doodson (1921))
    T0 = math.fmod(cT0, 360.0) * deg2rad
    h0 = math.fmod(ch0, 360.0) * deg2rad
    s0 = math.fmod(cs0, 360.0) * deg2rad
    
    # Calculation of V(t0) (source: Schureman (1958))
    V_M2 = 2*T0 + 2*h0 - 2*s0
    V_S2 = 2*T0
    V_K1 = T0 + h0 - 0.5*math.pi
    V_O1 = T0 + h0 - 2*s0 + 0.5*math.pi

    tide_land_mask = np.isnan(data_K1['phase'])[i_lat_min:i_lat_max,i_lon_min:i_lon_max]
    tide_array = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    
    for i1,date in enumerate(time_array):
        
        time = (date - day_start).total_seconds()
        # time = (date - t0).total_seconds() #dont use
        
        t = ((time + t0rel)/86400.0)/36525.0
        
        # Calculation of factors N, I, nu, xi at time (source: Schureman (1958))
        # Since these factors change only very slowly over time, we take them as constant over the time step dt
        N = math.fmod(cN0 + cN1*t, 360.0) * deg2rad
        I = math.acos(0.91370 - 0.03569*math.cos(N))
        tanN = math.tan(0.5*N)
        at1 = math.atan(1.01883 * tanN)
        at2 = math.atan(0.64412 * tanN)
        nu = at1 - at2
        xi = -at1 - at2 + N
        nuprim = math.atan(math.sin(2*I) * math.sin(nu)/(math.sin(2*I)*math.cos(nu) + 0.3347))
        
        # Calculation of u, f at current time (source: Schureman (1958))
        u_M2 = 2*xi - 2*nu
        f_M2 = (math.cos(0.5*I))**4/0.9154
        u_S2 = 0
        f_S2 = 1
        u_K1 = -nuprim
        f_K1 = math.sqrt(0.8965*(math.sin(2*I))**2 + 0.6001*math.sin(2*I)*math.cos(nu) + 0.1006)
        u_O1 = 2*xi - nu
        f_O1 = math.sin(I)*(math.cos(0.5*I))**2/0.3800
        
        
        ampl_K1 = f_K1 * data_K1['amplitude'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_K1 = V_K1 + u_K1 - data_K1['phase'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_K1 = ampl_K1 * np.cos(omega_K1 * (time+t0rel) + pha_K1)
        
        ampl_M2 = f_M2 * data_M2['amplitude'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M2 = V_M2 + u_M2 - data_M2['phase'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M2 = ampl_M2 * np.cos(omega_M2 * (time+t0rel) + pha_M2)
        
        ampl_O1 = f_O1 * data_O1['amplitude'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_O1 = V_O1 + u_O1 - data_O1['phase'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_O1 = ampl_O1 * np.cos(omega_O1 * (time+t0rel) + pha_O1)
        
        ampl_S2 = f_S2 * data_S2['amplitude'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_S2 = V_S2 + u_S2 - data_S2['phase'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_S2 = ampl_S2 * np.cos(omega_S2 * (time+t0rel) + pha_S2)
        
        tide_tot = tide_K1 + tide_M2 + tide_O1 + tide_S2
        tide_array[i1,:,:] = tide_tot
        
    ds = xr.Dataset(
        {"tide": (("time", "lat", "lon"), tide_array ),
         "explanation": 'tides calculated from FES dataset'},
        coords={
            "lon": lons[i_lon_min:i_lon_max],
            "lat": lats[i_lat_min:i_lat_max],
            "time": time_array,
        },
    )   
   
    return ds, X_tides, Y_tides, tide_land_mask


distances_select = [0,20,50,100]
times_lag = [1, 3, 9, 30]
#variables with radii and lead times
vars_names = ['VHM0_mean','VHM0_max','mag_Stokes_mean','mag_Stokes_max','mag_wind_mean',
                  'mag_wind_max','in_Stokes_mean','in_Stokes_max','in_Stokes_min','in_wind_mean','in_wind_max','in_wind_min',
                  'in_currents_mean','in_currents_max','in_currents_min',
                  'beaching_f_tau25','beaching_p_tau25','beaching_r_tau25','beaching_f_tau75','beaching_p_tau75','beaching_r_tau75',
                  'beaching_f_tau150','beaching_p_tau150','beaching_r_tau150']

#variables with lead times only
vars_names2 = ['temp_mean','temp_max','tide_max','tide_std','tide_derivative']

#variables with radii only
distances_select2 = [0,20,50,100]
vars_names3 = ['pop_density','coastal_length']

#'instantaneous' variables (conditions during the tour of 6 hours)
vars_names4 = ['tide_tour_max','tide_tour_min']


vars_calculate = []
for dist_ in distances_select:
    for time_ in times_lag:
        for var_ in vars_names:
            vars_calculate.append('%s_%3.3i_%3.3i' % (var_,dist_,time_))

for time_ in times_lag:
    for var_ in vars_names2:
        vars_calculate.append('%s_%3.3i' % (var_,time_))

for dist_ in distances_select2:
    for var_ in vars_names3:
        vars_calculate.append('%s_%3.3i' % (var_,dist_))

vars_calculate.extend(vars_names4)


print('Calculating regression parameters...')
regression_table = pd.DataFrame(columns=['lon','lat','time','kg/m']+vars_calculate+['participants'])
    
c_table = 0    
for i1,yr in enumerate(data_BBCT):
    df = data_BBCT[yr]

    data_waves = xr.load_dataset('/Users/kaandorp/Data/CMEMS/Waves_NorthSea/GLOBAL_REANALYSIS_WAV_001_032_%4i0101.nc' % yr)
    data_wind = xr.load_dataset('/Users/kaandorp/Data/ERA5/Wind_NorthSea_old/wind_%4i_065--020-040-013.nc' % yr)
    data_temp = xr.load_dataset('/Users/kaandorp/Data/ERA5/Temp_NorthSea/temp_%4i_054-003-051-006.nc' % yr)
    data_currents = load_data_currents(yr)
    
    X_waves,Y_waves = np.meshgrid(data_waves['longitude'].data,data_waves['latitude'].data)
    waves_land_mask = np.all( np.isnan( data_waves['VHM0'][:,:,:]), axis=0).data
    X_wind,Y_wind = np.meshgrid(data_wind['longitude'].data,data_wind['latitude'].data)
    X_temp,Y_temp = np.meshgrid(data_temp['longitude'].data,data_temp['latitude'].data)
    X_curr,Y_curr = np.meshgrid(data_currents['longitude'].data,data_currents['latitude'].data)
    
    data_tides,X_tides,Y_tides,tides_land_mask = initialize_tides(yr)
    
    # dist_select = 50 #km
    # time_lag = 1 #days
    for i2, (date_,lon_,lat_,kg_,km_) in enumerate(zip(df['Datum'],df['lon'],df['lat'],df['Gewicht'],df['kms'])):

        if 'Deelnemers' in df.keys():
            participants_ = df['Deelnemers'].iloc[i2]
        else:
            participants_ = np.nan
        
        if date_.hour == 0:
            date_ = date_ + timedelta(hours=11) #assume noon
        calculated_variables = [lon_,lat_,date_,kg_/km_]

        # variables with radii and lead times
        for dist_select in distances_select:
            for time_lag in times_lag:
                
                # calculate variables: (np.nanmean(x),np.nanmax(x),np.nanmin(x),np.nanstd(x))
                VHM0_mean, VHM0_max, _, _ = calculate_variables(lon_,lat_,date_,data_waves,
                                                             dist_select,time_lag,'VHM0',X_waves,Y_waves,quantity='all',
                                                             use_land_mask=True,land_mask=waves_land_mask)

                mag_Stokes_mean, mag_Stokes_max, _, _ = calculate_variables(lon_,lat_,date_,data_waves,
                                                                         dist_select,time_lag,['VSDX','VSDY'],X_waves,Y_waves,quantity='all',
                                                                         use_land_mask=True,land_mask=waves_land_mask)
                
                mag_wind_mean, mag_wind_max, _, _ = calculate_variables(lon_,lat_,date_,data_wind,
                                                                     dist_select,time_lag,['u10','v10'],X_wind,Y_wind,quantity='all',use_land_mask=.5,land_mask=mask_land_sea)
           
                mag_currents_mean, mag_currents_max, _, _ = calculate_variables(lon_,lat_,date_,data_currents,
                                                                     dist_select,time_lag,['uo','vo'],X_curr,Y_curr,quantity='all',
                                                                     use_land_mask=True,land_mask=currents_landmask)

                in_Stokes_mean, in_Stokes_max, in_Stokes_min = calculate_inproduct(lon_,lat_,date_,data_waves,
                                                                                   dist_select,time_lag,['VSDX','VSDY'],X_waves,Y_waves,quantity='all',
                                                                                   use_land_mask=True,land_mask=waves_land_mask)
                
                in_wind_mean, in_wind_max, in_wind_min = calculate_inproduct(lon_,lat_,date_,data_wind,
                                                                             dist_select,time_lag,['u10','v10'],X_wind,Y_wind,quantity='all',use_land_mask=.5,land_mask=mask_land_sea)
                
                in_currents_mean, in_currents_max, in_currents_min = calculate_inproduct(lon_,lat_,date_,data_currents,
                                                                             dist_select,time_lag,['uo','vo'],X_curr,Y_curr,quantity='all',use_land_mask=True,land_mask=currents_landmask)
                
                
                beaching_f_tau25 = calculate_variables(lon_,lat_,date_,data_beaching_f,
                                                             dist_select,time_lag,'beaching_tau25',X_beaching,Y_beaching,quantity='sum')
                beaching_p_tau25 = calculate_variables(lon_,lat_,date_,data_beaching_p,
                                                             dist_select,time_lag,'beaching_tau25',X_beaching,Y_beaching,quantity='sum')
                beaching_r_tau25 = calculate_variables(lon_,lat_,date_,data_beaching_r,
                                                             dist_select,time_lag,'beaching_tau25',X_beaching,Y_beaching,quantity='sum')
                beaching_f_tau75 = calculate_variables(lon_,lat_,date_,data_beaching_f,
                                                             dist_select,time_lag,'beaching_tau75',X_beaching,Y_beaching,quantity='sum')
                beaching_p_tau75 = calculate_variables(lon_,lat_,date_,data_beaching_p,
                                                             dist_select,time_lag,'beaching_tau75',X_beaching,Y_beaching,quantity='sum')
                beaching_r_tau75 = calculate_variables(lon_,lat_,date_,data_beaching_r,
                                                             dist_select,time_lag,'beaching_tau75',X_beaching,Y_beaching,quantity='sum')
                beaching_f_tau150 = calculate_variables(lon_,lat_,date_,data_beaching_f,
                                                             dist_select,time_lag,'beaching_tau150',X_beaching,Y_beaching,quantity='sum')
                beaching_p_tau150 = calculate_variables(lon_,lat_,date_,data_beaching_p,
                                                             dist_select,time_lag,'beaching_tau150',X_beaching,Y_beaching,quantity='sum')
                beaching_r_tau150 = calculate_variables(lon_,lat_,date_,data_beaching_r,
                                                             dist_select,time_lag,'beaching_tau150',X_beaching,Y_beaching,quantity='sum')
                
                print(date_,in_wind_mean,in_wind_min,in_Stokes_mean,in_Stokes_min,dist_select,time_lag)
                
                calculated_variables.extend([VHM0_mean,VHM0_max,mag_Stokes_mean,mag_Stokes_max,mag_wind_mean,
                  mag_wind_max,in_Stokes_mean,in_Stokes_max,in_Stokes_min,in_wind_mean,in_wind_max,in_wind_min,
                  in_currents_mean,in_currents_max,in_currents_min,
                  beaching_f_tau25,beaching_p_tau25,beaching_r_tau25,beaching_f_tau75,beaching_p_tau75,beaching_r_tau75,
                  beaching_f_tau150,beaching_p_tau150,beaching_r_tau150])

        # variables with lead times only (temp and tides)
        print('Calculating temp and tides')
        for time_lag in times_lag:
            temp_mean, temp_max, _, _ = calculate_variables(lon_, lat_, date_, data_temp, 0, time_lag, 't2m', X_temp, Y_temp, quantity='all')
            
            _, tide_max, _, tide_std = calculate_variables(lon_, lat_, date_, data_tides, 0, time_lag, 'tide', X_tides, Y_tides, quantity='all',use_land_mask=True,land_mask=tides_land_mask)
            
            tide_derivative = calculate_tide_derivative(lon_, lat_, date_, data_tides, 0, time_lag, 'tide', X_tides, Y_tides, use_land_mask=True, land_mask=tides_land_mask)
            
            calculated_variables.extend([temp_mean,temp_max,tide_max,tide_std,tide_derivative])

            
        # variables with radii only (MPW density, coastal lengths)
        print('Calculating population density, coastal lengths')
        for dist_select in distances_select2:
            pop_density = calculate_variables(lon_, lat_, date_, data_popden, dist_select, 0, 'popdensity', X_pop, Y_pop, quantity='mean',use_land_mask=True,land_mask=~land_mask_pop)
            
            coastal_length = calculate_variables(lon_, lat_, date_, data_coastal_length, dist_select, 0, 'coastline_length', X_coast, Y_coast, quantity='sum',use_land_mask=False)

            calculated_variables.extend([pop_density,coastal_length])
            
            
        print('Calculating tides along tour')
        # 'instantaneous' variables (tide during tour): nearest location, lead time of 0.25 day (6 hours)
        _, tide_tour_max, tide_tour_min, _ = calculate_variables(lon_, lat_, date_+timedelta(hours=6), data_tides, 0, 0.25, 'tide', X_tides, Y_tides, quantity='all',use_land_mask=True,land_mask=tides_land_mask)
        calculated_variables.extend([tide_tour_max,tide_tour_min])
            
        calculated_variables.extend([participants_])
        
        regression_table.loc[c_table] = calculated_variables
        c_table += 1

    print('Year %i done' % yr)

regression_table['dot_mesh_coast'] = array_dot_mesh_coast

filename_rt = 'regression_table_%3.3i_%3.3i_%4.4i%2.2i%2.2i.pickle' % (regression_table.shape[0],regression_table.shape[1],datetime.today().year,datetime.today().month,datetime.today().day)
outfile = open(os.path.join('./pickle_files/',filename_rt),'wb')
pickle.dump(regression_table,outfile)
outfile.close()    


#%% Part 2: the machine learning analysis, where random forests are trained on the regression table.
# features are picked from the feature clusters using k-fold analysis

def find_features_containing(list_features,string_):
    return np.array([True if string_ in feature_ else False for feature_ in list_features])

def find_features_not_containing(list_features,string_):
    return np.array([False if string_ in feature_ else True for feature_ in list_features])


from sklearn.model_selection import KFold
from sklearn import linear_model
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance
from collections import defaultdict


def cluster_select_features_2(x,threshold=2.89,include_model_feat=True,separate_model_weather=False,exclude=np.array([2,3])):

    if separate_model_weather:
        i_nonmodel_features_ = np.where(find_features_not_containing(x.keys(),'beaching'))[0]
    else:
        i_nonmodel_features_ = np.where(find_features_not_containing(x.keys(),'xxxxxx'))[0]
    
    i_nonmodel_features_ = np.setdiff1d(i_nonmodel_features_,exclude)

    fig1,ax1 = plt.subplots(1,figsize=(20,15))
    fig2,ax2 = plt.subplots(1,figsize=(20,15))
    
    corr = spearmanr(x.iloc[:,i_nonmodel_features_]).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=list(x.iloc[:,i_nonmodel_features_].keys()), ax=ax1, leaf_rotation=90,  leaf_font_size=8,
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))
    
    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical',fontsize=8)
    ax2.set_yticklabels(dendro['ivl'],fontsize=6)
    
    fig1.tight_layout()
    fig2.tight_layout()
    
    
    cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = x.iloc[:,i_nonmodel_features_].keys()[selected_features]
    
    clusters_out = []
    for cluster_ in cluster_id_to_feature_ids.values():
        clusters_out.append(list(i_nonmodel_features_[cluster_]))
    
    if separate_model_weather:
        if include_model_feat:
            clusters_out.append(list(np.where(find_features_containing(x.keys(),'beaching_f'))[0]))
            clusters_out.append(list(np.where(find_features_containing(x.keys(),'beaching_p'))[0]))
            clusters_out.append(list(np.where(find_features_containing(x.keys(),'beaching_r'))[0]))
            
    return np.array(selected_features),selected_features_names,clusters_out #cluster_id_to_feature_ids.values()
    

def normalize_beaching_variables(regression_table_):
    '''
    Normalize the beaching fluxes to the length of the coastline (-> concentrations)
    '''
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


def calculate_most_common_feature(best_feature_in_cluster):
    """
    Look in each cluster which feature was picked to most often
    """
    best_feature = []
    for i1 in range(best_feature_in_cluster.shape[1]):
        a,b, = np.unique(best_feature_in_cluster[:,i1],return_counts=True)
        index_feat = a[np.argmax(b)]
        best_feature.append(index_feat)
    return np.array(best_feature,dtype=int)


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
    """
    For some years the participant numbers are missing, these are imputed here
    """
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
            


filename = 'pickle_files/regression_table_180_420_20210610.pickle'

with open(filename, 'rb') as f:
    regression_table = pickle.load(f)

regression_table_ = regression_table.dropna(axis=0,how='any',subset=['kg/m']).copy()
regression_table_ = normalize_beaching_variables(regression_table_)

impute_participant_numbers(regression_table_)

corrmat = regression_table_.corr()
corr_waste = corrmat.iloc[2,:]
i_sort = np.argsort(corr_waste)

regressor = 'RFR' #RFR, LR, GLM
use_scaling = True
use_PCA = False

if regressor == 'RFR':
    reg1 = RandomForestRegressor(oob_score=True,max_features=.33)
elif regressor == 'LR':
    reg1 = linear_model.LinearRegression()
elif regressor == 'GLM':
    # reg1 = linear_model.GammaRegressor(alpha=1.,fit_intercept=True)
    reg1 = linear_model.TweedieRegressor(power=2,link='log',alpha=0.,fit_intercept=True)   
elif regressor == 'GPR':
    reg1 = GaussianProcessRegressor()    

# variables to exclude as features, such as lon/lat, tidal derivatives for longer timescales
exclude_strings = ['lon','lat','time','kg/m','pop_density','temp','_020_','derivative_009','derivative_030']

exclude = np.array([],dtype=int)
for i1,feat_ in enumerate(regression_table_.keys()):
    for s_ in exclude_strings:
        if s_ in feat_:
            exclude = np.append(exclude,i1)
    
_,_,feature_clusters = cluster_select_features_2(regression_table_,threshold=2.27,include_model_feat=True,separate_model_weather=True,exclude=exclude ) #2.89

n_vars = 0
for cluster_ in feature_clusters:
    n_vars += len(cluster_)

feature_numbers = np.concatenate([np.array(xi) for xi in feature_clusters])

x = regression_table_.iloc[:,feature_numbers]
y = np.log10(regression_table_.iloc[:,3])
n_repeat = 10

feature_importance_score = np.zeros(len(feature_clusters))
best_feature_in_cluster = np.zeros(len(feature_clusters))
feature_importance_score_mat = np.zeros([n_repeat*5,len(feature_clusters)])
best_feature_in_cluster_mat = np.zeros([n_repeat*5,len(feature_clusters)])

y_test_tot = np.array([])
y_pred_tot = np.array([])

c = 0
array_pearsonR = []

for i_outer in range(n_repeat):
    
    print('outer loop %i/%i' % (i_outer+1,n_repeat))
    
    kf = KFold(n_splits=5,shuffle=True)
    kf2 = KFold(n_splits=5,shuffle=True)
    
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
        
        
        #----------------select from the feature clusters the best feature----------
        r_validation = np.zeros(x.shape[1])
        for i2, (i_train_train, i_train_val) in enumerate(kf2.split(x_train)):    
            x_train_train = x_train[i_train_train]
            x_train_val = x_train[i_train_val]
            y_train_train = y_train.values[i_train_train]
            y_train_val = y_train.values[i_train_val]
    
            model_val = reg1.fit(x_train_train,y_train_train)
            if regressor == 'RFR':
                r_validation += model_val.feature_importances_
            elif regressor == 'GLM':
                r_validation += permutation_importance(model_val, x_train_val, y_train_val,n_repeats=10,random_state=0).importances_mean
            elif regressor == 'GPR':
                r_validation += permutation_importance(model_val, x_train_val, y_train_val,n_repeats=10,random_state=0).importances_mean
        
        selected_features = np.array([],dtype=int)
        for cluster in feature_clusters:
            if len(cluster) > 1:
                i_x = np.concatenate([np.where(feature_numbers == val)[0] for val in cluster])
                r_cluster = r_validation[i_x]
                i_cluster_best = np.argsort(r_cluster)[-1]
                i_select = i_x[i_cluster_best]
            else:
                i_select = np.where(feature_numbers == cluster[0])[0]
                                    
            selected_features = np.append(selected_features, int(i_select))
        
        if i1 == 0:
            best_feature_in_cluster = selected_features.copy()
        else:
            best_feature_in_cluster = np.vstack((best_feature_in_cluster,selected_features))
        
        
        x_train = x_train[:,selected_features]
        x_test = x_test[:,selected_features]
        
        
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
        best_feature_in_cluster_mat[c,:] = selected_features
        
        if i_outer == n_repeat-1:
            plt.plot(y_test,y_pred_test,'o',label='test fold %i, R: %f' %(i1,pearsonR))
        
        print(i1)
        c+=1
    

    selected_features_total = calculate_most_common_feature(best_feature_in_cluster)
    
    i_sort_feat_imp = np.argsort(np.abs(feature_importance_score))
    if not use_PCA:
        print('Most important features: ') 
        features_imp_to_least = x.keys()[selected_features_total][i_sort_feat_imp][-10:][::-1]
        for i2,feat_ in enumerate(features_imp_to_least):
            print('%i. %s' % (i2+1,feat_))

    
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


#%% analyze the effect of the features

indices = i_sort_feat_imp[::-1][0:12]
n_col = int(3)
n_row = int(np.ceil(len(indices)/n_col))
fig,ax = plt.subplots(n_row,n_col,figsize=(12,15))

for i2,index_ in enumerate(indices):

    feature_ = x.keys()[selected_features_total][index_]
    feature_syn_vals = np.linspace(x[feature_].min(),x[feature_].max(),30)
    
    x_table_syn = pd.DataFrame(columns=x.columns)
    
    for i1,feature_val_ in enumerate(feature_syn_vals):
     
        x_table_syn.loc[i1] = x.iloc[:,selected_features_total].mean()
        x_table_syn.loc[i1,feature_] = feature_val_
    
    
    if use_scaling:
        x_syn = scaler.transform(x_table_syn)
    
    x_syn = x_syn[:,selected_features_total]
    
    predictions = reg1.predict(x_syn)
    
    # fig,ax = plt.subplots(1)
    i_row = int(i2 // n_col)
    i_col = int(i2 % n_col)
    ax[i_row,i_col].plot(feature_syn_vals,predictions)
    ax[i_row,i_col].set_xlabel(feature_)
    ax[i_row,i_col].set_ylabel('Prediction [log10(kg/km)]')
    ax[i_row,i_col].set_title('Importance ranking: %i' % (i2+1))
fig.subplots_adjust(hspace=.4,wspace=.4)


#%% Final model properties

selected_features_total = calculate_most_common_feature(best_feature_in_cluster_mat)
i_sort_feat_imp_total = np.argsort(np.abs(feature_importance_score_mat.mean(axis=0)))

print('Total: most important features: ', x.keys()[selected_features_total][i_sort_feat_imp_total][-10:])
features_imp_to_least_total = x.keys()[selected_features_total][i_sort_feat_imp_total][-10:][::-1]
for i2,feat_ in enumerate(features_imp_to_least_total):
    print('%i. %s' % (i2+1,feat_))

scaler = StandardScaler()
scaler.fit(x)

final_model = reg1.fit(scaler.transform(x)[:,selected_features_total],y)
SAVE = False
LOAD = False
if SAVE:
    from joblib import dump, load
    now_ = datetime.now()
    filename_model = 'pickle_files/%s_%4.4i%2.2i%2.2i%2.2i.joblib' % (regressor,now_.year,now_.month,now_.day,now_.hour)
    dump(final_model, filename_model) 
if LOAD:
    final_model = load('pickle_files/RFR_2021050617.joblib') 


indices = i_sort_feat_imp_total[::-1][0:12]
n_col = int(3)
n_row = int(np.ceil(len(indices)/n_col))
fig,ax = plt.subplots(n_row,n_col,figsize=(12,15))

for i2,index_ in enumerate(indices):
    feature_ = x.keys()[selected_features_total][index_]
    feature_syn_vals = np.linspace(x[feature_].min(),x[feature_].max(),30)
    
    x_table_syn = pd.DataFrame(columns=x.columns)
    
    for i1,feature_val_ in enumerate(feature_syn_vals):
     
        x_table_syn.loc[i1] = x.iloc[:,selected_features_total].mean()
        x_table_syn.loc[i1,feature_] = feature_val_
    
    
    if use_scaling:
        x_syn = scaler.transform(x_table_syn)
    
    x_syn = x_syn[:,selected_features_total]
    
    predictions = final_model.predict(x_syn)
    
    i_row = int(i2 // n_col)
    i_col = int(i2 % n_col)
    ax[i_row,i_col].plot(feature_syn_vals,predictions)
    ax[i_row,i_col].set_xlabel(feature_)
    ax[i_row,i_col].set_ylabel('Prediction [log10(kg/km)]')
    ax[i_row,i_col].set_title('Importance ranking: %i' % (i2+1))
fig.subplots_adjust(hspace=.4,wspace=.4)    
    

for cluster in feature_clusters:
    print('variables:')
    print(regression_table_.keys()[cluster])
          
cluster_names = x.keys()[selected_features_total]

def find_cluster(index):     
    for cluster_,name_ in zip(feature_clusters,cluster_names):
        if index in cluster_:
            break
    return name_
        
fig,ax = plt.subplots(1,figsize=(12,12))
ax.boxplot(feature_importance_score_mat[:,i_sort_feat_imp_total], vert=False, labels=np.array(cluster_names)[i_sort_feat_imp_total])
ax.set_xlabel('Gini importance')
fig.tight_layout()

cluster_names_top10 = np.array([r'$h_{tide}$, max./std. lead time = 9,30d.',
                       r'$h_{tide}$, max./std. lead time = 1,3d.',
                       r'$l_{coast}$, r = 50,100km',
                       r'$\mathbf{n}_{grid} \cdot \mathbf{n}$',
                       r'$h_{tide}$, max. during tour',
                       r'$F_{beach.,riv.}$',
                       r'$\mathbf{U_{curr.} \cdot n}$, mean/min., r = 0km, lead time = 30d.',
                       r'$F_{beach.,fis.}$',
                       r'$F_{beach.,pop.}$',
                       r'$\mathbf{U_{curr.} \cdot n}$, max., r = 50,100km, lead time = 1-9d.'])
fig,ax = plt.subplots(1,figsize=(10,6))
ax.boxplot(feature_importance_score_mat[:,i_sort_feat_imp_total][:,-10:], vert=False)
ax.set_yticklabels(cluster_names_top10[::-1],fontsize=13)
ax.set_xlabel('Gini importance',fontsize=13)
fig.tight_layout()


#%% Calculate the variogram with errorbars

from scipy.spatial.distance import squareform, pdist

filename = 'pickle_files/regression_table_180_321_20210517.pickle'

with open(filename, 'rb') as f:
    regression_table = pickle.load(f)
df = regression_table.dropna(axis=0,how='any').copy()
df = df.reset_index(drop=True)


def calculateVariogram_corrected2(df,threshold_dx,distance_matrices,x_var='dist',control_variables=None,dict_control=None):


    i_select = (distance_matrices['dist'] >= threshold_dx[0]) & (distance_matrices['dist'] < threshold_dx[1])
    
    if control_variables:
        for var_ in control_variables:
            if var_ == 'time':
                i_select_2 = ((distance_matrices['time'] >= dict_control['time'][0]) & (distance_matrices['time'] < dict_control['time'][1]))
                i_select = i_select & i_select_2
            else:
                i_select_2 = ((distance_matrices[var_] >= dict_control[var_][0]) & (distance_matrices[var_] < dict_control[var_][1]))
                i_select = i_select & i_select_2
    
    i_select2 = i_select & (~np.eye(len(df),dtype=bool))
    
    nom = 0
    N = 0    
    
    for i in range(len(df)):
        z_j = np.log10(df.loc[:,'kg/m'].values[i_select2[i,:]])
        z_i = np.log10(df.loc[:,'kg/m'].values[i] )
        
        diff2 = (z_i-z_j)**2
        
        nom += diff2.sum()
        N += len(z_j)
     
    if N == 0:
        return np.nan, 0
    else:
        return (1/(2*N))*(nom), N


def calculateVariogram_corrected_errorbar(df,threshold_dx,distance_matrices,x_var='dist',control_variables=None,dict_control=None):


    variance, N = calculateVariogram_corrected2(df,threshold_dx,distance_matrices,x_var,control_variables,dict_control)
     
    n_repeats = len(df)
    variance_samples = np.zeros(n_repeats)
    N_samples = np.zeros(n_repeats)
    
    for i in range(n_repeats):
        df_sample = df.drop(index=i)
        
        resampled_dist_mat = {}
        resampled_dist_mat['dist'] = distance_matrices['dist'][df_sample.index,:][:,df_sample.index]
        if control_variables:
            for var_ in control_variables:
                resampled_dist_mat[var_] = distance_matrices[var_][df_sample.index,:][:,df_sample.index]
        var_, N_ = calculateVariogram_corrected2(df_sample,threshold_dx,resampled_dist_mat,x_var,control_variables,dict_control)
        variance_samples[i] = var_
        N_samples[i] = N_
        
    return variance, N, variance_samples, N_samples


def bin_variable(var_,n_bins):
    max_diff = regression_table.loc[:,var_].max() - regression_table.loc[:,var_].min()
    array_diff = np.linspace(0,max_diff,n_bins+1)
    return array_diff[0:2]


def calculate_distance_matrices(df,control_variables):
    dist_lat = squareform(pdist( df.loc[:,'lat'].values[:,np.newaxis] ))*1.11e2
    dist_lon = squareform(pdist( df.loc[:,'lon'].values[:,np.newaxis] ))*1.11e2 * 0.616
    dist_space = np.sqrt(dist_lat**2 + dist_lon**2)
        
    distance_matrix = {}
    distance_matrix['dist'] = dist_space
    
    for var_ in control_variables:
         if var_ == 'time':
             dist_time = squareform(pdist( df.loc[:,'time'].values[:,np.newaxis] )) / 8.64e13 #days
             distance_matrix['time'] = dist_time
         else:
             distance_matrix[var_] = squareform(pdist( df.loc[:,var_].values[:,np.newaxis] ))
    
    return distance_matrix


x_var = 'dist'
dx_array_std = np.linspace(0,180,19)


var_array = []
n_array = []
dx_array = []

mean_jackknife = []
std_jackknife = []
n_jackknife = []

control_variables = ['time']

dict_control = {}
dict_control['time'] = np.array([0,3],dtype=float) #days
dict_control['dist'] = np.array([0,100])
dict_control['tide_std_030'] = bin_variable('tide_std_030',4)
dict_control['tide_max_003'] = bin_variable('tide_max_003',4)
dict_control['tide_std_003'] = bin_variable('tide_std_003',4)
dict_control['dot_mesh_coast'] = bin_variable('dot_mesh_coast',4)
dict_control['coastal_length_050'] = bin_variable('coastal_length_050',4)
dict_control['tide_tour_max'] = bin_variable('tide_tour_max',4)

distance_matrices = calculate_distance_matrices(df, control_variables)

d_dx = dx_array_std[1] - dx_array_std[0]
n_iter = 5
d_dx_iter = d_dx/(n_iter) # the standard bins are shifted slightly 'n_iter' times to effectively obtain more bins (sliding window)

print('calculating variogram...')
for i5 in range(n_iter):
    dx_array_bnd = dx_array_std + i5*d_dx_iter
    print(dx_array_bnd)
    
    midpoints = 0.5*(dx_array_bnd[:-1] + dx_array_bnd[1:])
    
    for i1 in range(len(dx_array_bnd)-1):
                 
        threshold_dx = [dx_array_bnd[i1],dx_array_bnd[i1+1]]

        var_,sum_pairs,jackk_samples,n_jackk_samples = calculateVariogram_corrected_errorbar(df,threshold_dx,distance_matrices,control_variables=control_variables,dict_control=dict_control)
   
        if sum_pairs == 0:
            var_ = 0
        else:
            var_ = var_#sum_squareDev / (2*sum_pairs)
            
        dx_array.append(midpoints[i1])
        var_array.append(var_)
        n_array.append(sum_pairs)
        
        J_j = len(df)*var_ - (len(df)-1)*jackk_samples
        var_jackk = np.mean(J_j)
        std_var_jackk = np.sqrt( (1/(len(df)*(len(df)-1))) * np.sum( (J_j - var_jackk)**2) )
        
        mean_jackknife.append(var_jackk)
        std_jackknife.append(std_var_jackk)
        n_jackknife.append(n_jackk_samples)
        
    print('%i/%i'%(i5,n_iter))

    
dx_array = np.array(dx_array)
var_array = np.array(var_array)
n_array = np.array(n_array)
mean_jackknife = np.array(mean_jackknife)
std_jackknife = np.array(std_jackknife) 
n_jackknife = np.array(n_jackknife) 

plt.figure(figsize=(10,7))
plt.scatter(dx_array[n_array > 1],var_array[n_array>1],c=n_array[n_array>1]/2,vmin=2,vmax=30)
cbar = plt.colorbar()
cbar.set_label('# of unique data pairs used to calculate variance', rotation=270,verticalalignment='bottom')
plt.title('Variogram in space of log10(data), calculated per year')
if x_var == 'dist':
    plt.xlabel('Lag distance $h$ [km]')
    plt.ylabel(r'Variance at lag distance $\gamma(h)$')     
elif x_var == 'time':
    plt.xlabel('Lag time $t$ [days]')
    plt.ylabel(r'Variance at lag time $\gamma(t)$')              
#plt.tight_layout()

weighted_mean_var = np.array([])
midpoints_unique = np.unique(dx_array)
n_points_arr = np.array([])
for midpoint_ in midpoints_unique:
    index_midpoint = np.where(dx_array == midpoint_)[0][0]
    var_midpoint = var_array[index_midpoint]
    n_midpoint = n_array[index_midpoint]
    weighted_mean_var = np.append(weighted_mean_var,(var_midpoint*n_midpoint).sum() / n_midpoint.sum())
    n_points_arr = np.append(n_points_arr,n_midpoint)


plt.plot(midpoints_unique,weighted_mean_var,'r--',label='weighted mean variance at lag distance')
plt.legend()
plt.ylim(0,np.nanmax(weighted_mean_var)*1.5)

str_variogram = ''
for str_ in control_variables:
    if str_ == 'time':
        str_append = 'time%3.3i%3.3id' % (dict_control['time'][0],dict_control['time'][1])
    else:
        str_append = str_
    str_variogram = str_variogram + str_append + '_dx%i'%d_dx + '_' 

plt.savefig('01_variograms_bootstrapped/' + str_variogram + '.png')


i_sort = np.argsort(dx_array)
plt.figure()
plt.plot(midpoints_unique,weighted_mean_var,'r--',label='weighted mean variance at lag distance')
plt.plot(dx_array[i_sort],var_array[i_sort]+1*std_jackknife[i_sort],'r-',alpha=.8)
plt.plot(dx_array[i_sort],var_array[i_sort]-1*std_jackknife[i_sort],'r-',alpha=.8)

dict_variogram = {}
dict_variogram['midpoints'] = dx_array[i_sort]
dict_variogram['variance'] = var_array[i_sort]
dict_variogram['n_points'] = n_array[i_sort]
dict_variogram['jackknife_mean'] = mean_jackknife[i_sort]
dict_variogram['jackknife_std'] = std_jackknife[i_sort]
dict_variogram['jackknife_n'] = n_jackknife[i_sort]


outfile = open('01_variograms_bootstrapped/' + str_variogram + '.pickle','wb')
pickle.dump(dict_variogram,outfile)
outfile.close()  

