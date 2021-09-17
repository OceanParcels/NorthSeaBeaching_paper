#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:14:47 2021
Script to calculate the orientations of beaches at which the cleanup tours took place
@author: kaandorp
"""
import numpy as np
import xarray as xr
import os
import glob
import matplotlib.pyplot as plt

import shapely.geometry
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from shapely.ops import split, transform

from shapely.geometry import (box, LineString, MultiLineString, MultiPoint,
    Point, Polygon, MultiPolygon, shape)

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


def get_landmask(field):
    
    L = np.all(field[:,0,:,:]==0,axis=0).data
    return L


def get_true_landMask(landMask,i_top = 364,i_bot = 10,i_left = 10,i_right = 269):
    true_landMask = landMask.copy()
    true_landMask[:i_bot,:] = False
    true_landMask[i_top+1:,:] = False
    true_landMask[:,:i_left] = False
    true_landMask[:,i_right+1:] = False
    return true_landMask    


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


inputDir = '/Users/kaandorp/Data/'
files_reanalysis = sorted(glob.glob(os.path.join(inputDir , 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/*')))

file_0 = xr.open_dataset(files_reanalysis[0])
lons = file_0.longitude.values
lats = file_0.latitude.values
fieldMesh_x,fieldMesh_y = np.meshgrid(lons,lats)
dlon, dlat = lons[1]-lons[0], lats[1]-lats[0]
lons_edges = lons-.5*dlon
lons_edges = np.append(lons_edges,lons_edges[-1]+dlon)
lats_edges = lats-.5*dlat
lats_edges = np.append(lats_edges,lats_edges[-1]+dlat)
meshPlotx,meshPloty = np.meshgrid(lons_edges,lats_edges)


file_landMask = './datafiles/datafile_trueLandMask_%ix_%iy' % (len(lons),len(lats))
landMask = np.loadtxt(file_landMask)

landBorder = landMask.copy()
landBorder = getLandBorder(landBorder,2)

lon_unique = np.array([3.4460915, 3.523498 , 3.57     , 3.690463 , 3.707169 , 3.86135  ,
       3.915527 , 3.9963855, 4.0557335, 4.1385885, 4.1926425, 4.3058035,
       4.39     , 4.3901315, 4.455728 , 4.5018315, 4.5433155, 4.576281 ,
       4.5954375, 4.6131295, 4.6293415, 4.64     , 4.67254  , 4.69     ,
       4.71     , 4.7120255, 4.736297 , 4.822059 , 5.050224 , 5.250568 ,
       5.298019 , 5.808781 , 6.188312 ])
lat_unique = np.array([51.41473  , 51.4728675, 51.59     , 51.6304485, 51.669751 ,
       51.790493 , 51.8272295, 51.9254835, 51.892401 , 52.011846 ,
       52.051085 , 52.1327795, 52.2      , 52.2063455, 52.2778685,
       52.340359 , 52.417028 , 52.481756 , 52.525717 , 52.58699  ,
       52.6656525, 52.73     , 52.802325 , 52.84     , 52.9      ,
       52.9080055, 53.081139 , 53.162393 , 53.304177 , 53.402055 ,
       53.408656 , 53.462033 , 53.497415 ])

shpfilename_1 = shpreader.natural_earth(resolution='10m',
                                      category='physical',
                                      name='coastline')


fig = plt.figure(figsize=(10,5),dpi=120)     
ax = plt.axes(projection=ccrs.PlateCarree())    
ax.plot(lon_unique,lat_unique,'o',transform=ccrs.PlateCarree())
ax.set_extent((3.2,6.8,51,53.7))
ax.coastlines(resolution='10m')

beach_orientations = {}
beach_orientations['lon'] = lon_unique
beach_orientations['lat'] = lat_unique
normal_vecs = np.zeros([2,len(lon_unique)])

# c = 0
for i1,(lon_,lat_) in enumerate(zip(lon_unique,lat_unique)):

    normal_vec = find_normal_vector(lon_,lat_,ax,shpfilename_1,radius=4)
    # c+=1
    normal_vecs[:,i1] = normal_vec
    
#exceptions: brouwersdam
normal_vecs[:,3] = normal_vector_2_points(np.array([3.68,3.72]),np.array([51.60,51.66]),ax)
beach_orientations['normal_vec'] = normal_vecs
beach_orientations['normal_vec_mesh'] = find_normal_vector_landborder(lon_unique,lat_unique,fieldMesh_x,fieldMesh_y,landBorder,radius=30)



output_filename = 'datafiles/netcdf_coastal_orientations.nc'
ds = xr.Dataset(
    {"normal_vec_cartopy": (( "xy","location" ), beach_orientations['normal_vec'] ),  
      "normal_vec_mesh": (( "xy","location" ), beach_orientations['normal_vec_mesh'] ),
      "lon": (( "location" ), lon_unique ),
      "lat": (( "location" ), lat_unique ),
      "explanation": 'unit normal vectors calculated for the Dutch coast at BBCT measurement locations'},
)    
ds.to_netcdf(output_filename)


