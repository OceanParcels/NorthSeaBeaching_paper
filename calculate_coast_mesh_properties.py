#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:51:35 2021
Script to calculate coastline properties, such as coastal length per gridbox
Output is written to a netcdf file
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

# BBCT locations
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


projectFolder = './01_coast_mesh_properties/'
if os.environ['USER'] == 'kaandorp': # desktop
    inputDir = os.path.join(os.environ['HOME'],'Data')
    # homeDir = os.environ['HOME']
    outDir = os.path.join(os.getcwd(), projectFolder)
elif os.environ['USER'] == 'kaand004': #gemini
    # homeDir = '/scratch/kaand004'
    inputDir = '/data/oceanparcels/input_data'
    outDir = os.path.join('/scratch/kaand004', projectFolder)
    # os.nice(0) #set low priority for cpu use
    # os.environ['MKL_NUM_THREADS'] = '4'
    
if os.path.exists(outDir):
    print('Writing files to %s\n' % outDir)
else:
    os.makedirs(outDir)
    print('Creating folder %s for output files\n' % outDir)


files_reanalysis = sorted(glob.glob(os.path.join(inputDir , 'CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/*')))

file_0 = xr.open_dataset(files_reanalysis[0])


lons = file_0['longitude']
lats = file_0['latitude']

i_lons = np.arange(len(lons)+1)
i_lats = np.arange(len(lats)+1)

dlon = lons[1]-lons[0]
dlat = lats[1]-lats[0]
lons_edges = lons-.5*dlon
lons_edges = np.append(lons_edges,lons_edges[-1]+dlon)#[i_lons]
lats_edges = lats-.5*dlat
lats_edges = np.append(lats_edges,lats_edges[-1]+dlat)#[i_lats]
meshPlotx,meshPloty = np.meshgrid(lons_edges,lats_edges)

coastline_length = np.nan*np.zeros([len(lats),len(lons)])

# array defining the North Sea front: used to discard coastlines which are 'inland', such as the Waddensea
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


fig = plt.figure(figsize=(10,5),dpi=120)     
ax = plt.axes(projection=ccrs.PlateCarree())    
ax.set_extent((3.2,6.8,51,53.7))
ax.coastlines(resolution='10m')


for lon_grid in lons_edges:
    ax.plot([lon_grid,lon_grid],[lats.min(),lats.max()],color='silver',transform=ccrs.PlateCarree(),linewidth=.5)

for lat_grid in lats_edges:
    ax.plot([lons.min(),lons.max()],[lat_grid,lat_grid],color='silver',transform=ccrs.PlateCarree(),linewidth=.5)


shpfilename = shpreader.natural_earth(resolution='10m',
                                      category='physical',
                                      name='coastline')

box_waterfront = Polygon(water_front)
line_waterfront = LineString(water_front)
line_waterfront_offset = line_waterfront.parallel_offset(-0.01) #offset the waterfront line outwards
box_waterfront_offset = Polygon(line_waterfront_offset)
dams = [np.array([[3.682,51.62],[3.704,51.66]]),np.array([[3.825,51.75],[3.866,51.80]])]

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
                    
                    if np.isnan(coastline_length[i2,i1]):
                        coastline_length[i2,i1] = transform(project,coastline_in_box).length
                    else:
                        coastline_length[i2,i1] += transform(project,coastline_in_box).length                        

        #otherwise, the box is outside of the waterfront box, just take the length of the sections
        else:
            coastline_split = split(coast_geom,box_linestring)

            indices_in_box = give_element_within_bounds2(coastline_split,box_analyze,tol=1.00)
            
            for index_in_box in indices_in_box:
                coastline_in_box = coastline_split[index_in_box]
                
                coastline_lons = np.array(coastline_in_box.xy[0])
                coastline_lats = np.array(coastline_in_box.xy[1])
                    
                ax.plot(coastline_lons,coastline_lats,'ro',transform=ccrs.PlateCarree())
                
                if np.isnan(coastline_length[i2,i1]):
                    coastline_length[i2,i1] = transform(project,coastline_in_box).length
                else:
                    coastline_length[i2,i1] += transform(project,coastline_in_box).length
          
    if box_poly.contains(coast_geom):
        print('coastline contained in box')
        coastline_length[i2,i1] += transform(project,coast_geom).length



for i1,lon_left,lon_right in zip(i_lons,lons_edges[:-1],lons_edges[1:]):
    
    for i2,lat_lower, lat_upper in zip(i_lats,lats_edges[:-1],lats_edges[1:]):
        
        reader = shpreader.Reader(shpfilename)
        coastlines = reader.records()
   
        box_analyze = [[lon_left, lat_lower], [lon_left, lat_upper], 
                       [lon_right, lat_upper], [lon_right, lat_lower], [lon_left, lat_lower]]
        box_poly = shapely.geometry.Polygon(box_analyze)
        
        
        any_intersect = False
        
        c = 0
        for coast_ in coastlines:

            calculate_coastal_section_length(coast_.geometry)
            c += 1
        for dam_ in dams:
            calculate_coastal_section_length(LineString(dam_))
            

                
    print(i1)

output_filename = 'datafiles/netcdf_coastal_lengths.nc'
ds = xr.Dataset(
    {"coastline_length": (( "lat", "lon"), coastline_length ),       
     "explanation": 'coasline length per cell based on cartopy 10m coastlines, units: meters. Only use near Netherlands (projection is based on Netherlands)'},
    coords={
        "lon": lons.values,
        "lat": lats.values,
    },
)   
   
ds.to_netcdf(output_filename)
   


ax.pcolormesh(meshPlotx,meshPloty,coastline_length[i_lats[0]:i_lats[-1],i_lons[0]:i_lons[-1]],transform=ccrs.PlateCarree(),zorder=0)
ax.plot(line_waterfront.xy[0],line_waterfront.xy[1],'b-',transform=ccrs.PlateCarree())
