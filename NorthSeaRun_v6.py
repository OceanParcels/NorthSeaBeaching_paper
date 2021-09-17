#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:24:34 2020
Script to run the OceanParcels simulations used to create the beaching fluxes used.
Can be run from the command line with an argument -source, which can be f/r/p specifying the source of virtual particles
(fisheries,rivers,coastal population respectively)
@author: kaandorp
"""
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, Variable, Field, VectorField
import numpy as np
import os
import glob
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from datetime import timedelta,datetime
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
import shapely
import cartopy.io.shapereader as shpreader
import shapefile
import csv
import matplotlib
from time import sleep
from parcels import rng as random
import math
from parcels.tools.converters import Geographic, GeographicPolar 
import cmocean
from argparse import ArgumentParser
import parcels.rng as ParcelsRandom

def get_landmask(field,outfile='./tmp_landmask'):
    if os.path.exists(outfile):
        landMask = np.loadtxt(outfile)
        landMask = np.array(landMask,dtype=bool)    
    else:
        # landMask = np.all(file_0['uo'][:,0,:,:]==0,axis=0).data
        landMask = np.all(np.isnan(field[:,0,:,:].data),axis=0)
        np.savetxt(outfile,landMask)
    return landMask

def get_true_landmask(landMask,i_top = 364,i_bot = 10,i_left = 10,i_right = 269,outfile='./tmp_truelandmask'):
    if os.path.exists(outfile):
        true_landMask = np.loadtxt(outfile)
        true_landMask = np.array(true_landMask,dtype=int)    
    else:
        true_landMask = landMask.copy()
        true_landMask[:i_bot,:] = False
        true_landMask[i_top+1:,:] = False
        true_landMask[:,:i_left] = False
        true_landMask[:,i_right+1:] = False
        np.savetxt(outfile,true_landMask)
    return true_landMask  


def get_coastmask(landMask,i_top=np.inf,i_bot=-1,i_left=-1,i_right=np.inf,outfile='./tmp_coastmask'):
    """
    Function to obtain a mask of the coast, uses the landmask and searches for 
    boundaries with the sea (horiz./vert. adjacent cells only)
    TODO: check for cyclic boundaries
    TODO: check diagonal cells as well?
    """

    if os.path.exists(outfile):
        coastMask = np.loadtxt(outfile)
        coastMask = np.array(coastMask,dtype=int)
    else:
        print('calculating coast mask...')
        n_lat = landMask.shape[0]
        n_lon = landMask.shape[1]
        
        coastMask = np.zeros([n_lat,n_lon],dtype=bool)
    
        for i1 in range(n_lat):
            for i2 in range(n_lon):
                
                check_bot = True
                check_top = True
                check_left = True
                check_right = True
                
                # check whether land is located at boundary
                if i1 == 0 or i1 >= i_top:
                    check_top = False
                if i1 == n_lat-1 or i1 <= i_bot:
                    check_bot = False
                if i2 == 0 or i2 >= i_right:
                    check_left = False
                if i2 == n_lon-1 or i2 <= i_left:
                    check_right = False
                    
                # check whether cell is land, if so look for coast
                if landMask[i1,i2] == 1:
                    
                    if check_top:
                        if landMask[i1-1,i2] == 0:
                            coastMask[i1-1,i2] = True
                    if check_bot:
                        if landMask[i1+1,i2] == 0:
                            coastMask[i1+1,i2] = True
                    if check_left:
                        if landMask[i1,i2-1] == 0:
                            coastMask[i1,i2-1] = True
                    if check_right:
                        if landMask[i1,i2+1] == 0:
                            coastMask[i1,i2+1] = True
        np.savetxt(outfile,coastMask)
            
    return coastMask


def gridMPWData(fieldMesh_x,fieldMesh_y,landMask,outfile='./tmp_mpw_gridded'):

    if os.path.exists(outfile):
        mpw_mat_final = np.loadtxt(outfile)
    else:    
        shpfilename = shpreader.natural_earth(resolution='50m',
                                              category='cultural',
                                              name='admin_0_countries')
        
        data_mpw = pd.read_excel('/Users/kaandorp/Data/PlasticData/PlasticRiverInputs_Schmidt/plastic_rivers2sea_v2.xlsx',sheet_name='Mismanaged Plastic Waste')
        
        
        reader = shpreader.Reader(shpfilename)
        
        countries = reader.records()
        country = next(countries)
    
        mpw_mat = np.zeros(fieldMesh_x.shape)
        
        all_country_names = ['Belgium','Denmark','France','Germany','Iceland','Ireland','Netherlands',
                             'Norway','Portugal','Spain','Sweden','United Kingdom','Guernsey','Jersey','Isle of Man','Faeroe Islands']
        
        country_names = []
        mpw_ = []
        while country:
        
            geom = country.geometry
            country_name = country.attributes['NAME_LONG']
            
            if country_name in all_country_names:
                if country_name == 'Denmark':
                    break
                country_names.append(country_name)
        
                mpw = np.nan
                # go through excel data, find matching country name
                for i1 in range(len(data_mpw['Country'])):
                    country_mpw = data_mpw.loc[i1,'Country']
                    
                    if country_mpw in country_name:
                        mpw = data_mpw.iloc[i1,6]
                        
                if np.isnan(mpw):
                    pass
                else:
                    mpw_.append(mpw)
                
                # create initial matrix containing mpw
                for i1 in range(len(lons)):
                    for i2 in range(len(lats)):
                        xy_point = shapely.geometry.Point(fieldMesh_x[i2,i1],fieldMesh_y[i2,i1]) 
                        
                        if geom.contains(xy_point):
                            mpw_mat[i2,i1] = mpw
                        
            if country_name == 'Sint Maarten': #last country @ 50m
                break
            
            country = next(countries)
        
        #create refined matrix using the given landmask   
        nanmask = np.isnan(mpw_mat)
        mpw_mat_final = np.zeros(landMask.shape)
        
        for i1 in range(len(lons)):
            for i2 in range(len(lats)):
                
                if landMask[i2,i1]:
                    
                    fillVal = mpw_mat[i2,i1]
                    
                    if np.isnan(fillVal):
                        fillVal = griddata((fieldMesh_x[~nanmask],fieldMesh_y[~nanmask]),mpw_mat[~nanmask],(fieldMesh_x[i2,i1],fieldMesh_y[i2,i1]))
                        
                    mpw_mat_final[i2,i1] = fillVal
        
        np.savetxt(outfile,mpw_mat_final)
    

def get_coastallandmask(landMask):
    threshold = 50
    
    indices_land = np.where(landMask == 1)
    ids_lon_ = indices_land[1]
    ids_lat_ = indices_land[0]
    
    coastalLandMask = np.zeros(landMask.shape,dtype=bool)
    
    for i2 in range(len(indices_land[0])):
        lon_ = lons[ids_lon_[i2]]
        lat_ = lats[ids_lat_[i2]]      
        # i_min = np.argmin 
        dist_to_sea = np.sqrt( ((lon_ - fieldMesh_x[~landMask])*np.cos(lat_*np.pi/180)*1.11e2)**2 + ((lat_ - fieldMesh_y[~landMask])*1.11e2)**2)
        i_min_dist = np.argmin(dist_to_sea)
        if dist_to_sea[i_min_dist] < threshold:
            coastalLandMask[ids_lat_[i2],ids_lon_[i2]] = True

    return coastalLandMask


def initializePopulationMatrices(landMask,coastMask,fieldMesh_x,fieldMesh_y,MPWFile='./datafiles/datafile_MPW_gridded_297x_375y',
                                      threshold=50,lon_min=-20.,lon_max=13.,lat_min=40.,lat_max=66.,
                                      outfile_popmat='./tmp_popmat',outfile_coastmat='./tmp_coastalzone'):
    
    if os.path.exists(outfile_popmat) and os.path.exists(outfile_coastmat):
        populationCoastMatrix = np.loadtxt(outfile_popmat).reshape([5,len(lats),len(lons)])
        coastalLandMask = np.loadtxt(outfile_coastmat)
    else:
        print('calculating population input...')
        coastMask = np.array(coastMask,dtype=bool)
        
        indices_land = np.where(landMask == 1)
        ids_lon_ = indices_land[1]
        ids_lat_ = indices_land[0]
     
        
        mpw_gridded = np.loadtxt(MPWFile)
    
        populationCoastMatrix = np.zeros([5,fieldMesh_x.shape[0],fieldMesh_x.shape[1]])
        pop_datafile            = os.path.join(inputDir , 'populationDensity/gpw_v4_e_atotpopbt_dens_2pt5_min.nc')
        
        dataPop = Dataset(pop_datafile)
        
        lons_data = dataPop.variables['longitude'][:]
        lats_data = dataPop.variables['latitude'][:]
        ids_lon = [max(np.argmin(np.abs(lon_min - lons_data)),1)-1,
                   min(np.argmin(np.abs(lon_max - lons_data)),len(lons_data)-1)+1]
        ids_lat = [max(np.argmin(np.abs(lat_min - lats_data)),1)-1,
                   min(np.argmin(np.abs(lat_max - lats_data)),len(lats_data)-1)+1]
        meshPop_x, meshPop_y = np.meshgrid(lons_data[ids_lon[0]:ids_lon[1]],lats_data[ids_lat[1]:ids_lat[0]])
    
        
        coastalLandMask = np.zeros(landMask.shape,dtype=bool)
        indices_mat = np.meshgrid(np.arange(len(lons)),np.arange(len(lats)))
        
        for i1 in range(5):
            densityPop = dataPop.variables['Population Density, v4.10 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'][i1,ids_lat[1]:ids_lat[0],ids_lon[0]:ids_lon[1]]
            
        
            densityPop_i = griddata((meshPop_x.ravel(),meshPop_y.ravel()),densityPop.ravel(),(fieldMesh_x,fieldMesh_y))
            densityPop_i[densityPop_i < 0] = 0
            densityPop_i[np.isnan(densityPop_i)] = 0 
        
            
            density_MPW = densityPop_i * mpw_gridded
            
            for i2 in range(len(indices_land[0])):
                lon_ = lons[ids_lon_[i2]]
                lat_ = lats[ids_lat_[i2]]      
    
                # distance one land cell to all coastal cells
                dist_to_coast = np.sqrt( ((lon_ - fieldMesh_x[coastMask])*np.cos(lat_*np.pi/180)*1.11e2)**2 + ((lat_ - fieldMesh_y[coastMask])*1.11e2)**2)
                
                # where on the coastmask is the distance from the land cell minimum
                i_min_dist = np.argmin(dist_to_coast)
                if dist_to_coast[i_min_dist] < threshold:
                    
                    i_lon_coast_closest = indices_mat[0][coastMask][i_min_dist]
                    i_lat_coast_closest = indices_mat[1][coastMask][i_min_dist]
                    
                    populationCoastMatrix[i1,i_lat_coast_closest,i_lon_coast_closest] += density_MPW[ids_lat_[i2],ids_lon_[i2]]
                    
                    coastalLandMask[ids_lat_[i2],ids_lon_[i2]] = True
    
        np.savetxt(outfile_popmat, populationCoastMatrix.reshape(populationCoastMatrix.shape[0]*populationCoastMatrix.shape[1],populationCoastMatrix.shape[2]))
        np.savetxt(outfile_coastmat, coastalLandMask)
        
    return populationCoastMatrix,coastalLandMask


def initializeRiverMatrices(riverShapeFile, pollutionFile, coastMask, fieldMesh_x, fieldMesh_y, selection_estim='mid', 
                            plot_coastalinput = False,outfile='./tmp_rivmat'):
    """
    Initialize the river release matrices. This is one matrix for every specifed month.
    Based on data from Lebreton, where catchment areas/runoff were estimated together with
    MPW from Jambeck
    selection_estim: choices are 'mid','low','high', which correspond to the 
    lower-upper and middle of the confidence bounds of MPW emitted by rivers from
    Lebreton
    """
    if os.path.exists(outfile):
        riverInputMatrix = np.loadtxt(outfile).reshape([12,len(lats),len(lons)])

    else:
        print('calculating river input matrices...')
        coastMask = np.array(coastMask,dtype=int)
        #import shapefile
        sf = shapefile.Reader(riverShapeFile)
        
        #extract files within domain
        plottingDomain = [lons.min(),lons.max(),lats.min(),lats.max()]
        
        rivers = {}
        rivers['longitude'] = np.array([])
        rivers['latitude'] = np.array([])
        rivers['ID'] = np.array([],dtype=int)
        rivers['dataArray'] = np.array([])
        
        for i1 in range(len(sf.shapes())):
            long = sf.shape(i1).points[0][0]
            lat = sf.shape(i1).points[0][1]
            
            if plottingDomain[0] < long <plottingDomain[1] and plottingDomain[2] < lat < plottingDomain[3]:
                rivers['longitude'] = np.append(rivers['longitude'],long)
                rivers['latitude'] = np.append(rivers['latitude'],lat)
                rivers['ID'] = np.append(rivers['ID'],i1)
                
                
        with open(pollutionFile, 'r',encoding='ascii') as csvfile:
            filereader = csv.reader(csvfile, delimiter=';')
            i1 = 0
            for row in filereader:
                
                if i1 == 0:
                    riverHeaders = row
                
                if i1 > 0:
                    
                
                    data_ID = i1-1
                    
                    if i1 == 1:
                        dataArray = [float(row[i2].replace(',','.')) for i2 in range(len(row))]
                        rivers['dataArray'] = dataArray
                    else:
                        if data_ID in rivers['ID']:
                            dataArray = [float(row[i2].replace(',','.')) for i2 in range(len(row))]
                            rivers['dataArray'] = np.vstack([rivers['dataArray'],dataArray])
                i1 += 1
            
        # check which columns contain data for the specified confidence level from selection_estim parameter    
        columnNumbers = []
    #    i1 = 0
        for idx, columnName in enumerate(riverHeaders):
            if (selection_estim + '_') in columnName:
                columnNumbers.append(idx)
    #        i1 += 1
        assert(len(columnNumbers) == 12), "there should be 12 entries corresponding to waste emitted per month"
        
        coastIndices = np.where(coastMask == 1)
        assert(np.shape(coastIndices)[0] == 2), "coastMask.data should be 2 by something"
        
        # array containing indices of rivers not belonging to mediterranean, which are to be deleted
        deleteEntries = np.array([],dtype=int)
        
        # matrix corresponding to fieldmesh, with per coastal cell the amount of river pollution
        riverInputMatrix = np.zeros([12,fieldMesh_x.shape[0],fieldMesh_x.shape[1]])
        
        # for every river
        for i1 in range(len(rivers['longitude'])):
            
            lon_river = rivers['longitude'][i1]
            lat_river = rivers['latitude'][i1]
            
            dist = 1e10
            # check which point is closest
            for i2 in range(np.shape(coastIndices)[1]):
                lon_coast = lons[coastIndices[1][i2]]
                lat_coast = lats[coastIndices[0][i2]]
            
                lat_dist = (lat_river - lat_coast) * 1.11e2
                lon_dist = (lon_river - lon_coast) * 1.11e2 * np.cos(lat_river * np.pi / 180)
                dist_tmp = np.sqrt(np.power(lon_dist, 2) + np.power(lat_dist, 2))
                
                # save closest distance
                if dist_tmp < dist:
                    dist = dist_tmp
                    lat_ID = coastIndices[0][i2]
                    lon_ID = coastIndices[1][i2]
                
            # if distance to closest point > threshold (3*approx cell length), delete entry
            if dist > 3*0.125*1.11e2:
                deleteEntries = np.append(deleteEntries,i1)
            # else: get pollution river, and add to releasematrix
            else:
                # add plastic input as obtained from the dataset
                for idx, val in enumerate(columnNumbers):
                    riverInputMatrix[idx,lat_ID,lon_ID] += rivers['dataArray'][i1,val]
                
        
        # rivers ending in mediterranean
        rivers_medit = {}
        rivers_medit['longitude'] = np.delete(rivers['longitude'],deleteEntries)
        rivers_medit['latitude'] = np.delete(rivers['latitude'],deleteEntries)
        rivers_medit['ID'] = np.delete(rivers['ID'],deleteEntries)
        rivers_medit['dataArray'] = np.delete(rivers['dataArray'],deleteEntries,axis=0)
                
        if plot_coastalinput:
            times = ['jan','feb','mar','apr','may','june','july','aug','sep','oct','nov','dec']
            for i1 in range(riverInputMatrix.shape[0]):
                
                minval = 1e-15
                maxval = riverInputMatrix.max()
                riverInputMatrix_plt = np.copy(riverInputMatrix)
                riverInputMatrix_plt[riverInputMatrix_plt == 0.0] = 1e-15
                figsize=(30,12)
                plt.figure(figsize=figsize)
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines(resolution='10m')
                cmap = plt.cm.inferno
                handle = ax.contourf(fieldMesh_x,fieldMesh_y,riverInputMatrix_plt[i1,:,:],vmin=minval,vmax=maxval,norm=matplotlib.colors.LogNorm(),cmap=cmap,transform=ccrs.PlateCarree())
                plt.colorbar(handle)
                plt.title('River inputs, %s' % times[i1])
        
        
            dataArray_ID = 2

            plt.figure(figsize=(12,24))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.set_extent((plottingDomain[0],plottingDomain[1],plottingDomain[2],plottingDomain[3]), ccrs.PlateCarree())
    
            maxVal = 1.0
            maxPoll = np.max(rivers['dataArray'][:,dataArray_ID])
            maxPoll = 200
            for i1 in range(len(rivers['longitude'])):
                poll = rivers['dataArray'][i1,dataArray_ID]
                
                if 0.2 < poll < 2:
                    poll = 50
                    color = 'green'
                elif 2 < poll < 20:
                    poll = 75
                    color = 'yellow'
                elif 20 < poll < 200:
                    poll = 100
                    color = 'orange'
                elif poll > 200:
                    poll = 200
                    color = 'red'
                else:
                    poll = 10
                    color = 'blue'
                
                circle1 = plt.Circle((rivers['longitude'][i1],rivers['latitude'][i1]), maxVal*(poll/maxPoll), color=color,fill=True, transform=ccrs.PlateCarree())
                ax.add_patch(circle1)
            plt.show()
 
            plt.figure(figsize=(12,24))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.set_extent((plottingDomain[0],plottingDomain[1],plottingDomain[2],plottingDomain[3]), ccrs.PlateCarree())
    
            maxVal = 1.0
            maxPoll = np.max(rivers_medit['dataArray'][:,dataArray_ID])
            maxPoll = 200
            for i1 in range(len(rivers_medit['longitude'])):
                poll = rivers_medit['dataArray'][i1,dataArray_ID]
                
                if 0.2 < poll < 2:
                    poll = 50
                    color = 'green'
                elif 2 < poll < 20:
                    poll = 75
                    color = 'yellow'
                elif 20 < poll < 200:
                    poll = 100
                    color = 'orange'
                elif poll > 200:
                    poll = 200
                    color = 'red'
                else:
                    poll = 10
                    color = 'blue'
                
                circle1 = plt.Circle((rivers_medit['longitude'][i1],rivers_medit['latitude'][i1]), maxVal*(poll/maxPoll), color=color,fill=True, transform=ccrs.PlateCarree())
                ax.add_patch(circle1)
            plt.show()
        
        np.savetxt(outfile, riverInputMatrix.reshape(riverInputMatrix.shape[0]*riverInputMatrix.shape[1],riverInputMatrix.shape[2]))

        

    return riverInputMatrix


def initializeFisheriesMatrices(landMask,lons,lats,outfile='./tmp_fisheries'):
    
    if os.path.exists(outfile):
        fishingMatrix = np.loadtxt(outfile)
    else:
        print('calculating fishing input...')
        homeDir = os.environ['HOME']
        fileList = list(np.sort(glob.glob(os.path.join(homeDir , 'Data/Fisheries/daily_csvs/*'))))
        
        # enter your domain here, make sure it is rounded to 0.1 values (since this is the precision given in the fisheries file)
        # the values are specified for the lower/left edges of the mesh (hence the 36.9 instead of 37)
        lon_min = np.floor(lons.min())
        lon_max = np.ceil(lons.max())
        lat_min = np.floor(lats.min())
        lat_max = np.ceil(lats.max())
        
        lon_edge = np.linspace(lon_min,lon_max,int((lon_max - lon_min + 0.1)*10) )
        lat_edge = np.linspace(lat_min,lat_max,int((lat_max - lat_min + 0.1)*10) )
        
        mat_fh = np.zeros([len(lon_edge),len(lat_edge)]) #matrix with yearly fishing intensity
        mat_fh_month = np.zeros([5*12,len(lon_edge),len(lat_edge)]) #matrix with monthly fishing intensity
        X,Y = np.meshgrid(lon_edge,lat_edge)
        X = X.T
        Y = Y.T
        
        lons_mid = lon_edge + 0.05 #midpoints, mesh dx = 0.1
        lats_mid = lat_edge + 0.05
        X_mid,Y_mid = np.meshgrid(lons_mid,lats_mid)
        X_mid = X_mid.T
        Y_mid = Y_mid.T
        
        X2,Y2 = np.meshgrid(np.append(lon_edge,lon_edge[-1]+0.1),np.append(lat_edge,lat_edge[-1]+0.1))
        X2 = X2.T
        Y2 = Y2.T
        
        for i1 in range(len(fileList)):
            year = int(os.path.basename(fileList[i1])[0:4])
            month = int(os.path.basename(fileList[i1])[5:7])
            day = int(os.path.basename(fileList[i1])[8:10])
            
            index_month = (year-2012)*12 + month-1
            
            data = pd.read_csv(fileList[i1])
            lats = data.loc[:,'lat_bin']/10
            lons = data.loc[:,'lon_bin']/10
            fh = data.loc[:,'fishing_hours']
            
            indices_med = ((lons >= lon_min) & (lons <= lon_max)) & ((lats >= lat_min) & (lats <= lat_max))
            
            lats_med = lats[indices_med]
            lons_med = lons[indices_med]
            fh_med = fh[indices_med]
            
            for i2 in range(len(lats_med)):
                index_lon = np.where(abs(lon_edge-lons_med.iloc[i2]) < 0.00001)  
                index_lat = np.where(abs(lat_edge-lats_med.iloc[i2]) < 0.00001) 
                mat_fh[index_lon,index_lat] += fh_med.iloc[i2]
                mat_fh_month[index_month,index_lon,index_lat] += fh_med.iloc[i2]                                
        
        fishingMatrix = griddata((X_mid.ravel(),Y_mid.ravel()),mat_fh.ravel(),(fieldMesh_x,fieldMesh_y))
        fishingMatrix[landMask] = 0
        
        np.savetxt(outfile, fishingMatrix)

    return fishingMatrix

def get_input_matrix(source,time):
    input_matrix = []
    monthly_fraction = 1.
    if source == 'f':
        # print('Getting matrix for fishing activity...')
        input_matrix = fishingInputMatrix / fishingInputMatrix.sum()
    elif source == 'r':
        # print('Getting matrix for river input...')
        t_month = time.month
        input_matrix = riverInputMatrices[t_month-1,:,:] / riverInputMatrices[t_month-1,:,:].sum()
        monthly_fraction = riverInputMatrices[t_month-1,:,:].sum() / (riverInputMatrices.sum()/12)
    elif source == 'p':
        input_matrix = populationInputMatrices[3,:,:] / populationInputMatrices[3,:,:].sum()
        # print('Getting matrix for population input...')
    
    return input_matrix,monthly_fraction

def writeParticleInfo(fname,t,lat,lon,t0=datetime(2001,1,1)):
    """
    Write release info particles for efficient post-processing
    """
    if type(t[0]) == np.float64:       
        f = open(fname,'w')
        f.write('%s\n' % str(t0))
        for i1 in range(len(t)):
            f.write('%i\t%3.10f\t%3.10f\n' % (t[i1],lat[i1],lon[i1]) )
        f.close()  

    elif type(t[0]) == datetime:
        t_epoch = t-t0
        f = open(fname,'w')
        f.write('%s\n' % str(t0))
        for i1 in range(len(t)):
            f.write('%i\t%3.10f\t%3.10f\n' % (t_epoch[i1].total_seconds(),lat[i1],lon[i1]) )
        f.close()          

    else:
        raise RuntimeError('not implemented')

def read_xarray_files(files):
    print('reading files using xarray, this can take some time...')
    df_out = xr.open_dataset(files[0])
    
    for i1 in np.arange(len(files))[1:]:
        df_new = xr.open_dataset(files[i1])
        
        # check if first time is same as last time, delete if so
        if df_out['time'][-1].data == df_new['time'][0].data:
            df_out = df_out.where(df_out['time'] < df_out['time'][-1],drop=True)
        
        df_out = xr.concat([df_out,df_new],dim='time')
    
    return df_out


def get_true_landMask(landMask,i_top = 364,i_bot = 10,i_left = 10,i_right = 269):
    true_landMask = landMask.copy()
    true_landMask[:i_bot,:] = False
    true_landMask[i_top+1:,:] = False
    true_landMask[:,:i_left] = False
    true_landMask[:,i_right+1:] = False
    return true_landMask    


def calculateLandCurrent(landMask,fieldMesh_x,fieldMesh_y,i_top = 364,i_bot = 10,
                         i_left = 10,i_right = 269,do_plot=False,outfile_1='./tmp_landcurrentx',outfile_2='./tmp_landcurrenty'):
    """
    Calculate closest cell with water for all land cells, create vector field to
    closest water cell
    """
    if os.path.exists(outfile_1) and os.path.exists(outfile_2):
        landVectorField_x = np.loadtxt(outfile_1)
        landVectorField_y = np.loadtxt(outfile_2)
    else:
        print('Calculating landcurrent...')
        landMask_true = get_true_landMask(landMask,i_top = i_top,i_bot = i_bot,i_left = i_left,i_right = i_right)

        # true_landMask[:,i_top:] = False
       
        oceanCellIndices = np.where(landMask == 0)         #which indices in the grid correspond to ocean
        landCellIndices = np.where(landMask_true == 1) #which correspond to land
            
        
        landVectorField_x = np.zeros(fieldMesh_x.shape)
        landVectorField_y = np.zeros(fieldMesh_y.shape)
        
        for i1 in range(len(landCellIndices[1])): #go through all land cells
            lon_coast = fieldMesh_x[landCellIndices[0][i1],landCellIndices[1][i1]]    
            lat_coast = fieldMesh_y[landCellIndices[0][i1],landCellIndices[1][i1]]
    
            distMat_lon = (lon_coast - fieldMesh_x[oceanCellIndices[0],oceanCellIndices[1]]) #find distances coastal element w.r.t. ocean cells. 
            distMat_lat = (lat_coast - fieldMesh_y[oceanCellIndices[0],oceanCellIndices[1]])
    
            distance_toOcean = np.sqrt(np.power(distMat_lon, 2) + np.power(distMat_lat, 2))    
            minDist = np.min(distance_toOcean)
            i_minDist = np.where(distance_toOcean == minDist)
            if len(i_minDist[0]) == 1:
                #easy case: vector to single point
                lon_ocean = fieldMesh_x[oceanCellIndices[0][i_minDist],oceanCellIndices[1][i_minDist]]
                lat_ocean = fieldMesh_y[oceanCellIndices[0][i_minDist],oceanCellIndices[1][i_minDist]]
                
                landVectorField_x[landCellIndices[0][i1],landCellIndices[1][i1]] = (lon_ocean - lon_coast) / np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)
                landVectorField_y[landCellIndices[0][i1],landCellIndices[1][i1]] = (lat_ocean - lat_coast) / np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)
            
            elif len(i_minDist[0]) > 1:
                #multiple ocean cells are the closest: take mean x and y vals
                lon_ocean = np.mean(fieldMesh_x[oceanCellIndices[0][i_minDist],oceanCellIndices[1][i_minDist]])
                lat_ocean = np.mean(fieldMesh_y[oceanCellIndices[0][i_minDist],oceanCellIndices[1][i_minDist]])
                
                landVectorField_x[landCellIndices[0][i1],landCellIndices[1][i1]] = (lon_ocean - lon_coast) / np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)
                landVectorField_y[landCellIndices[0][i1],landCellIndices[1][i1]] = (lat_ocean - lat_coast) / np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)            
    
    
        if do_plot:
            plt.figure()
            plt.quiver(landVectorField_x[::3,::3],landVectorField_y[::3,::3],scale=30)
        
        np.savetxt(outfile_1, landVectorField_x)        
        np.savetxt(outfile_2, landVectorField_y)        

    return landVectorField_x,landVectorField_y


def coastalDynamics(particle, fieldset, time):
    '''
    Kernel which calculates in-products and magnitudes of weather variables.
    For the in-product, we take the maximum on the rectangular grid 
    (i.e. we look at all neighboring cellsif they are land, and then take the max)
    '''
    lon_spacing_ = fieldset.lon_spacing
    lat_spacing_ = fieldset.lat_spacing
        
    landZone_p = fieldset.landZone[time, particle.depth, particle.lat, particle.lon] 
    
    if (landZone_p > fieldset.landzone_threshold):   
        
        (u_curr, v_curr) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        (u_stokes, v_stokes) = fieldset.UV_Stokes[time, particle.depth, particle.lat, particle.lon]
        (u_wind, v_wind) = fieldset.UV_wind[time, particle.depth, particle.lat, particle.lon]
        u_tide = particle.U_tide
        v_tide = particle.V_tide
        tidal_height = particle.tidal_height
        VHM0_ = fieldset.VHM0[time, particle.depth, particle.lat, particle.lon]
        
        #-------------------------dot products-------------------------------
        land_top = fieldset.land_nearest[time, particle.depth, particle.lat+.5*lat_spacing_, particle.lon]
        land_bot = fieldset.land_nearest[time, particle.depth, particle.lat-.5*lat_spacing_, particle.lon]
        land_right = fieldset.land_nearest[time, particle.depth, particle.lat, particle.lon+.5*lon_spacing_]
        land_left = fieldset.land_nearest[time, particle.depth, particle.lat, particle.lon-.5*lon_spacing_]
    
        dot_wind = math.nan
        dot_curr = math.nan
        dot_stokes = math.nan
        dot_tide = math.nan

        if land_top > 0.5:
            
            dot_curr_t = u_curr*0. + v_curr*1.
            dot_stokes_t = u_stokes*0. + v_stokes*1.
            dot_wind_t = u_wind*0. + v_wind*1.
            dot_tide_t = u_tide*0. + v_tide*1.
            
            if (dot_wind_t > dot_wind) or math.isnan(dot_wind):
                dot_wind = dot_wind_t 
            if dot_curr_t > dot_curr or math.isnan(dot_curr):
                dot_curr = dot_curr_t 
            if dot_stokes_t > dot_stokes or math.isnan(dot_stokes):
                dot_stokes = dot_stokes_t 
            if dot_tide_t > dot_tide or math.isnan(dot_tide):
                dot_tide= dot_tide_t
    
        if land_bot > 0.5:
            
            dot_curr_t= u_curr*0. + v_curr*-1.
            dot_stokes_t = u_stokes*0. + v_stokes*-1.
            dot_wind_t = u_wind*0. + v_wind*-1.
            dot_tide_t = u_tide*0. + v_tide*-1.
            
            if (dot_wind_t > dot_wind) or math.isnan(dot_wind):
                dot_wind = dot_wind_t 
            if dot_curr_t > dot_curr or math.isnan(dot_curr):
                dot_curr = dot_curr_t 
            if dot_stokes_t > dot_stokes or math.isnan(dot_stokes):
                dot_stokes = dot_stokes_t 
            if dot_tide_t > dot_tide or math.isnan(dot_tide):
                dot_tide= dot_tide_t
            
        if land_left > 0.5:
            
            dot_curr_t = u_curr*-1. + v_curr*0.
            dot_stokes_t = u_stokes*-1. + v_stokes*0.
            dot_wind_t = u_wind*-1. + v_wind*0.
            dot_tide_t = u_tide*-1. + v_tide*0.
            
            if (dot_wind_t > dot_wind) or math.isnan(dot_wind):
                dot_wind = dot_wind_t 
            if dot_curr_t > dot_curr or math.isnan(dot_curr):
                dot_curr = dot_curr_t 
            if dot_stokes_t > dot_stokes or math.isnan(dot_stokes):
                dot_stokes = dot_stokes_t 
            if dot_tide_t > dot_tide or math.isnan(dot_tide):
                dot_tide= dot_tide_t
        
        if land_right > 0.5:
            
            dot_curr_t = u_curr*1. + v_curr*0.
            dot_stokes_t = u_stokes*1. + v_stokes*0.
            dot_wind_t = u_wind*1. + v_wind*0.
            dot_tide_t = u_tide*1. + v_tide*0.
            
            if (dot_wind_t > dot_wind) or math.isnan(dot_wind):
                dot_wind = dot_wind_t 
            if dot_curr_t > dot_curr or math.isnan(dot_curr):
                dot_curr = dot_curr_t 
            if dot_stokes_t > dot_stokes or math.isnan(dot_stokes):
                dot_stokes = dot_stokes_t 
            if dot_tide_t > dot_tide or math.isnan(dot_tide):
                dot_tide= dot_tide_t
    
        # this should seldomly happen, but in case no land is detected in adjacent cells, set to zero
        if math.isnan(dot_wind):
            dot_wind = 0
        if math.isnan(dot_curr):
            dot_curr = 0            
        if math.isnan(dot_stokes):
            dot_stokes = 0
        if math.isnan(dot_tide):
            dot_tide = 0
            
        dtt = particle.dt
        # save the integrated values. In postprocessing the mean is taken per day by dividing by the coastal time
        particle.dot_wind += (dot_wind * dtt)
        particle.dot_curr += (dot_curr * dtt)
        particle.dot_stokes += (dot_stokes * dtt)
        particle.dot_tide += (dot_tide * dtt)
        
        particle.mag_curr += (dtt*math.sqrt(u_curr**2+v_curr**2) )
        particle.mag_wind += (dtt*math.sqrt(u_wind**2+v_wind**2) )
        particle.mag_stokes += (dtt*math.sqrt(u_stokes**2+v_stokes**2) )
        particle.mag_tide += (dtt*math.sqrt(u_tide**2+v_tide**2) )
        particle.mag_tide_h += (dtt*tidal_height )
        particle.mag_VHM0 += (dtt*VHM0_ )
            



def TidalMotionM2S2K1O1(particle, fieldset, time):
    """
    Kernel that calculates tidal currents U and V due to M2, S2, K1 and O1 tide at particle location and time
    and advects the particle in these currents (using Euler forward scheme)
    Calculations based on Doodson (1921) and Schureman (1958)
    """        
    # Number of Julian centuries that have passed between t0 and time
    t = ((time + fieldset.t0rel)/86400.0)/36525.0
    
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
    
    # Euler forward method to advect particle in tidal currents

    lon0, lat0 = (particle.lon, particle.lat)

    # Zonal amplitudes and phaseshifts at particle location and time
    Uampl_M2_1 = f_M2 * fieldset.UaM2[time, particle.depth, lat0, lon0]
    Upha_M2_1 = V_M2 + u_M2 - fieldset.UgM2[time, particle.depth, lat0, lon0]
    Uampl_S2_1 = f_S2 * fieldset.UaS2[time, particle.depth, lat0, lon0]
    Upha_S2_1 = V_S2 + u_S2 - fieldset.UgS2[time, particle.depth, lat0, lon0]
    Uampl_K1_1 = f_K1 * fieldset.UaK1[time, particle.depth, lat0, lon0]
    Upha_K1_1 = V_K1 + u_K1 - fieldset.UgK1[time, particle.depth, lat0, lon0]
    Uampl_O1_1 = f_O1 * fieldset.UaO1[time, particle.depth, lat0, lon0]
    Upha_O1_1 = V_O1 + u_O1 - fieldset.UgO1[time, particle.depth, lat0, lon0]
    # Meridional amplitudes and phaseshifts at particle location and time
    Vampl_M2_1 = f_M2 * fieldset.VaM2[time, particle.depth, lat0, lon0]
    Vpha_M2_1 = V_M2 + u_M2 - fieldset.VgM2[time, particle.depth, lat0, lon0]
    Vampl_S2_1 = f_S2 * fieldset.VaS2[time, particle.depth, lat0, lon0]
    Vpha_S2_1 = V_S2 + u_S2 - fieldset.VgS2[time, particle.depth, lat0, lon0]
    Vampl_K1_1 = f_K1 * fieldset.VaK1[time, particle.depth, lat0, lon0]
    Vpha_K1_1 = V_K1 + u_K1 - fieldset.VgK1[time, particle.depth, lat0, lon0]
    Vampl_O1_1 = f_O1 * fieldset.VaO1[time, particle.depth, lat0, lon0]
    Vpha_O1_1 = V_O1 + u_O1 - fieldset.VgO1[time, particle.depth, lat0, lon0]
    # Zonal and meridional tidal currents; time + fieldset.t0rel = number of seconds elapsed between t0 and time
    Uvel_M2_1 = Uampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Upha_M2_1)
    Uvel_S2_1 = Uampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Upha_S2_1)
    Uvel_K1_1 = Uampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Upha_K1_1)
    Uvel_O1_1 = Uampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Upha_O1_1)
    Vvel_M2_1 = Vampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Vpha_M2_1)
    Vvel_S2_1 = Vampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Vpha_S2_1)
    Vvel_K1_1 = Vampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Vpha_K1_1)
    Vvel_O1_1 = Vampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Vpha_O1_1)
    # Total zonal and meridional velocity
    U1 = Uvel_M2_1 + Uvel_S2_1 + Uvel_K1_1 + Uvel_O1_1 # total zonal velocity
    V1 = Vvel_M2_1 + Vvel_S2_1 + Vvel_K1_1 + Vvel_O1_1 # total meridional velocity
    
    particle.U_tide = U1
    particle.V_tide = V1
    # New lon + lat
    particle.lon += U1*particle.dt
    particle.lat += V1*particle.dt
    
    
def TidalHeightM2S2K1O1(particle, fieldset, time):
    """
    Kernel that calculates tidal currents U and V due to M2, S2, K1 and O1 tide at particle location and time
    and advects the particle in these currents (using Euler forward scheme)
    Calculations based on Doodson (1921) and Schureman (1958)
    """        
    # Number of Julian centuries that have passed between t0 and time
    t = ((time + fieldset.t0rel)/86400.0)/36525.0
    
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
    
    lon0, lat0 = (particle.lon, particle.lat)
    
    amplitude_M2 = f_M2 * fieldset.amp_M2[time, particle.depth, lat0, lon0]
    phase_M2 = V_M2 + u_M2 - fieldset.pha_M2[time, particle.depth, lat0, lon0]
    amplitude_S2 = f_S2 * fieldset.amp_S2[time, particle.depth, lat0, lon0]
    phase_S2 = V_S2 + u_S2 - fieldset.pha_S2[time, particle.depth, lat0, lon0]
    amplitude_K1 = f_K1 * fieldset.amp_K1[time, particle.depth, lat0, lon0]
    phase_K1 = V_K1 + u_K1 - fieldset.pha_K1[time, particle.depth, lat0, lon0]
    amplitude_O1 = f_O1 * fieldset.amp_O1[time, particle.depth, lat0, lon0]
    phase_O1 = V_O1 + u_O1 - fieldset.pha_O1[time, particle.depth, lat0, lon0]    
    
    tide_M2 = amplitude_M2 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + phase_M2)
    tide_S2 = amplitude_S2 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + phase_S2)
    tide_K1 = amplitude_K1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + phase_K1)
    tide_O1 = amplitude_O1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + phase_O1)    

    particle.tidal_height = tide_M2 + tide_S2 + tide_K1 + tide_O1
    

def SamplePerDay(particle, fieldset, time):
    '''
    keep track of min/max of a certain variable per day in the coastal zone
    function looks at the modulus of the time w.r.t. 86400 seconds
    '''
    time_day = time / (60*60*24)    
    day_mod = time_day - math.floor(time_day)

    tidal_height = particle.tidal_height
    
    if day_mod < particle.day_mod_previous: #reset the sampled value when new day has begun
        particle.tidal_height_min = math.nan
        particle.tidal_height_max = math.nan
    
    landZone_p = fieldset.landZone[time, particle.depth, particle.lat, particle.lon] 
    if (landZone_p > fieldset.landzone_threshold):
        if (tidal_height < particle.tidal_height_min) or math.isnan(particle.tidal_height_min):
            particle.tidal_height_min = tidal_height
        if (tidal_height > particle.tidal_height_max) or math.isnan(particle.tidal_height_max):
            particle.tidal_height_max = tidal_height

    particle.day_mod_previous = day_mod

def DeletePerDay(particle,fieldset,time):
    time_day_current = time / (60*60*24)    
    time_day_next = (time + fieldset.parcels_dt ) / (60*60*24)
    
    day_mod_current = time_day_current - math.floor(time_day_current)
    day_mod_next = time_day_next - math.floor(time_day_next)
    
    if day_mod_next < day_mod_current: #reset the sampled value when new day has begun
        if particle.flag_delete == 1:
            particle.delete()

class PlasticParticle(JITParticle):
    age = Variable('age', dtype=np.float32, initial=0., to_write=True)
    
    # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach
    beached = Variable('beached',dtype=np.int32,initial=0., to_write=False)
    
    coastalZoneAge   = Variable('coastalZoneAge', dtype=np.float32, initial=0., to_write=True)
    flag_delete     = Variable('flag_delete', dtype=np.int32, initial=0, to_write=False)
    
    tidal_height = Variable('tidal_height', dtype=np.float32, initial=0., to_write=False)
    U_tide = Variable('U_tide', dtype=np.float32, initial=0., to_write=False)    
    V_tide = Variable('V_tide', dtype=np.float32, initial=0., to_write=False)    
    
    day_mod_previous = Variable('day_mod_previous', dtype=np.float32, initial=0., to_write=False)
    
    tidal_height_min = Variable('tidal_height_min', dtype=np.float32, initial=math.nan, to_write=True)
    tidal_height_max = Variable('tidal_height_max', dtype=np.float32, initial=math.nan, to_write=True)
    
    dot_wind = Variable('dot_wind', dtype=np.float32, initial=0, to_write=True)
    dot_curr = Variable('dot_curr', dtype=np.float32, initial=0, to_write=True)
    dot_stokes = Variable('dot_stokes', dtype=np.float32, initial=0, to_write=True)
    dot_tide = Variable('dot_tide', dtype=np.float32, initial=0, to_write=True)
    
    mag_curr = Variable('mag_curr', dtype=np.float32, initial=0, to_write=True)
    mag_wind = Variable('mag_wind', dtype=np.float32, initial=0, to_write=True)
    mag_stokes = Variable('mag_stokes', dtype=np.float32, initial=0, to_write=True)
    mag_tide = Variable('mag_tide', dtype=np.float32, initial=0, to_write=True)
    mag_tide_h = Variable('mag_tide_h', dtype=np.float32, initial=0, to_write=True)
    mag_VHM0 = Variable('mag_VHM0', dtype=np.float32, initial=0, to_write=True)
    

    
def BoundaryCondition(particle, fieldset, time):
    if particle.lon < -18.666687 or particle.lon > 9.88859 or particle.lat < 40.800003 or particle.lat > 64.266785:
        # particle.delete()
        particle.flag_delete = 1

def Ageing(particle, fieldset, time):
    
    landZone_p = fieldset.landZone[time, particle.depth, particle.lat, particle.lon] 
    
    if (landZone_p > fieldset.landzone_threshold):
        particle.coastalZoneAge += particle.dt
    
    # delete particle when it has 1% mass left in the tau=150 days case
    if particle.coastalZoneAge > (9.1 * 86400): #691.
        # particle.delete()
        particle.flag_delete = 1
        
    particle.age += particle.dt


def StokesUV(particle, fieldset, time):
    (u_uss, v_uss) = fieldset.UV_Stokes[time, particle.depth, particle.lat, particle.lon]

    particle.lon += u_uss * particle.dt
    particle.lat += v_uss * particle.dt
    particle.beached = 3


def DiffusionUniformKh(particle, fieldset, time):
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.

    Assumes that fieldset has constant fields `Kh_zonal` and `Kh_meridional`.
    These can be added via e.g.
        fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)
        fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)
    where mesh is either 'flat' or 'spherical'

    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
    Since the perturbation due to diffusion is in this case isotropic independent, this
    kernel contains no advection and can be used in combination with a seperate
    advection kernel.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
    by = math.sqrt(2 * fieldset.Kh_meridional[particle])

    particle.lon += bx * dWx
    particle.lat += by * dWy
   
    particle.beached = 3    
    
    
def BeachTesting(particle, fieldset, time):
    if particle.beached == 2 or particle.beached == 3:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if u == 0 and v == 0:
            if particle.beached == 2:
                particle.beached = 4
            else:
                particle.beached = 1
        else:
            particle.beached = 0


def UnBeaching(particle, fieldset, time):
    if particle.beached == 4 or particle.beached == 1:
        dtt = particle.dt
        (u_land, v_land) = fieldset.UV_unbeach[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_land * dtt
        particle.lat += v_land * dtt
        particle.beached = 0

#%%

if __name__=="__main__":
    p = ArgumentParser(description="""Particle detection and classification""")
    p.add_argument('-source', '--source', default='p', help='Plastic source type: f r p')
    p.add_argument('-K', '--K', default=10, type=float, help='Eddy diffusivity')
    p.add_argument('-ppday', '--ppday', default=1, type=float, help='particles per day')
    p.add_argument('-outfolder', '--outfolder', default='01_defaultFolder', help='output folder')
    
    
    args = p.parse_args()
    K = args.K
    source = args.source
    particles_per_day = args.ppday
    projectFolder = args.outfolder
    
    parcels_dt = 20 #minutes          
    day_start = datetime(2011,1,1,12,0)
    day_end = datetime(2019,9,30,12,00) 

    date_now = datetime.now()
    output_file = 'particlefile_%s_%4.4i%2.2i%2.2i.nc' % (source,date_now.year,date_now.month,date_now.day)
    
    
    if os.environ['USER'] == 'kaandorp': # desktop
        inputDir = os.path.join(os.environ['HOME'],'Data')
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
    
    
    filenames_re = {'U': files_reanalysis,
                  'V': files_reanalysis}
    variables_re = {'U': 'uo',
                  'V': 'vo'}
    dimensions_re = {'lat': 'latitude',
                  'lon': 'longitude',
                  'time': 'time'}
    
    fieldset_re = FieldSet.from_netcdf(filenames_re, variables_re, dimensions_re)
    
    lons = fieldset_re.U.lon
    lats = fieldset_re.U.lat
    fieldMesh_x,fieldMesh_y = np.meshgrid(lons,lats)
    lon_spacing = lons[1]-lons[0]
    lat_spacing = lats[1]-lats[0]
    fieldset_re.add_constant('lon_spacing',lon_spacing)
    fieldset_re.add_constant('lat_spacing',lat_spacing)
    fieldset_re.add_constant('landzone_threshold',0.00001)
    fieldset_re.add_constant('parcels_dt',int(parcels_dt*60))
    
    threshold_coastal_input = 50
    
    file_landMask = './datafiles/datafile_landMask_%ix_%iy' % (len(lons),len(lats))
    landMask = get_landmask(file_0['uo'],outfile=file_landMask)
    file_trueLandMask = './datafiles/datafile_trueLandMask_%ix_%iy' % (len(lons),len(lats))
    trueLandMask = get_true_landmask(landMask,outfile=file_trueLandMask)
      
    file_coastMask = './datafiles/datafile_coastMask_%ix_%iy' % (len(lons),len(lats))
    coastMask = get_coastmask(landMask,i_top = 364,i_bot = 10,i_left = 10,i_right = 269,outfile=file_coastMask)
    
    
    #---------------Stokes
    files_Stokes = sorted(glob.glob(os.path.join(inputDir , 'CMEMS/GLOBAL_REANALYSIS_WAV_001_032_NWSHELF_Stokes/*')))
    filenames_Stokes = {'U_stokes': files_Stokes,
                  'V_stokes': files_Stokes}
    variables_Stokes = {'U_stokes': 'VSDX',
                  'V_stokes': 'VSDY'}
    dimensions_Stokes = {'lat': 'latitude',
                  'lon': 'longitude',
                  'time': 'time'}
    
    fieldset_Stokes = FieldSet.from_netcdf(filenames_Stokes, variables_Stokes, dimensions_Stokes,mesh='spherical')
    fieldset_Stokes.U_stokes.units = GeographicPolar()
    fieldset_Stokes.V_stokes.units = Geographic()
    
    fieldset_re.add_field(fieldset_Stokes.U_stokes)
    fieldset_re.add_field(fieldset_Stokes.V_stokes)
    
    vectorField_Stokes = VectorField('UV_Stokes',fieldset_re.U_stokes,fieldset_re.V_stokes)
    fieldset_re.add_vector_field(vectorField_Stokes)
    
    #---------------VHM0
    files_VHM0 = sorted(glob.glob(os.path.join(inputDir , 'CMEMS/GLOBAL_REANALYSIS_WAV_001_032_NWSHELF_VHM0/*')))
    filenames_VHM0 = {'VHM0': files_VHM0}
    variables_VHM0 = {'VHM0': 'VHM0'}
    dimensions_VHM0 = {'lat': 'latitude',
                  'lon': 'longitude',
                  'time': 'time'}
    
    fieldset_VHM0 = FieldSet.from_netcdf(filenames_VHM0, variables_VHM0, dimensions_VHM0,mesh='spherical')
    fieldset_re.add_field(fieldset_VHM0.VHM0)

        
    #---------------Wind
    files_wind = sorted(glob.glob(os.path.join(inputDir , 'ERA5/NWSHELF_Wind/wind*')))
    filenames_wind = {'U_wind': files_wind,
                  'V_wind': files_wind}
    variables_wind = {'U_wind': 'u10',
                  'V_wind': 'v10'}
    dimensions_wind = {'lat': 'latitude',
                  'lon': 'longitude',
                  'time': 'time'}
    
    fieldset_wind = FieldSet.from_netcdf(filenames_wind, variables_wind, dimensions_wind,mesh='spherical')
    fieldset_wind.U_wind.units = GeographicPolar()
    fieldset_wind.V_wind.units = Geographic()
    
    fieldset_re.add_field(fieldset_wind.U_wind)
    fieldset_re.add_field(fieldset_wind.V_wind)
    
    vectorField_wind = VectorField('UV_wind',fieldset_re.U_wind,fieldset_re.V_wind)
    fieldset_re.add_vector_field(vectorField_wind)    
    
    
    #---------------unbeaching
    file_landCurrent_U = './datafiles/datafile_landCurrentU_%ix_%iy' % (len(lons),len(lats))
    file_landCurrent_V = './datafiles/datafile_landCurrentV_%ix_%iy' % (len(lons),len(lats))
    landCurrent_U,landCurrent_V = calculateLandCurrent(landMask,fieldMesh_x,fieldMesh_y,i_top = 364,i_bot = 10,
                              i_left = 10,i_right = 269,do_plot=False,outfile_1=file_landCurrent_U,outfile_2=file_landCurrent_V)
    U_land = Field('U_land',landCurrent_U,lon=lons,lat=lats,fieldtype='U',mesh='spherical')
    V_land = Field('V_land',landCurrent_V,lon=lons,lat=lats,fieldtype='V',mesh='spherical')
    
    fieldset_re.add_field(U_land)
    fieldset_re.add_field(V_land)
    
    vectorField_unbeach = VectorField('UV_unbeach',U_land,V_land)
    fieldset_re.add_vector_field(vectorField_unbeach)
    
    
    #-----------------misc fields
    K_m = K*np.ones(fieldMesh_x.shape)
    K_z = K*np.ones(fieldMesh_x.shape)
    
    
    Kh_meridional = Field('Kh_meridional', K_m,lon=lons,lat=lats,mesh='spherical')
    Kh_zonal = Field('Kh_zonal', K_z,lon=lons,lat=lats,mesh='spherical')
    coastalZone = Field('coastalZone',coastMask,lon=lons,lat=lats,mesh='spherical')
    landZone = Field('landZone',trueLandMask,lon=lons,lat=lats,mesh='spherical')
    land_nearest = Field('land_nearest',trueLandMask,lon=lons,lat=lats,mesh='spherical')

    fieldset_re.add_field(Kh_meridional)
    fieldset_re.add_field(Kh_zonal)
    fieldset_re.add_field(coastalZone)
    fieldset_re.add_field(landZone)
    fieldset_re.add_field(land_nearest)
    fieldset_re.land_nearest.interp_method = 'nearest'
    
    #----------------Input scenarios
    file_coastalInput = './datafiles/datafile_populationInputMatrices_thres%i_%ix_%iy' % (threshold_coastal_input,len(lons),len(lats) )
    file_coastalZone = './datafiles/datafile_coastalLandMask_thres%i_%ix_%iy' % (threshold_coastal_input,len(lons),len(lats) )
    populationInputMatrices,coastalLandMask = initializePopulationMatrices(landMask,coastMask,fieldMesh_x,fieldMesh_y,
                                                                            outfile_popmat=file_coastalInput,outfile_coastmat=file_coastalZone)
    
    file_riverInput = './datafiles/datafile_riverInputMatrices_%ix_%iy' % (len(lons),len(lats) )
    rivers_shapefile    = os.path.join(inputDir , 'PlasticData/PlasticRiverInputs_Lebreton/PlasticRiverInputs.shp')
    rivers_waste        = os.path.join(inputDir , 'PlasticData/PlasticRiverInputs_Lebreton/PlasticRiverInputs.csv')
    riverInputMatrices  = initializeRiverMatrices(rivers_shapefile, rivers_waste, coastMask, 
                                                  fieldMesh_x, fieldMesh_y, selection_estim='mid',outfile=file_riverInput)
    
    file_fishingInput = './datafiles/datafile_fishingInputMatrices_%ix_%iy' % (len(lons),len(lats) )
    fishingInputMatrix = initializeFisheriesMatrices(landMask,lons,lats,outfile=file_fishingInput)
    
    
    #-----------------tides--------------
    t0 = datetime(1900,1,1,0,0) # origin of time = 1 January 1900, 00:00:00 UTC
    fieldset_re.add_constant('t0rel', (day_start - t0).total_seconds()) # number of seconds elapsed between t0 and starttime
    
    """ ----- Creating the tidal Fields ----- """
    
    files_eastward = os.path.join(inputDir , 'FES2014Data/eastward_velocity/')
    files_northward = os.path.join(inputDir , 'FES2014Data/northward_velocity/')
    files_tideh = os.path.join(inputDir , 'FES2014Data/ocean_tide/')
    
    deg2rad = math.pi/180.0 # factor to convert degrees to radians
    
    
    def create_fieldset_tidal_currents(name,filename):
        '''
        Create fieldset for a given type of tide (name = M2, S2, K1, or O1)

        '''
        filename_U = files_eastward + '%s.nc' %filename
        filename_V = files_northward + '%s.nc' %filename
           
        filenames = {'Ua%s'%name: filename_U,
                     'Ug%s'%name: filename_U,
                     'Va%s'%name: filename_V,
                     'Vg%s'%name: filename_V}
        variables = {'Ua%s'%name: 'Ua',
                     'Ug%s'%name: 'Ug',
                     'Va%s'%name: 'Va',
                     'Vg%s'%name: 'Vg'}
        dimensions = {'lat': 'lat',
                      'lon': 'lon'}
        
        fieldset_tmp = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='spherical')
        
        exec('fieldset_tmp.Ua%s.set_scaling_factor(1e-2)'%name)
        exec('fieldset_tmp.Ug%s.set_scaling_factor(deg2rad)'%name) # convert from degrees to radians
        exec('fieldset_tmp.Va%s.set_scaling_factor(1e-2)'%name) #cm/s to m/s
        exec('fieldset_tmp.Vg%s.set_scaling_factor(deg2rad)'%name)
        
        exec('fieldset_tmp.Ua%s.units = GeographicPolar()'%name)
        exec('fieldset_tmp.Va%s.units = Geographic()'%name)
        
        exec('fieldset_re.add_field(fieldset_tmp.Ua%s)'%name)
        exec('fieldset_re.add_field(fieldset_tmp.Ug%s)'%name)
        exec('fieldset_re.add_field(fieldset_tmp.Va%s)'%name)
        exec('fieldset_re.add_field(fieldset_tmp.Vg%s)'%name)
            
    def create_fieldset_tidal_height(name,filename):
        '''
        Create fieldset for a given type of tide (name = M2, S2, K1, or O1)

        '''
        filename = files_tideh + '%s.nc' %filename
           
        filenames = {'amp_%s'%name: filename,
                     'pha_%s'%name: filename}
        variables = {'amp_%s'%name: 'amplitude',
                     'pha_%s'%name: 'phase'}
        dimensions = {'lat': 'lat',
                      'lon': 'lon'}
        
        fieldset_tmp = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='spherical')
        
        exec('fieldset_tmp.amp_%s.set_scaling_factor(1e-2)'%name) #cm/s to m/s
        exec('fieldset_tmp.pha_%s.set_scaling_factor(deg2rad)'%name) # convert from degrees to radians
        
        exec('fieldset_re.add_field(fieldset_tmp.amp_%s)'%name)
        exec('fieldset_re.add_field(fieldset_tmp.pha_%s)'%name)

          
    create_fieldset_tidal_currents('M2','conv_m2')
    create_fieldset_tidal_currents('S2','conv_s2')
    create_fieldset_tidal_currents('K1','conv_k1')
    create_fieldset_tidal_currents('O1','conv_o1')
    
    create_fieldset_tidal_height('M2','conv_m2')
    create_fieldset_tidal_height('S2','conv_s2')
    create_fieldset_tidal_height('K1','conv_k1')
    create_fieldset_tidal_height('O1','conv_o1')
    
    omega_M2 = 28.9841042 # angular frequency of M2 in degrees per hour
    fieldset_re.add_constant('omegaM2', (omega_M2 * deg2rad) / 3600.0) # angular frequency of M2 in radians per second
    
    omega_S2 = 30.0000000 # angular frequency of S2 in degrees per hour
    fieldset_re.add_constant('omegaS2', (omega_S2 * deg2rad) / 3600.0) # angular frequency of S2 in radians per second
    
    omega_K1 = 15.0410686 # angular frequency of K1 in degrees per hour
    fieldset_re.add_constant('omegaK1', (omega_K1 * deg2rad) / 3600.0) # angular frequency of K1 in radians per second
    
    omega_O1 = 13.9430356 # angular frequency of O1 in degrees per hour
    fieldset_re.add_constant('omegaO1', (omega_O1 * deg2rad) / 3600.0) # angular frequency of O1 in radians per second

    #%%
    
    n_days = (day_end-day_start).days
    releaseTimes = np.arange(0, n_days) * timedelta(hours=24)
    releaseTimes += day_start 
    
    
    particles_lat   = np.array([])
    particles_lon   = np.array([])
    particles_t     = np.array([])
    
    
    for i1,time in enumerate(releaseTimes):
        
        # get monthly waste [tonnes] at given time
        input_matrix,monthly_fraction = get_input_matrix(source,time) #monthly_fraction: 1 for pop&fishing, assuming constant input
        n_particles = particles_per_day*monthly_fraction #*correction_ndays
    
        # convert n_particles, which is a float, to integer amount of particles, where the fraction
        # is converted to 0 or 1 depending on its magnitude  
        total_particles_base = np.floor(n_particles)
        chance_0 = 1-(n_particles-total_particles_base) # chances of adding 0/1
        chance_1 = n_particles-total_particles_base
        n_particles_int = int(total_particles_base + np.random.choice([0,1],size=1,p=[chance_0,chance_1])[0])
        
        
        prob_array = input_matrix[input_matrix>0] #array with probabilities of drawing a particle
    
        #total_particles ~ total waste generated at certain time
        for i2 in range(n_particles_int):
    
            drawParticle = np.random.rand(len(prob_array)) < prob_array # adapted from  https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s
            lats_draw = fieldMesh_y[input_matrix>0][drawParticle]
            lons_draw = fieldMesh_x[input_matrix>0][drawParticle]
    
            particles_lat = np.append(particles_lat,lats_draw)
            particles_lon = np.append(particles_lon,lons_draw)
            particles_t = np.append(particles_t,np.array([(time-day_start).total_seconds() for i in range(len(lons_draw))]))   
    
    time_origin = datetime.utcfromtimestamp(np.datetime64(str(fieldset_re.time_origin)).astype(int)*1e-9)
    time_start = day_start - time_origin
    particles_t += time_start.total_seconds()
        
    file_particle_info = os.path.join(outDir,'particlesReleaseInfo.txt')
    writeParticleInfo(file_particle_info,particles_t,particles_lat,particles_lon,t0=time_origin)
        
    
    #%%
    
    n_days_re = (day_end - day_start).days
    
    pset1 = ParticleSet.from_list(fieldset_re, PlasticParticle, particles_lon,
                                  particles_lat,time=particles_t)
    
    
    
    
    kernel = (pset1.Kernel(DeletePerDay) + pset1.Kernel(AdvectionRK4) + pset1.Kernel(StokesUV) +
              pset1.Kernel(BeachTesting) + pset1.Kernel(UnBeaching) +
              pset1.Kernel(TidalMotionM2S2K1O1) + pset1.Kernel(DiffusionUniformKh) + 
              pset1.Kernel(BoundaryCondition) + pset1.Kernel(Ageing) + 
              pset1.Kernel(BeachTesting) + pset1.Kernel(UnBeaching) + 
              pset1.Kernel(TidalHeightM2S2K1O1) +
              pset1.Kernel(SamplePerDay) + pset1.Kernel(coastalDynamics) )
    
    
    tmp_filename = os.path.join(outDir,output_file)
    append_int = 1
    while os.path.exists(tmp_filename):
        tmp_filename = os.path.join(outDir, output_file[:-3] + '_%i.nc'%append_int )
        append_int += 1
    output_filename = tmp_filename

    print('Running parcels from %s to %s, \n output: %s' %(str(day_start),str(day_end),output_filename))
    print('Settings: K %f, source %s, particles per day %f,' %(K,source,particles_per_day))
    
    pset1.execute(kernel,
                runtime=timedelta(days=n_days),  # runtime controls the interval of the plots
                dt=timedelta(minutes=parcels_dt),
                output_file=pset1.ParticleFile(name=output_filename, outputdt=timedelta(hours=24)) )
    print('Run done')
    
    
    
    #%%

    data_traj = xr.open_dataset(output_filename)
    
    landMask_plot = get_true_landMask(landMask)
    dlon = lons[1]-lons[0]
    dlat = lats[1]-lats[0]
    lons_edges = lons-.5*dlon
    lons_edges = np.append(lons_edges,lons_edges[-1]+dlon)
    lats_edges = lats-.5*dlat
    lats_edges = np.append(lats_edges,lats_edges[-1]+dlat)
    meshPlotx,meshPloty = np.meshgrid(lons_edges,lats_edges)
    
    plt.figure(figsize=(8,8))     
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(meshPlotx,meshPloty,landMask_plot,cmap=cmocean.cm.topo,
                  vmin=-.3,vmax=1.15,transform=ccrs.PlateCarree(),alpha=.7)
    for i in range(min(50,len(data_traj['traj']))):
        i1 = np.random.randint(0,data_traj['lon'].shape[0])
        ax.plot(data_traj['lon'][i1,:],data_traj['lat'][i1,:],'o-',transform=ccrs.PlateCarree())
