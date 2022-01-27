#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:28:29 2021

@author: kaandorp
"""
import numpy as np
import os
from datetime import datetime
import pandas as pd
import math
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature, LAND

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
    # nonlinear, see Guo et al. (2019) JGR oceans
    omega_M4 = omega_M2 + omega_M2 # angular frequency of M4 in degrees per hour
    omega_M6 = omega_M2 + omega_M4 # angular frequency of M6 in degrees per hour
    
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
    #nonlinear
    V_M4 = 4*T0 - 4*s0 + 4*h0
    V_M6 = 6*T0 - 6*s0 + 6*h0
        
    
    tide_land_mask = np.isnan(data_K1['phase'])[i_lat_min:i_lat_max,i_lon_min:i_lon_max]
    tide_array = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    tide_array_U = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    tide_array_V = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])

    # tide_array_M2 = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    # tide_array_M4 = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    # tide_array_M6 = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    # tide_array_K1 = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    # tide_array_O1 = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    # tide_array_S2 = np.zeros([len(time_array),len(lats[i_lat_min:i_lat_max]),len(lons[i_lon_min:i_lon_max])])
    
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
        #nonlinear
        u_M4 = 4*xi - 4*nu
        f_M4 = (f_M2)**2
        u_M6 = 6*xi - 6*nu
        f_M6 = (f_M2)**3        
        
        #tidal height
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

        ampl_M4 = f_M4 * data_M4['amplitude'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M4 = V_M4 + u_M4 - data_M4['phase'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M4 = ampl_M4 * np.cos(omega_M4 * (time+t0rel) + pha_M4)

        ampl_M6 = f_M6 * data_M6['amplitude'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M6 = V_M6 + u_M6 - data_M6['phase'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M6 = ampl_M6 * np.cos(omega_M6 * (time+t0rel) + pha_M6)

        #u_velocity
        ampl_K1_U = f_K1 * data_K1_U['Ua'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_K1_U = V_K1 + u_K1 - data_K1_U['Ug'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_K1_U = ampl_K1_U * np.cos(omega_K1 * (time+t0rel) + pha_K1_U)
        
        ampl_M2_U = f_M2 * data_M2_U['Ua'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M2_U = V_M2 + u_M2 - data_M2_U['Ug'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M2_U = ampl_M2_U * np.cos(omega_M2 * (time+t0rel) + pha_M2_U)
        
        ampl_O1_U = f_O1 * data_O1_U['Ua'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_O1_U = V_O1 + u_O1 - data_O1_U['Ug'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_O1_U = ampl_O1_U * np.cos(omega_O1 * (time+t0rel) + pha_O1_U)
        
        ampl_S2_U = f_S2 * data_S2_U['Ua'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_S2_U = V_S2 + u_S2 - data_S2_U['Ug'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_S2_U = ampl_S2_U * np.cos(omega_S2 * (time+t0rel) + pha_S2_U)

        ampl_M4_U = f_M4 * data_M4_U['Ua'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M4_U = V_M4 + u_M4 - data_M4_U['Ug'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M4_U = ampl_M4_U * np.cos(omega_M4 * (time+t0rel) + pha_M4_U)

        ampl_M6_U = f_M6 * data_M6_U['Ua'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M6_U = V_M6 + u_M6 - data_M6_U['Ug'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M6_U = ampl_M6_U * np.cos(omega_M6 * (time+t0rel) + pha_M6_U)
        
        
        #v_velocity
        ampl_K1_V = f_K1 * data_K1_V['Va'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_K1_V = V_K1 + u_K1 - data_K1_V['Vg'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_K1_V = ampl_K1_V * np.cos(omega_K1 * (time+t0rel) + pha_K1_V)
        
        ampl_M2_V = f_M2 * data_M2_V['Va'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M2_V = V_M2 + u_M2 - data_M2_V['Vg'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M2_V = ampl_M2_V * np.cos(omega_M2 * (time+t0rel) + pha_M2_V)
        
        ampl_O1_V = f_O1 * data_O1_V['Va'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_O1_V = V_O1 + u_O1 - data_O1_V['Vg'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_O1_V = ampl_O1_V * np.cos(omega_O1 * (time+t0rel) + pha_O1_V)
        
        ampl_S2_V = f_S2 * data_S2_V['Va'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_S2_V = V_S2 + u_S2 - data_S2_V['Vg'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_S2_V = ampl_S2_V * np.cos(omega_S2 * (time+t0rel) + pha_S2_V)

        ampl_M4_V = f_M4 * data_M4_V['Va'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M4_V = V_M4 + u_M4 - data_M4_V['Vg'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M4_V = ampl_M4_V * np.cos(omega_M4 * (time+t0rel) + pha_M4_V)

        ampl_M6_V = f_M6 * data_M6_V['Va'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] *1e-2
        pha_M6_V = V_M6 + u_M6 - data_M6_V['Vg'][i_lat_min:i_lat_max,i_lon_min:i_lon_max] * deg2rad
        tide_M6_V = ampl_M6_V * np.cos(omega_M6 * (time+t0rel) + pha_M6_V)
                 
        
        
        tide_tot = tide_K1 + tide_M2 + tide_O1 + tide_S2 + tide_M4 + tide_M6
        tide_array[i1,:,:] = tide_tot
        # tide_array_K1[i1,:,:] = tide_K1
        # tide_array_M2[i1,:,:] = tide_M2
        # tide_array_O1[i1,:,:] = tide_O1
        # tide_array_S2[i1,:,:] = tide_S2
        # tide_array_M4[i1,:,:] = tide_M4
        # tide_array_M6[i1,:,:] = tide_M6
        
        U_tot = tide_K1_U + tide_M2_U + tide_O1_U + tide_S2_U + tide_M4_U + tide_M6_U
        V_tot = tide_K1_V + tide_M2_V + tide_O1_V + tide_S2_V + tide_M4_V + tide_M6_V
        tide_array_U[i1,:,:] = U_tot
        tide_array_V[i1,:,:] = V_tot
        
    # ds = xr.Dataset(
    #     {"tide": (("time", "lat", "lon"), tide_array ),
    #      "tide_K1": (("time", "lat", "lon"), tide_array_K1 ),
    #      "tide_M2": (("time", "lat", "lon"), tide_array_M2 ),
    #      "tide_O1": (("time", "lat", "lon"), tide_array_O1 ),
    #      "tide_S2": (("time", "lat", "lon"), tide_array_S2 ),
    #      "tide_M4": (("time", "lat", "lon"), tide_array_M4 ),
    #      "tide_M6": (("time", "lat", "lon"), tide_array_M6 ),
    #      "mask_land": (("lat", "lon"), tide_land_mask ),
    #      "explanation": 'tides calculated from FES dataset'},
    #     coords={
    #         "lon": lons[i_lon_min:i_lon_max],
    #         "lat": lats[i_lat_min:i_lat_max],
    #         "time": time_array,
    #     },
    # )   
        
    ds = xr.Dataset(
        {"tide": (("time", "lat", "lon"), tide_array ),
         "tide_U": (("time", "lat", "lon"), tide_array_U ),
         "tide_V": (("time", "lat", "lon"), tide_array_V ),
         "mask_land": (("lat", "lon"), tide_land_mask ),
         "explanation": 'tides calculated from FES dataset'},
        coords={
            "lon": lons[i_lon_min:i_lon_max],
            "lat": lats[i_lat_min:i_lat_max],
            "time": time_array,
        },
    )     
    return ds, X_tides, Y_tides, tide_land_mask




tides_folder = '/Users/kaandorp/Data/FES2014Data/ocean_tide/'
u_folder = '/Users/kaandorp/Data/FES2014Data/eastward_velocity/'
v_folder = '/Users/kaandorp/Data/FES2014Data/northward_velocity/'

file_K1 = os.path.join(tides_folder,'conv_k1.nc')
file_M2 = os.path.join(tides_folder,'conv_m2.nc')
file_O1 = os.path.join(tides_folder,'conv_o1.nc')
file_S2 = os.path.join(tides_folder,'conv_s2.nc')
file_M4 = os.path.join(tides_folder,'conv_m4.nc')
file_M6 = os.path.join(tides_folder,'conv_m6.nc')

data_K1 = xr.open_dataset(file_K1)
data_M2 = xr.open_dataset(file_M2)
data_O1 = xr.open_dataset(file_O1)
data_S2 = xr.open_dataset(file_S2)
data_M4 = xr.open_dataset(file_M4)
data_M6 = xr.open_dataset(file_M6)


file_K1_U = os.path.join(u_folder,'conv_k1.nc')
file_M2_U = os.path.join(u_folder,'conv_m2.nc')
file_O1_U = os.path.join(u_folder,'conv_o1.nc')
file_S2_U = os.path.join(u_folder,'conv_s2.nc')
file_M4_U = os.path.join(u_folder,'conv_m4.nc')
file_M6_U = os.path.join(u_folder,'conv_m6.nc')

data_K1_U = xr.open_dataset(file_K1_U)
data_M2_U = xr.open_dataset(file_M2_U)
data_O1_U = xr.open_dataset(file_O1_U)
data_S2_U = xr.open_dataset(file_S2_U)
data_M4_U = xr.open_dataset(file_M4_U)
data_M6_U = xr.open_dataset(file_M6_U)


file_K1_V = os.path.join(v_folder,'conv_k1.nc')
file_M2_V = os.path.join(v_folder,'conv_m2.nc')
file_O1_V = os.path.join(v_folder,'conv_o1.nc')
file_S2_V = os.path.join(v_folder,'conv_s2.nc')
file_M4_V = os.path.join(v_folder,'conv_m4.nc')
file_M6_V = os.path.join(v_folder,'conv_m6.nc')

data_K1_V = xr.open_dataset(file_K1_V)
data_M2_V = xr.open_dataset(file_M2_V)
data_O1_V = xr.open_dataset(file_O1_V)
data_S2_V = xr.open_dataset(file_S2_V)
data_M4_V = xr.open_dataset(file_M4_V)
data_M6_V = xr.open_dataset(file_M6_V)

years = np.arange(2014,2020)

for year_ in years:
   
    ds,X,Y,mask = initialize_tides(year_)
    # year_ = 2014
    ds.to_netcdf('datafiles/tides_%i.nc' % year_)

# components = ['K1','M2','O1','S2','M4','M6']
# plt.figure()
# for c_ in components:
#     plt.plot(ds['time'][0:100],ds['tide_%s' % c_][0:100,50,40],label=c_)
# plt.plot(ds['time'][0:100],ds['tide'][0:100,50,40],label='total')
# plt.legend()

# for i_ in range(24):
#     e_ = 4
#     fig = plt.figure(figsize=(8,8),dpi=120)     
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     # ax.set_extent((0.5,5.45,49,53.5))
#     ax.add_feature(LAND, zorder=0,edgecolor='black')
#     # plt.figure(figsize=(10,10))
#     ax.quiver(X[::e_,::e_],Y[::e_,::e_],ds['tide_U'][i_,::e_,::e_],ds['tide_V'][i_,::e_,::e_])
#     fig.savefig('tide_%i.png' % i_)
#     plt.close('all')