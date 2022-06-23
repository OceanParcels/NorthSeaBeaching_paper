#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:56:19 2022

@author: kaandorp
"""

import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


filename = 'demo.pickle'
with open(filename, 'rb') as f:
    regression_table = pickle.load(f)

pdb.set_trace()

x = regression_table.iloc[:,5:]
y = np.log10(regression_table.iloc[:,3])

pdb.set_trace()

regressor = RandomForestRegressor(max_features=.33)
use_PCA = True

train_test_folds = KFold(n_splits=5,shuffle=True)

fig,ax = plt.subplots(1,figsize=(5,5))
ax.plot([.5,2.5],[.5,2.5],'k--')
ax.set_xlabel('Predicted value')
ax.set_ylabel('True value')

y_test_total = np.array([])
y_predict_total = np.array([])

for i1, (i_train,i_test) in enumerate(train_test_folds.split(x)):
    
    x_train = x.iloc[i_train,:]
    y_train = y.iloc[i_train]
    x_test = x.iloc[i_test,:]
    y_test = y.iloc[i_test]
    
    
    if use_PCA:
        pca = PCA(.95)
        pca.fit(x_train)
        
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        
        if i1 == 0:
            pdb.set_trace()
        
    regressor.fit(x_train,y_train)
    
    y_predict = regressor.predict(x_test)
    
    ax.plot(y_predict,y_test,'o')
    
    y_predict_total = np.append(y_predict_total,y_predict)
    y_test_total = np.append(y_test_total,y_test)
        
        
ax.set_title('Total Pearson R: %2.2f' % pearsonr(y_predict_total,y_test_total)[0])