from numpy.core.records import array
from utils import getdata
import torch
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd

from utils import dataloader
from sklearn.model_selection import train_test_split
from utils import get_data



#train_data,N_train,E_train,unit_list_train = getdata(window_size=50, path='/home/thales/DSAKT/data_set/ass09_origin/assist09_train.csv', model_type='sakt')


data,E = get_data('/home/thales/DSAKT/errex data subproblems.csv',350)

train_data_test,valid_data_test = train_test_split(data.permute(1,0,2),test_size = 0.2)

#valid_data,N_val,E_test,unit_list_val = getdata(200,'/home/thales/DSAKT/data_set/ass09_origin/assist09_test.csv','sakt')
train_data,N_train,E_train,unit_list_train = getdata(window_size=200, path='/home/thales/DSAKT/data_set/ass09_origin/assist09_train.csv', model_type='sakt')


train_data_test = train_data_test.permute(1,0,2)
valid_data_test = valid_data_test.permute(1,0,2)

'''
print(data[0][1])
print(data[1][1])
print(data[2][1])
'''