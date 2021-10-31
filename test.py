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

valid_data,N_val,E_test,unit_list_val = getdata(200,'/home/thales/DSAKT/data_set/ass09_origin/assist09_test.csv','sakt')
train_data,N_train,E_train,unit_list_train = getdata(window_size=200, path='/home/thales/DSAKT/data_set/ass09_origin/assist09_train.csv', model_type='sakt')


train_data_test = train_data_test.permute(1,0,2)
valid_data_test = valid_data_test.permute(1,0,2)
'''
print(train_data)
train_loader = dataloader(train_data,100,True)
print(train_loader.shape)

print(valid_data.shape)
print(N_val)

print(valid_data_test.shape[1])
'''

#print(unit_list_val)

'''
print(unit_list_train[:10])
print(valid_data.shape)
print(train_data[0][0])
print(train_data[1][0])
print(train_data[2][0])
'''

unit_train = []
count = 0
for i in range(train_data.shape[1]):
    for j in range(train_data.shape[2]):
        if train_data[0][i][j] !=0:
            count = count +1
    unit_train.append(count)
    count = 0


print(unit_train[:10])

#TODO: enteder o que está acontecendo dentro do train loader
#mudar o tipo para long ao invés de float
#tentar treinar novamente no colab