from numpy.core.records import array
from utils import getdata
import torch
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
from predict import predict
from utils import dataloader
from sklearn.model_selection import train_test_split
from utils import get_data



#train_data,N_train,E_train,unit_list_train = getdata(window_size=50, path='/home/thales/DSAKT/data_set/ass09_origin/assist09_train.csv', model_type='sakt')

window_size = 350

data,E = get_data('/home/thales/DSAKT/errex_dropped.csv',window_size)

model_path = '/home/thales/DSAKT/save_SAKT_300_epochs_0_916_AUC.pth'

model = torch.load(model_path);
assert model.window_size == window_size;
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
model.to(device);
model.eval();

