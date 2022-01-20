from pyBKT.models import Model
import pyBKT
import pandas as pd


df = pd.read_csv('/home/thales/DSAKT/errex data subproblems.csv')
model = Model(seed = 42,num_fits = 1)
defaults = {'skill_name':}