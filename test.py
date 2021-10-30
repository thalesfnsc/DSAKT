from numpy.core.records import array
from utils import getdata
import torch
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def get_data(data_path,max_sequence_size):
    
    array_responses = []
    array_problems = []
    array_problems_ahead = []
    array_responses_ahead = []
    #Loading data in a pandas DataFrame

    df = pd.read_csv(data_path)

    problems_ids = df['problem_id'].unique()
    E = len(problems_ids)
    
    users_data = df.groupby('student_id')[['problem_id','condition','skill_name','correct']].agg(lambda x:list(x))
    index_to_id = np.unique([*itertools.chain.from_iterable(users_data['problem_id'])])
    id_to_index = {index_to_id[i]: i for i in range(len(index_to_id))}

    #print(id_to_index)
    sequence_sizes = []

    for index,student in users_data.iterrows():
        sequence_size = len(student['problem_id'])
        sequence_sizes.append(sequence_size)
        student_problem_id = [id_to_index[i] for i in student['problem_id']]

        '''
        if sequence_size > max_sequence_size:
            for i in range(max_sequence_size,sequence_size):
                array_responses.append(student['correct'][(i - max_sequence_size):i])
                array_problems.append(student_problem_id[(i - max_sequence_size):i])

        else:
        '''    
        len_responses = len(student['correct'])
        len_exercises = len(student_problem_id)

        array_responses.append(student['correct'][:len_responses -1] + [0] * ((max_sequence_size+1) - sequence_size)) 
        array_problems.append(student_problem_id[:len_exercises -1] + [0] * ((max_sequence_size+1) - sequence_size))
        array_problems_ahead.append(student_problem_id[1:] + [0]*((max_sequence_size+1) - sequence_size))
        array_responses_ahead.append(student['correct'][1:] + [0] * ((max_sequence_size+1) -sequence_size ))

    
    problems = torch.Tensor(array_problems)
    problems_ahead = torch.Tensor(array_problems_ahead)
    
    responses = torch.Tensor(array_responses)
    responses_ahead = torch.Tensor(array_responses_ahead)

    interaction = problems + E*responses

    
    data = torch.stack((interaction,problems_ahead,responses_ahead))
    train,val = train_test_split(data.permute(1,0,2),test_size = 0.2)

    
    return  train,val




train_data,N_train,E_train,unit_list_train = getdata(window_size=50, path='/home/thales/DSAKT/data_set/ass09_origin/assist09_train.csv', model_type='sakt')


data = get_data('/home/thales/DSAKT/errex data subproblems.csv',350)



train,test = train_test_split(data.permute(1,0,2),test_size = 0.2,)

print(train.shape)
print(test.shape)

'''
print(data[0][0])
print(data[1][0])
print(data[2][0])
'''
print(len(unit_list_train))
print(train_data.shape)
print(N_train)