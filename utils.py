import torch
import random
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import numpy as np

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


    for index,student in users_data.iterrows():
        sequence_size = len(student['problem_id'])
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
    problems = torch.IntTensor(array_problems).type(torch.int64)
    problems_ahead = torch.IntTensor(array_problems_ahead).type(torch.int64)
    responses = torch.IntTensor(array_responses).type(torch.int64)
    responses_ahead = torch.IntTensor(array_responses_ahead).type(torch.int64)
    interaction = problems + E*responses
    
    data = torch.stack((interaction,problems_ahead,responses_ahead))

    return  data,E



def get_data_predict(data_path,max_sequence_size):
    
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


    for index,student in users_data.iterrows():
        sequence_size = len(student['problem_id'])
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
        array_responses.append(student['correct'][:len_responses] + [0] * ((max_sequence_size+1) - sequence_size)) 
        array_problems.append(student_problem_id[:len_exercises ] + [0] * ((max_sequence_size+1) - sequence_size))
        array_problems_ahead.append(student_problem_id[1:] + [0]*((max_sequence_size+1) - sequence_size))
        array_responses_ahead.append(student['correct'][1:] + [0] * ((max_sequence_size+1) -sequence_size ))
    problems = torch.IntTensor(array_problems).type(torch.int64)
    problems_ahead = torch.IntTensor(array_problems_ahead).type(torch.int64)
    responses = torch.IntTensor(array_responses).type(torch.int64)
    responses_ahead = torch.IntTensor(array_responses_ahead).type(torch.int64)
    interaction = problems + E*responses
    
    data = torch.stack((interaction,problems,responses))

    return  data,E

def getdata(window_size,path,model_type,drop=False):
    '''
    @param model_type: 'sakt' or 'saint'
    '''
    N=0
    count=0
    E=-1
    units=[]
    input_1=[]
    input_2=[]
    input_3=[]
    input_4=[]
    bis=0
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if count%3==0:
            pass
        elif count%3==1:
            tlst=line.split('\n')[0].split(',')

            for item in tlst:
                if int(item) >E:
                    E=int(item)

            tlst_1=tlst[0:len(tlst)-1]
            tlst_2=tlst[1:len(tlst)]
            

            if drop:
                if len(tlst_1)>window_size:
                    tlst_1=tlst_1[0:window_size]
                    tlst_2=tlst_2[0:window_size]

            while len(tlst_1)>window_size:
                input_1.append([int(i)+1 for i in tlst_1[0:window_size]])
                N+=1
                tlst_1= tlst_1[window_size:len(tlst_1)]
                units.append(window_size)
            units.append(len(tlst_1))
            tlst_1=[int(i)+1 for i in tlst_1]+[0]*(window_size - len(tlst_1))
            N+=1
            input_1.append(tlst_1)

            while len(tlst_2)>window_size:
                input_3.append([int(i)+1 for i in tlst_2[0:window_size]])
                tlst_2= tlst_2[window_size:len(tlst_2)]
            tlst_2=[int(i)+1 for i in tlst_2]+[0]*(window_size - len(tlst_2))
            input_3.append(tlst_2)
        else:   #1:False 2:True
            tlst=line.split('\n')[0].split(',')

            tlst_1=tlst[0:len(tlst)-1]
            tlst_2=tlst[1:len(tlst)]

            if drop:
                if len(tlst_1)>window_size:
                    tlst_1=tlst_1[0:window_size]
                    tlst_2=tlst_2[0:window_size]


            while len(tlst_1)>window_size:
                input_2.append([int(i)+bis for i in tlst_1[0:window_size]])
                tlst_1= tlst_1[window_size:len(tlst_1)]
            tlst_1=[int(i)+bis for i in tlst_1]+[0]*(window_size - len(tlst_1))
            input_2.append(tlst_1)

            while len(tlst_2)>window_size:
                input_4.append([int(i)+bis for i in tlst_2[0:window_size]])
                tlst_2= tlst_2[window_size:len(tlst_2)]
            tlst_2=[int(i)+bis for i in tlst_2]+[0]*(window_size - len(tlst_2))
            input_4.append(tlst_2)
        count+=1;
    file.close()
    E+=1

    input_1=torch.tensor(input_1)
    input_2=torch.tensor(input_2)
    input_3=torch.tensor(input_3)
    input_4=torch.tensor(input_4)
    
    print(E)
    print("Exercise ids  (t)")
    print(input_1[0]) #exercise id
    print("Response (t)")
    print(input_2[0]) #correct response
    print("Exercise ids  (t+1)")
    print(input_3[0]) #exercise ids 1 position ahead
    print("Response (t +1)")
    print(input_4[0])
    print("\n")

    if model_type=='sakt':
        input_1=input_1+E*input_2;
        
        print("============NETWORK INPUT=============== \n")
        print("E = 26688")
        print("INTERACTION SEQUENCE: Exercise ids  (t) + E*Response (t) \n",input_1[0])
        print("QUESTION SEQUENCE : Exercise ids  (t+1)\n ", input_3[0])
        print("RESPONSE SEQUENCE: Responses (t+1) \n", input_4[0] )
        
        return torch.stack((input_1,input_3,input_4),0),N,E,units
    elif model_type=='saint':
        return torch.stack((input_1,input_2),0),N,E,units
    else:
        raise Exception('model type error')

def dataloader(data, batch_size, shuffle:bool):
    data = data.permute(1,0,2);
    lis = [x for x in range(len(data))];
    if shuffle:
        random.shuffle(lis);
    lis = torch.Tensor(lis).long();
    ret = [];
    for i in range(int(len(data)/batch_size)):
        temp = torch.index_select(data, 0, lis[i*batch_size : (i+1)*batch_size]);
        ret.append(temp);
    #print(ret)  
    #print(len(ret))
    return torch.stack(ret, 0).permute(0,2,1,3);

class NoamOpt:
    def __init__(self, optimizer:torch.optim.Optimizer, warmup:int, dimension:int, factor=0.1):
        self.optimizer = optimizer;
        self._steps = 0;
        self._warmup = warmup;
        self._factor = factor;
        self._dimension = dimension;
        
    def step(self):
        self._steps += 1;
        rate = self._factor * (self._dimension**(-0.5) * min(self._steps**(-0.5), self._steps * self._warmup**(-1.5)));
        for x in self.optimizer.param_groups:
            x['lr'] = rate;