from numpy.core import numeric
from numpy.lib.arraysetops import unique
import pandas as pd 
from collections import Counter
from itertools import count

from pandas.core.indexing import need_slice

df = pd.read_csv('/home/thales/DSAKT/errex data subproblems.csv')

#df_1 = df[(df['student_id']== 'Stu_011b4d60b1589e545adb9ae7d5ad6f14') & (df['skill_name']=='CompleteTheSequence')]
#print(df_1['correct'].sum())


df_2 = pd.read_excel('/home/thales/DSAKT/ErrEx posttest data.xlsx')



students = df['student_id'].unique()
problems = df['problem_id'].unique()
kcs = df['skill_name'].unique()

'''
attempts_count = {}

for student in students:
    attempts_count[student] = dict(Counter(df[df['student_id']==student]['problem_id'].values))
    

student_repeat = []
for student in students:
    for count in attempts_count[student].values():
        if count > 1:
            student_repeat.append(student)
            break
'''


problem_per_KC = {}

for kc in kcs:
    problem_per_KC[kc] = len(df[df['skill_name'] == kc]['problem_id'].unique())

df_2 = df_2.drop([0,1], axis=0)
df_2 = df_2.drop(df_2.columns[1:5],axis=1)
for i in df_2.columns:
    if(('Unnamed'in str(i)) or (str(i).isalnum())):
        df_2.drop(i,axis='columns',inplace=True)
df_2.drop(['totals:', '11.1', '22.1', '4.1', '6.1'],axis='columns',inplace=True)


students_in_sheet = df_2['Anon Student Id'].values
students_in_csv = df['student_id'].unique()


'''
columns_sum = {}
i = 0
for column in df_2.columns:
    columns_sum[i] = df_2[column].sum()
    i = i +1

print(columns_sum)
'''
            

test_1 = df_2['Regz_AddDecimals1_Pre'].sum() 

problems_count_sum= {}

for id in problems:
    problems_count_sum[id] = df[df['problem_id']==id]['correct'].sum()

print(problems_count_sum)

#count pre columns
counter_pre = count(start=0,step=1)

for column in df_2.columns:
    if('_Pre' in column):
        next(counter_pre)
print(counter_pre)

#count pos columns
counter_pos = count(start=0,step=1)

for column in df_2.columns:
    if('_Post' in column):
        next(counter_pos)
print(counter_pos)


#count delPost columns 

counter_delpos = count(start=0,step=1)
for column in df_2.columns:
    if('_DelPost' in column):
        next(counter_delpos)
        
print(counter_delpos)

print('Total problems ids:',len(problems))