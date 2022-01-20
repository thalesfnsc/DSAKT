import csv
import math
from os import close
import pandas as pd
import numpy as np 
#https://github.com/sagarjauhari/educational-data-mining

SR = 0.0
kcdict={}
gamma_kc = []
rho_kc = []
student_obj_dict={}
sr_list = []
student_ability = {}
SSR_min = 9999999.0 
b_min = 0.0
g_min = 0.0
r_min = 0.0
p_min = 0.0




def init_gamma_rho(g,r):
	global gamma_kc
	global rho_kc
	for i in range(0,len(kcdict)):
		gamma_kc.append(g)
		rho_kc.append(r)

####
def createKCList():
	id = 0
	with open("/home/thales/DSAKT/errex data subproblems.csv","rb") as file:
		df = pd.read_csv(file)
	
	kc_list = df['skill_name'].unique()
	print(len(kc_list))
	for i in range(len(kc_list)):
		kcdict[kc_list[i]] = id
		id = id +1

####

class Student:
	gamma = [0] * len(kcdict)
	rho = [0] * len(kcdict)
	success_param = 0
	fail_param = 0

def update_student(student, right, kc):

	if right == 1:
		student_obj_dict[student].gamma[kc] = student_obj_dict[student].gamma[kc] + gamma_kc[kc]

	else:
		student_obj_dict[student].rho[kc] = student_obj_dict[student].rho[kc] + rho_kc[kc]

	student_obj_dict[student].success_param = sum(student_obj_dict[student].gamma)
	student_obj_dict[student].fail_param = sum(student_obj_dict[student].rho)

def do_pfa(beta):
	SSR = 0
	
    ######
	with open('/home/thales/DSAKT/errex data subproblems.csv','rb') as csvfile:
		
		df = pd.read_csv(csvfile)
		
		for index,row in df.iterrows():
			p = 0.0
			student = row[0]
			kc = kcdict[row[3]]
			right = int(row[4])
			first_att = int(row[4])
	
		
		######
			if student in student_obj_dict:
				update_student(student, right, kc)			
			else:
				student_obj_dict[student] = Student()
				student_obj_dict[student].gamma = [0] * len(kcdict)
				student_obj_dict[student].rho = [0] * len(kcdict)
				update_student(student, right, kc)

			m = student_obj_dict[student].gamma[kc] + student_obj_dict[student].rho[kc] + beta
			
			if m < -9:
				p = 0
			else:
				p = 1 / (1 + math.e**(-m))
			#sr = math.pow((p - int(right)),2)*int(first_att)
			sr = math.pow((p - int(right)),2)
			
			SSR = SSR + sr
	return SSR,p

def do_optimization():

	global SSR_min
	global g_min
	global r_min
	global b_min
	global p_min
	
	for beta in np.arange(-2,-1,0.5):
		for gamma in np.arange(1,2,0.5):
			for rho in np.arange(1,2,0.5):
				init_gamma_rho(gamma,rho)
				SSR,p = do_pfa(beta)
				print (beta, gamma, rho, SSR, p)
				if SSR <= SSR_min:	
					SSR_min = SSR
					g_min = gamma
					r_min = rho
					b_min = beta
					p_min = p
		if (p == 1.0):
			break

def do_optimization_test():
	global SSR_min
	init_gamma_rho(0.5,0.5)
	SSR = do_pfa(0.5)
	if SSR < SSR_min:
		SSR_min = SSR

def compute_ability():	

	for student in student_obj_dict:
		student_ability[student] = {}
		for j in kcdict:
			m = student_obj_dict[student].gamma[kcdict[j]] + student_obj_dict[student].rho[kcdict[j]] + b_min
			p = 1 / (1 + math.e**(-m))
			student_ability[student][j] = p


createKCList()

do_optimization()
compute_ability()
count = 0


print(len(student_obj_dict))
mean_knowledge_kc = {}
for j in kcdict:	
	sum = 0
	for student in student_obj_dict:
		sum = student_ability[student][j] + sum 
	mean_knowledge_kc[j] = sum/len(student_obj_dict)

	

print(mean_knowledge_kc)

'''
with open("students_learning_probabilities.txt","w") as f:
	
	for student in student_ability:
		count = count + 1
		f.write('\n')
		f.write('student id: ' + str(student))
		f.write('\n')
		f.write('skill learning probability: ' + str(student_ability[student]))
		f.write('\n')
		if count == 5:
			break


'''


count = 0
with open("students_info.txt","w") as f:

		for student in student_obj_dict:
			f.write('\n')
			f.write('student id : ' + str(student))
			f.write('\n')
			f.write('Gamma: ' + str(student_obj_dict[student].gamma))
			f.write('\n')
			f.write('Rho:' + str(student_obj_dict[student].rho))
			f.write('\n')
			f.write('Sucess param:' + str(student_obj_dict[student].success_param))
			f.write('\n')
			f.write('Fail param:' + str(student_obj_dict[student].fail_param))
			f.write('\n')
			f.write("min Beta:" + str(b_min))

			if count ==5:
				break

			count = count + 1



#TODO: Think about how to understand the parameters
#check this again  : https://github.com/thosgt/kt-algos
#encode a relation of what skill represent the position of student_obj_dict 
#kc will be the column 4, not the 1
#kc = skill

'''
In this study, the algorithm was implemented in Excel following the formulas in Pavlik et al. (2009), and
using the Excel equation solver to determine optimal parameter estimates. The final learning probability
was recorded for each skill for each student.
'''