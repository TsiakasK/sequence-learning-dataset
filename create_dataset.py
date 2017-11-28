#!/usr/bin/python
import numpy as np 
import re

# dict for user-clusterID
def load_clusters(filename):
	filehandler = open(filename, 'r')
	lines = filehandler.readlines()
	filehandler.close()
	Clusters = {}
	for line in lines:
		A = re.split('\s+', line)
		Clusters[A[0]] = A[1]

	return Clusters	

clusters = load_clusters('datasets/user_models_both.csv')
for index in [1,2,3,4,5]: 
	DF1 = open('datasets/normalized_' + str(index) + '.csv','r')
	DF2 = open('datasets/final_' + str(index) + '.csv','w')
	DF2.write('ID cluster length robot_feedback previous_score current_result current_score engagement action\n')
	lines = DF1.readlines()[1:]

	for line in lines: 
		a = line.strip().split()
		DF2.write(a[0] + ' ' +  clusters[a[0]] + ' ' + a[1] + ' '  + a[2] + ' '  + a[3] + ' '  + a[4] + ' '  + a[5] + ' '  + a[6] + ' ' + a[7] + '\n')
	DF2.close()
