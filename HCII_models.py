#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
import pandas as pd
from collections import Counter
from sklearn.svm import SVR
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
import ann_HCI

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

data = pd.read_csv('datasets/HCI.csv', delimiter=' ')				
C = data[['cluster','engagement', 'length','robot_feedback', 'previous_score', 'current_score' , 'current_result', 'action']]

# cluster data
C0 = C.loc[C['cluster']==0]
C1 = C.loc[C['cluster']==1]
C2 = C.loc[C['cluster']==2]

P0 = C0[['engagement', 'length','robot_feedback', 'previous_score', 'current_result']]
P1 = C1[['engagement', 'length','robot_feedback', 'previous_score', 'current_result']]
P2 = C2[['engagement', 'length','robot_feedback', 'previous_score', 'current_result']]

a0 =  P0.groupby(['length', 'previous_score'])
a1 =  P1.groupby(['length', 'previous_score'])
a2 =  P2.groupby(['length', 'previous_score'])

D = [3,5,7,9]
S = [-4,-3,-2,-1,0,1,2,3,4]

L = [0.25, 0.5, 0.75, 1.0]
PS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

combs = (L, PS)
states = list(itertools.product(*combs))

state_level = []
tmp = 0 
for i, s in enumerate(states):
	if tmp < s[0]:
		state_level.append(i)
		tmp = s[0]	  
	
for ii, cluster in enumerate([a0,a1,a2]):
	train_X = []
	train_Y = []
	for key, item in cluster:
		A = cluster.get_group(key)
		wins = len(A.loc[A['current_result']==1]) 
		losses = len(A.loc[A['current_result']==-1]) 
		if wins == 0: 
			p = 0.0
		elif losses == 0:
			p = 1.0
		else:
			p = wins/float(wins+losses)

		training = [L[D.index(key[0])], PS[S.index(key[1])]]
		target = p
		train_X.append(training)
		train_Y.append(target)

	#svr = NuSVR(kernel='rbf', C=10.0)
	#svr.fit(train_X, train_Y)
	#svr_prediction = svr.predict(train_X)
    	#svr_mse = mean_squared_error(train_Y, svr_prediction)
    	#svr_rmse1 = np.sqrt(svr_mse)
    	#print "\t\tTraining Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse1)

	#NN = MLPRegressor(activation='logistic', max_iter=5000, solver='lbfgs', hidden_layer_sizes=(3, 6), random_state=42)
	#NN.fit(train_X, train_Y)
	#NN_prediction = NN.predict(train_X)

	N = ann_HCI.build_pmodel()
	x = np.asarray(train_X)
	y = np.asarray(train_Y)
	
	for a1, a2 in zip(x,y):
		print a1, a2
	#raw_input()
		
	N.fit(x, y, epochs=10000, verbose=0)
	
	preds = []
	for s in states:
		#inputs.append([s[0], s[1][0], s[1][1], s[1][2], s[2]])
		preds.append(N.predict(np.asarray([s[0], s[1]]).reshape(1,2))[0][0])
	
	#preds = N.predict(np.asarray(inputs))[0][0]
	plt.bar(left=state_level, height=[1.2,1.2,1.2,1.2], width=27, color=['k','w', 'k', 'w'], alpha=0.1)
	plt.hold(True)
	plt.plot(preds, label = 'regression values')
	first = 1
	for a,b in zip(train_X, train_Y): 
		if first: 
			plt.plot(states.index(tuple([a[0], a[1]])), b, 'or', label = 'real values')	
			first = 0
		else: 
			plt.plot(states.index(tuple([a[0], a[1]])), b, 'or')	
	plt.legend()
	plt.savefig('simulation/HCI_c' + str(ii) + '.png')
	plt.close()
	
	N.save('simulation/user' + str(ii) + '_HCI.h5', 'wb')


