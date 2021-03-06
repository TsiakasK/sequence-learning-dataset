#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
from collections import Counter

def normalize_by_range(x, nmin = 0, nmax = 1):
	x = np.asarray(x)
	return (nmax-nmin)*(x-min(x))/(max(x)-min(x)) + nmin

def normalize_by_mean(x): 
	x = np.asarray(x)
	return x-x.mean()

def statistics(x):
	x = np.asarray(x)
	return min(x), max(x), x.mean()

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

def read_from_file(f):
	a, b, g, d, t, c = [], [], [], [], [], []
	lines = f.readlines()
	for line in lines:	
		w = line.split()
		if w[0] == 'a': 
			a.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'b': 
			b.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'g': 
			g.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'd': 		
			d.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 't': 
			t.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'c':
			c.append(float(w[1]))
	
	return np.asarray(a).mean(axis=1), np.asarray(b).mean(axis=1),np.asarray(g).mean(axis=1),np.asarray(d).mean(axis=1),np.asarray(t).mean(axis=1), c

def ewma(Y, a = 0.1): 
	S = []
	for i, y in enumerate(Y): 
		if i == 0: 
			S.append(y)
		else: 
			S.append(a*Y[i-1] + (1-a)*S[i-1])
	return S	

#clusters = load_clusters('datasets/user_clusters.csv')

if not os.path.exists('EEG_analysis/'):
	os.makedirs('EEG_analysis/')

D = [3,5,7,9]
indices = [1,2,3,4,5]
indices = [1]
dirname = "clean_data"
users = os.listdir(dirname)

for index in indices: 
	DF1 = open('datasets/index_' + str(index) + '.csv','w')
	DF1.write('ID length robot_feedback previous_score current_result current_score engagement action\n')
	DF2 = open('datasets/normalized_' + str(index) + '.csv','w')
	DF2.write('ID length robot_feedback previous_score current_result current_score engagement action\n')
	AF1 = open('EEG_analysis/index_' + str(index), 'w')
	
	F = 1 
	FULL = []
	means = []
	for user in users:
		sessions = os.listdir(dirname + '/' + user)
		for session in sessions:
			#print user + '/' + session

			EE = []
			turn_mean = []
			i = 0

			if not os.path.exists('EEG_analysis/' + user + '/' + session):
				os.makedirs('EEG_analysis/' + user + '/' + session)

			UF1 = open('EEG_analysis/' + user + '/' + session + '/index_' + str(index), 'w')
			#ff2 = open('EEG_analysis/' + user + '/' + session + '/normalized', 'w')
		
			file_name = dirname + '/' + user + '/' + session
			log1 = open(file_name + '/state_EEG', 'r')
			lines1 = log1.readlines()
			log1.close()

			log2= open(file_name + '/logfile','r')
			lines2 = log2.readlines()
			log2.close()

			last = 1
			turn = [0]
			for a,b in zip(lines1, lines2): 
				A = re.split('\s+', a)[:-1]
				B = re.split('\s+', b)[:-1]

				eeg_filename = A[3]
				length = A[0]
				rf = A[1]
				ps = A[2]
				result = B[4]
				score = int(B[4])*int(D.index(int(length)) + 1)

				action = D.index(int(length))
				if int(rf) == 1: 
					action = 4
				if int(rf) == 2: 
					action = 5

				if F:
					F = 0
					last = 0 
				else:  
					if last: 
						#DF1.write(' -1\n')
						#AF1.write(' -1\n')
						last = 0
					else: 
						DF1.write(' ' + str(action) + '\n')
						AF1.write(' ' + str(action) + '\n')
						UF1.write(' ' + str(action) + '\n')

				f = open(file_name + '/' + eeg_filename, 'r')
				a, b, g, d, t, c = read_from_file(f)
				a_smoothed = ewma(a)
				b_smoothed = ewma(b)
				t_smoothed = ewma(t)
				concentration = ewma(c)

				if index == 1: 
					# index 1
					e = [x+y for x, y in zip(a_smoothed, t_smoothed)]
					engagement = [x/y for x, y in zip(b_smoothed, e)]
				elif index == 2: 
					# index 2
					engagement = [x/y for x, y in zip(b_smoothed, t_smoothed)]
				elif index == 3: 
					# index 3
					engagement = [x/y for x, y in zip(b_smoothed, a_smoothed)]
				elif index == 4: 
					# index 4
					engagement = [x/y for x, y in zip(t_smoothed, a_smoothed)]
				elif index == 5: 
					# Muse built-in concentration
					engagement = concentration 
			
				#clength = len(engagement)
				i = i + len(engagement)
				turn.append(i)
				turn_mean.append(np.asarray(engagement).mean())
				means.append(np.asarray(engagement).mean())			

				DF1.write(user + '/' + session + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score) + ' ' + str(np.asarray(engagement).mean()))
				AF1.write(user + '/' + session + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score))
				UF1.write(user + '/' + session + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score))
			
				for E in engagement: 
					AF1.write(' ' + str(E))
					UF1.write(' ' + str(E))
					FULL.append(E)
					EE.append(E)	
			
			DF1.write(' -1\n')
			AF1.write(' -1\n')
			UF1.write(' -1\n')
			UF1.close()
			
			#scaled = normalize_by_mean(EE)
			normed = normalize_by_range(EE)
			plt.subplot(311)
			plt.plot(EE)
			plt.xlim([0,len(EE)])
			plt.subplot(312)
			plt.plot(normed)
			plt.xlim([0,len(normed)])
			plt.hold(False)
		
			UF2 = open('EEG_analysis/' + user + '/' + session + '/normalized_' + str(index), 'w')		
		
			normed_mean = []
			tmp = []
			aa = 1 
			for ii, n in enumerate(normed):
				if ii < len(normed) and aa < len(turn): 
					if ii == turn[aa]: 
						UF2.write('\n')
						normed_mean.append(np.asarray(tmp).mean())
						tmp = []
						aa += 1
				UF2.write(str(n) + ' ')
				tmp.append(n)
				
			normed_mean.append(np.asarray(tmp).mean())
			UF2.write('\n')
			UF2.close()

			# normalized index file
			f1 = open('EEG_analysis/' + user + '/' + session + '/index_' + str(index), 'r')
			f2 = open('EEG_analysis/' + user + '/' + session + '/normalized_' + str(index), 'r')
		
			lines1 = f1.readlines()
			lines2 = f2.readlines()
			f1.close()
			f2.close()
			
			for a,b in zip(lines1, lines2):
				aa = a.split()
				bb = b.split()
				eng = np.asarray(bb).astype(float).mean()
				DF2.write(str(aa[0])+ ' '+str(aa[1])+' '+str(aa[2])+' '+str(aa[3])+' '+str(aa[4])+' '+str(aa[5])+' '+str(eng)+' '+str(aa[-1]) + '\n')
		
			plt.subplot(313)
			plt.plot(range(1,26), normed_mean)
			#plt.title('Mean engagement per turn')
			plt.xlabel('Turns')
			plt.xlim([1,25])
			plt.savefig('EEG_analysis/' + user + '/' + session + '/index_' + str(index) + '.png')
			plt.hold(False)
			plt.close()
			
			#print normed_mean
			#raw_input()

	#weights = np.ones_like(means)/float(len(means))
	#plt.hist(means, bins = 10, weights = weights)
	#plt.title('engagement means - index ' + str(index))
	#plt.savefig('EEG_analysis/index_' + str(index) + '.png')
		


