#!/usr/bin/python
from naoqi import ALProxy
import numpy as np 
import readchar
import time
from random import randint
import os 
import muse_pyliblo_server as mps
import matplotlib.pyplot as plt
import gc
import random 
from math import isnan
from random import randint
import webbrowser
from RL import MDP
from RL import Policy
from RL import Learning
from RL import Representation
import pickle
import random
from datetime import datetime
from options import GetOptions
from keras.models import load_model
from scipy.spatial.distance import euclidean
import itertools
import csv

# initialize random seed
random.seed(time.time())

# NAO parameters
ROBOT_IP = "129.107.118.226" 
tts = ALProxy("ALTextToSpeech", ROBOT_IP, 9559)
memory = ALProxy("ALMemory", ROBOT_IP, 9559)
aup = ALProxy("ALAudioPlayer", ROBOT_IP, 9559)

# define state-action space
def state_action_space():	
	length   = [3,5,7,9]
	feedback = [0,1,2]
	previous = [-4,-3,-2,-1,0,1,2,3,4]
	
	combs = (length, feedback, previous)
	states = list(itertools.product(*combs))
	states.append((0,0,0))
	
	l = [1, 2, 3, 4]
	f = [[1,0,0], [0,1,0], [0,0,1]]
	combs = (l, f, previous)
	normalized_states = list(itertools.product(*combs))
	normalized_states.append((0,[0,0,0],0))

	actions = [0,1,2,3,4,5]
	actions_oh = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]]
	return states, normalized_states, actions, actions_oh


def normalize_by_range(x, nmin = 0, nmax = 1):
	x = np.asarray(x)
	return (nmax-nmin)*(x-min(x))/(max(x)-min(x)) + nmin

def ewma(Y, a = 0.2): 
	S = []
	for i, y in enumerate(Y): 
		if i == 0: 
			S.append(y)
		else: 
			S.append(a*Y[i-1] + (1-a)*S[i-1])
	return S

def estimate_engagement(filehandler):
	a, b, t, = [], [], []
	lines = filehandler.readlines()
	for line in lines:	
		w = line.split()
		if w[0] == 'a': 
			if isnan(float(w[1])):
				a.append(0.01)
			else: 
				a.append(float(w[1]))
		elif w[0] == 'b': 
			if isnan(float(w[1])):
				b.append(0.01)
			else: 
				b.append(float(w[1]))
		elif w[0] == 't': 
			if isnan(float(w[1])):
				t.append(0.01)
			else: 
				t.append(float(w[1]))

	#if np.isnan(a) or np.isnan(b) or np.isnan(t): 
	#	return [0]
	else: 
		a_smoothed = ewma(a)
		b_smoothed = ewma(b)
		t_smoothed = ewma(t)
		e = [x+y for x, y in zip(a_smoothed, t_smoothed)]
		engagement = [x/y for x, y in zip(b_smoothed, e)]
		return engagement
	
def get_next_state(state, action, previous):
	levels = {3:1, 5:2, 7:3, 9:4}
	if action == 0: 
		feedback = 0 
		length = 3
	if action == 1: 
		feedback = 0 
		length = 5
	if action == 2: 
		feedback = 0 
		length = 7
	if action == 3: 
		feedback = 0 
		length = 9
	if action == 4: 
		feedback = 1
		length = state[0]
	if action == 5: 
		feedback = 2
		length = state[0]
	
	next_state = [length, feedback, previous]
		
	return next_state

 
def introduction(): 
	# Robot Introduction 
	tts.say("Hi! My name is Stewie!")
	tts.say("Let's play a game!")
	time.sleep(0.5)
	"""
	tts.say("I will say a sequence of letters and you have to repeat it!")
	time.sleep(0.5)
	tts.say("After the sequence is completed, you will listen to a beep sound!")
	time.sleep(0.8)
	aup.playSine(1000, 100, 0, 1)
	
	time.sleep(1.5)
	tts.say("After the sound, you have to respond, by pressing the buttons in the correct order") 
	time.sleep(0.5)
	tts.say("Before each sequence, I will tell you, the difficulty level of the next sequence.") 
	time.sleep(0.5)
	tts.say("Level one has three letters! Level two, has five!! Level three, has 7 letters and level four, has nine!") 
	time.sleep(0.5)
	tts.say("Please remember!! You need to give your response. After the beep sound!")
	time.sleep(0.5)
	tts.say("Use only one hand, and make sure, that each button is pressed properly!")
	time.sleep(0.5)
	tts.say("Let's try with an example!")
	time.sleep(0.5)

	# example
	seq = ["b", "b", "a", "c", "a"]
	tts.say("Level 2")
	time.sleep(0.5)
	for item in seq:
		time.sleep(0.5)
		tts.say(item)
	aup.playSine(1000, 100, 0, 1)
	sig2 = 1 
	res = []	
	while(sig2):
		res.append(readchar.readkey().lower())
		if len(res) == len(seq):
			sig2 = 0

	time.sleep(0.7)
	"""
	tts.say("Great! Are you ready to start the session?")
	raw_input('Press to start')


def assessment(server,folder):
	engagement = []
	performance = []
	LEVELS = []
	E = {}
	D = [3,5,7,9]
	L = ('a','b','c')
	turns = 2
	t = 1
	while t <= turns:
		## record EEG signals when robot announces the sequence ##
		out = open(folder + "/assessment_" + str(t), 'w')
		server.f = out 

		level = int(raw_input("Level: "))
		LEVELS.append(level)
		time.sleep(0.5)

		diff_level = "Level" + str(level)
		tts.say(diff_level)
		time.sleep(0.5)

		#announce sequence
		seq = list(np.random.choice(L, D[level-1]))		
		for item in seq:
			time.sleep(0.8)
			tts.say(item)
		aup.playSine(1000, 100, 0, 1)
		time.sleep(1)

		## record EEG signals when user presses the buttons ##
		out = open(folder + '/dump', 'w')
		server.f = out
		
		# get engagement
		eegfile = open(folder + "/assessment_" + str(t), 'r')
		eng = estimate_engagement(eegfile)
		engagement.append(eng)
		
		# start time to measure response time
		start_time = time.time()

		# CHECK USER RESPONSE AND CALCULATE SCORE 
		sig2 = 1
		first = 0 
		res = []	
		while(sig2):
			res.append(readchar.readchar().lower())
			if first == 0: 
				reaction_time = time.time() - start_time
				first = 1
			if len(res) == D[level-1]:
				sig2 = 0 
	
		completion_time = time.time() - start_time
		if seq != res:
			success = -1.0
			score = -1.0*level
		else: 
			success = 1
			score = level

		performance.append(score)
		t += 1

	# normalize EEG values - engagement 
	m1, m2 = 0, 0
	for e in engagement: 
		if m1 > min(e): 
			m1 = min(e)
		if m2 < max(e):
			m2 = max(e)
	
	normed = []
	for e in engagement: 
		e = np.asarray(e)
		n = (m2-m1)*(e-min(e))/(max(e)-min(e)) + m1
		normed.append(n)


	for a,b in zip(LEVELS, normed): 
		if E.has_key(a):
			E[a].append(np.asarray(b).mean())
		else: 
			E[a] = [np.asarray(b).mean()]

	um = []
	for i in [1.0,2.0,3.0,4.0]: 
		if performance.count(i) == 0 and performance.count(-1*i) == 0: 
			um.append(0.0)
		else: 
			um.append(performance.count(i)/float((performance.count(i) + performance.count(-1*i))))
	
	en = [0,0,0,0]	
	for i in [1,2,3,4]: 
		if E.has_key(i):
			en[i-1] = np.asarray(E[i]).mean()
	
	UM =  [um[0], um[1], um[2], um[3], en[0], en[1], en[2], en[3]]

	UM0 = [0.99, 0.8,  0.39, 0.05, 0.33, 0.35, 0.36, 0.36]
	UM1 = [0.99, 0.93, 0.57, 0.54, 0.31, 0.31, 0.33, 0.34]
	UM2 = [0.97, 0.94, 0.66, 0.15, 0.37, 0.38, 0.39, 0.4 ]
	DIST = [euclidean(UM, UM0), euclidean(UM, UM1), euclidean(UM, UM2)]
	
	print DIST, DIST.index(min(DIST))

	#raw_input("Agree with cluster?")
	
	return DIST.index(min(DIST)), m1, m2


def training(server,cond, folder, um, m1, m2): 
	D = [3,5,7,9]
	L = ('a','b','c')
	A = ['L1', 'L2', 'L3', 'L4', 'RF1', 'RF2']

	positive_success = ["That was great! Keep up the good work!", "Wow, you do really great! Go on!", "That's awesome! You got it! Keep going!", "Fantastic! You do great! Keep going!"]
	positive_failure = ["Oh, that was wrong! But that's fine! Don't give up!", "Oh, you missed it! No problem! Go on!", "Oops, that was not correct! That's OK! Keep going!", "Oops, too close! Stay focused and you will do it!"]
	negative_success = ["Ok, that was easy enough! Let's see again!", "Well, ok! Maybe you were lucky! Let's check the next one!", "OK, you got it! Was it random?? Let's try again.", "OK, I guess you made it! Maybe, it was an easy one!"]
	negative_failure = ["Hmmm! I don't think you are paying any attention! Try harder!", "Hey! Are you there? Stay focused when I speak!", "Oh! was that wrong? Well, you actually need to pay attention!", "If you want to succeed, you need to pay attention!"]

	q = 'policies/p' + str(um)

	# start and terminal states and indices
	states, normed_states, actions, actions_oh = state_action_space()
	start_state = (0,0,0)
	start_state_index = states.index(tuple(start_state))
	previous_state = []
	previous_result = 0

	# define MDP, learning and q_table
	m = MDP(start_state, actions)
	m.states = states
	To = 4
	gamma = 0.95
	alpha = 0.6	 
	table = Representation('qtable', [m.actlist, m.states])
	Q = np.asarray(table.Q)
	ins = open(q,'r')
	Q = [[float(n) for n in line.split()] for line in ins]
	ins.close()	
	table.Q = Q
	egreedy = Policy('softmax',To)
	learning = Learning('qlearn', [alpha, gamma])

	# session parameters
	t = 1
	turns = 20
	state = start_state
	done = 0
	logfile = open(folder + "/logfile", 'w')
	while t<turns:

		egreedy.param *= 0.65
		if egreedy.param <= 1: 
			egreedy.param = 1

		# record EEG signals when robot announces the sequence 
		out = open(folder + "/training_l_" + str(t), 'w')
		server.f = out
		
		# SELECT ACTION 
		if condition == 1: # random actions
			if first: 
				action = randint(0,4)
			else: 
				action = randint(0,6)
		elif condition == 2: # RL-based policy
			state_index = states.index(tuple(state))
			if state_index == start_state_index: 
				egreedy.Q_state = Q[state_index][:4]
			else: 
				egreedy.Q_state = Q[state_index][:]
			action = egreedy.return_action()	
	

		# PROVIDE FEEDBACK - IF NEEDED
		rf = 0 	
		if action == 4: 
			rf = 1
			seq = list(np.random.choice(L, Dold))
		elif action == 5: 
			rf = 2
			seq = list(np.random.choice(L, Dold))
		else: 
			seq = list(np.random.choice(L, D[action]))
			Dold = D[action]


		r = randint(0,3)
		if rf == 1: 
			if previous_success == 1: 
				tts.say(positive_success[r])
			else: 
				tts.say(positive_failure[r])

		if rf == 2: 
			if previous_success == 1: 
				tts.say(negative_success[r])
			else: 
				tts.say(negative_failure[r])

		# announce difficulty level
		time.sleep(0.5)
		length = len(seq)
		diff_level = "Level" + str(D.index(length)+1)
		tts.say(diff_level)
		time.sleep(0.5)
	
		# announce sequence and beep sound
		for item in seq:
			time.sleep(0.8)
			tts.say(item)
		aup.playSine(1000, 100, 0, 1)
		time.sleep(1)

		## record EEG signals when user presses the buttons ##
		out = open(folder + "/training_r_" + str(t), 'w')
		server.f = out

		# get engagement
		eegfile = open(folder + "/training_l_" + str(t), 'r')
		eng = estimate_engagement(eegfile)

		# normalize EEG engagement 
		if min(eng) < m1: 
			m1 = min(eng)
		if max(eng) > m2: 
			m2 = max(eng)

		eng = normalize_by_range(eng, m1, m2)
		engagement = eng.mean()
		feedback = engagement 

		# start time to measure response time
		start_time = time.time()

		# CHECK USER RESPONSE AND SCORE
		sig2 = 1
		first = 0 
		res = []	
		while(sig2):
			res.append(readchar.readchar().lower())
			if first == 0: 
				reaction_time = time.time() - start_time
				first = 1
			if len(res) == Dold:
				sig2 = 0 
	
		completion_time = time.time() - start_time
		if seq != res:
			success = -1
			score = success
			result = -1*(D.index(length) + 1)
		else: 
			success = 1
			score = D.index(length) + 1
			result = score
	
		## record EEG signals when user presses the buttons ##
		out = open(folder + '/dump', 'w')
		server.f = out
		######################################################
		
		reward = result if result >= 0 else -1.0
		reward += 4.5*feedback

		
		next_state = (length, rf, int(previous_result)) 
		next_state_index = states.index(tuple(next_state))
		
		if t == turns: 
			done = 1

		egreedy.Q_state = Q[next_state_index][:]
		next_action = egreedy.return_action()
		
		if condition == 2:
			Q[state_index][:], error = learning.update(state_index, action, next_state_index, next_action, reward, Q[state_index][:], Q[next_state_index][:], done)
		
		logfile.write(str(t) + '... ' + str(state) + ' ' + str(A[action]) + ' ' + str(next_state) + ' ' + str(reward)  + ' ' + str(score)  + ' ' +  str(engagement) + '\n')

		previous_result = result
	        state = next_state
		previous_success = success

		t += 1

	return Q
	
# user info and folders
user = raw_input('Enter userID: ')
condition = int(raw_input('Enter condition (1 or 2): '))
user_folder = "data/user" + str(user) + '/condition' + str(condition)
if not os.path.exists(user_folder):
	os.makedirs(user_folder)
intro = open(user_folder + "/intro", 'w')

server = 0
m1 = 0
m2 = 1 

# MAIN 
server = mps.initialize(intro)
server.start()

introduction()
tts.say('Great! Lets start!')
time.sleep(0.5)

um, m1, m2 = assessment(server, user_folder)
print um, m1, m2

Q = training(server, condition, user_folder, um, m1, m2)
with open(user_folder + '/q_table', 'w') as f:
	writer = csv.writer(f,delimiter=' ')
	writer.writerows(Q)
