#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
import pandas as pd
from collections import Counter
import itertools
from keras.models import load_model
import random


user = raw_input("enter user ID: ")
model = load_model('simulation/user' + str(user) + '_HCI.h5')

D = [1,2,3,4]
S = [-4,-3,-2,-1,0,1,2,3,4]

L = [0.25, 0.5, 0.75, 1.0]
PS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

previous_level = 0

scores = []
for i in range(6):
	level = raw_input("Select Level: ")
	point = [L[D.index(int(level))], PS[S.index(previous_level)]] 
	

	prob = model.predict(np.asarray(point).reshape(1,2))[0][0]
	
	if random.random() <= prob: 
		success = 1
	else: 
		success = -1

	previous_level = int(level)*success
	scores.append(previous_level)
	
	print success

um = []
for i in [1,2,3,4]: 
	if scores.count(i) == 0 and scores.count(-1*i) == 0: 
		um.append(-1.0)
	else: 
		um.append(scores.count(i)/float((scores.count(i) + scores.count(-1*i))))



print um 
#plt.bar([1,2,3,4], um)
plt.bar(range(6), scores)
plt.show()
