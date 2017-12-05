#!/usr/bin/python
import numpy as np 
import os 
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import csv 
import sys
from sklearn.metrics import euclidean_distances
from sklearn import manifold
import random
import itertools
import pandas as pd
from sklearn.cluster import DBSCAN
seed = np.random.RandomState(seed=100)

markers = ['o', 'v', 'h', 'H', 'o', 'v', 'h', 'H', 'h', 'H', 'o', 'v', 'h']
colors = ['b', 'r','g','c','y','m', 'b', 'r','g','c','y','m','y','m', 'b']
combs = list(itertools.product(markers, colors))

for index in [1]:
	data = pd.read_csv('datasets/normalized_' + str(index) + '.csv', delimiter=' ')

	C = data[['ID','engagement','length','current_result']]
	users = C['ID'].unique()
	clusters = 3

	ff = open('datasets/user_models.csv','w')

	user_models = []
	perf_models = []
	eng_models = []
	userID = []
	for user in users: 
		userID.append(user)
		D = C.loc[C['ID']==user]
		um = [-1.0,-1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0]
		perf = [-1.0,-1.0,-1.0,-1.0]
		eng = [0.0, 0.0, 0.0, 0.0]	
		for i, l in enumerate([3,5,7,9]):
			L = D.loc[D['length']==l]
			wins = len(L.loc[L['current_result']==1]) 
			losses =  len(L.loc[L['current_result']==-1])
			if wins == 0: 
				um[i] = 0.0
				perf[i] = 0.0
			elif losses == 0:
				um[i] = 1.0
				perf[i] = 1.0
			else:
				um[i] = wins/float(wins+losses)
				perf[i] = wins/float(wins+losses)
		
			um[i+4] = L['engagement'].mean()
			eng[i] = L['engagement'].mean()
		
		user_models.append(um)
		perf_models.append(perf)
		eng_models.append(eng)

		ff.write(user)
		for m in um: 	
			ff.write(' ' + str(m))
		ff.write('\n')
	ff.close()	


	#print user_models
	#labels = ["performance_engagement", "performance_based", "engagement_based"]
	#models = [user_models, perf_models, eng_models]
	labels = ["performance", "engagement", "both"]
	models = [perf_models, eng_models, user_models]	

	clusterID = []
	for model, label in zip(models, labels):
		M = model
		model -= np.asarray(model).mean()
		similarities = euclidean_distances(model)
		mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-6, random_state=seed, dissimilarity="precomputed", n_jobs=2)
		pos = mds.fit(similarities).embedding_

		#for i, p in enumerate(pos): 
		#	plt.plot(p[0], p[1], combs[i][0], markersize=9, color = combs[i][1])
		#plt.title('Multidimensional Scaling')
		#plt.savefig(label + "_mds.png")
		#plt.close()

		# CLUSTERING on 2-D
		kmeans = KMeans(n_clusters=clusters, random_state=0).fit(model)
		#kmeans = DBSCAN(eps = 0.1, min_samples=15).fit(pos)
		#print kmeans.labels_
		mm = ['b', 'r', 'g', 'c']
		first = [1,1,1,1]
		for i, p in enumerate(pos): 
			if first[kmeans.labels_[i]]: 
				plt.plot(p[0], p[1], 'o', markersize=9, color = mm[kmeans.labels_[i]], label = 'cluster_' + str(kmeans.labels_[i] + 1))
				first[kmeans.labels_[i]] = 0 
			else: 
				plt.plot(p[0], p[1], 'o', markersize=9, color = mm[kmeans.labels_[i]])

			clusterID.append(kmeans.labels_[i])

		plt.legend()
		plt.title('Clustering using MDS')
		plt.savefig('figures/'+ label + '_clustering.png')
		plt.close()

		f = open('datasets/user_models_' + str(label) + '.csv','w')
		if label == "peformance":
			f.write("ID cluster p1 p2 p3 p4\n")
		if label == "engagement":
			f.write("ID cluster e1 e2 e3 e4\n")
		if label == "both": 
			f.write("ID cluster p1 p2 p3 p4 e1 e2 e3 e4\n")

		for u, c, m in zip(userID, kmeans.labels_, M): 
			f.write(u + ' ' + str(c))
			for mm in m: 
				f.write(' ' + str(mm))
			f.write('\n')	
		f.close()
			
#f = open('datasets/user_models_clusters.csv','w')
#f1 = open('datasets/user_models.csv','w')
#f2 = open('datasets/user_clusters.csv','w')
#for a,b, model in zip(userID, clusterID, perf_models): 
#	f.write(str(a) + ' ' + str(b) + ' ' + str(model[0]) + ' ' + str(model[1]) + ' ' + str(model[2]) + ' ' + str(model[3]) + '\n')
#	f1.write(str(a) + ' ' + str(model[0]) + ' ' + str(model[1]) + ' ' + str(model[2]) + ' ' + str(model[3]) + '\n')
#	f2.write(str(a) + ' ' + str(b) + '\n')
