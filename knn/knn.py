import os, json, gc, pickle
from collections import Counter
from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pickle
import math
from random import randrange
from statistics import mode
from decimal import * 
from math import sqrt
import numpy as np

#sys.path.insert(os.path.join(os.path.dirname(file), '../datapipeline'))
#from datapipeline import *

from load_data import get_x_y_floor, gen_for_serialisation, get_data_for_test

pt_testset= ''
path_to_save =''
path_to_train = ""
path_to_sample = ""
# we use data pipeline to retrieve RSSID values
training = [[-999, 34, 36], [16, -999, 43], [26, -999, 43], [36, -999, 43], [46, -999, 43], [56, -999, 43], [66, -999, 43], [76, -999, 43], [86, -999, 43], [96, -999, 43], [106, -999, 43]]
# we use data pipeline to then retrieve the ground truth data corresponding to the train array
# format: [x,y, floor]
ground_truth = [[1, 2, 3], [11, 4, 7], [2, 4, 7], [3, 4, 7], [4, 4, 7], [5, 4, 7], [6, 4, 7], [7, 4, 7], [8, 4, 7], [9, 4, 7], [10, 4, 7]]

def cross_validation(train_set, n_folds, ground_truth, num_neighbors):
	ground_truth = np.asarray(ground_truth).astype(np.dtype(float))
	train_set = np.asarray(train_set).astype(np.dtype(float))
	kfold = KFold(n_splits = n_folds, shuffle=True)
	scores = list()

	for train, test in kfold.split(train_set, ground_truth): 
		for row in test:
			prediction = predict_regression(train_set[train], train_set[row], num_neighbors, ground_truth[train])
			actual = ground_truth[row]
			scores.append(accuracy_metric(actual, prediction))
		print(1/range(len(actual)) * sum(scores))
	return scores
		
	# folds = cross_validation_split(dataset, n_folds)
	# scores = list()
	# for fold in folds:
	# 	train_set = list(folds)
	# 	train_set.remove(fold)
	# 	train_set = sum(train_set, [])
	# 	test_set = list()
	# 	for row in fold:
	# 		row_copy = list(row)
	# 		test_set.append(row_copy)
	# 		row_copy[-1] = None
	# 	predicted = predict_regression(train_set, test_set, num_neighbors, ground_truth)
	# 	actual = [row[-1] for row in fold]
	# 	accuracy = accuracy_metric(actual, predicted)
	# 	scores.append(accuracy)
	# return scores

# def cross_validation_split(dataset, n_folds):
# 	dataset_split = list()
# 	dataset_copy = list(dataset)
# 	fold_size = int(len(dataset) / n_folds)
# 	for _ in range(n_folds):
# 		fold = list()
# 		while len(fold) < fold_size:
# 			index = randrange(len(dataset_copy))
# 			fold.append(dataset_copy.pop(index))
# 		dataset_split.appenad(fold)
# 	return dataset_split

def accuracy_metric(actual, predicted):
	for i in range(len(actual)-1):
		print(actual)
		print(predicted)
		correct = actual[i] - predicted[i]
		print(correct)
	return sqrt(sum(correct)^2)

def predict_regression(train, row, num_neighbors, ground_truth):
	neighbors = closest_neighbors(num_neighbors, train, ground_truth, row)
	x = list()
	y = list()
	floor = list()
	prediction = list()

	for neighbor in neighbors:
		x.append(neighbor[0])
		y.append(neighbor[1])
		floor.append(neighbor[2])
	
	prediction.append(sum(x) / len(x))
	prediction.append(sum(y) / len(y))
	prediction.append(mode(floor))
	
	return prediction

def closest_neighbors(num_neighbors, train, ground_truth, row):
	distances = list()
	array_number = list()
	i = 0
	for train_row in train:
		dist = euclidean_distance(row, train_row)
		distances.append((ground_truth[i], dist))
		i = i+1
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	print(neighbors)
	return neighbors

def euclidean_distance(row1, row2):
	distance = 0.0
	for point1, point2 in np.nditer([row1, row2]):
		distance += ((int(point1) - int(point2))**2)
	return math.sqrt(distance)

def minkowski_distance(row1, row2):
    distance = 0.0 
    for i in range(len(row1)-1):
        distance += (abs(row1[i] - row1[j]) + abs(row2[i] - row2[j]))
    return distance

def get_sample_submission_index(path_to_sample):
    df = pd.read_csv(path_to_sample)
    return df["site_path_timestamp"]

""" def __main__():
	gen = gen_for_serialisation(path_to_train)
	for site, train, truth in gen:
		test_data = get_data_for_test(pt_testset, site)
		
		break """


f = open('./5a0546857ecc773753327266.pickle', "rb")
site, train, ground = pickle.load(f)
f.close
#train_dict = {}
#train = np.array(train)
#ground = np.array(ground)
#for train_row in train: 
	#prediction = predict_regression(train, train_row, 5, ground)
#print(type(ground))
#train_dict = dict(enumerate(train.flatten(), enumerate(ground.flatten()))
#prediction = predict_regression(train, train[1], 3, ground)
scores = cross_validation(train, 10, ground, 10)

#print(prediction)
#evaluate_algorithm(train, )
#print('Expected %s, Got %s.' % (train[0], predict))
#n_folds = 5
#num_neighbors = 5
#scores = evaluate_algorithm(train, 5, n_folds, num_neighbors)