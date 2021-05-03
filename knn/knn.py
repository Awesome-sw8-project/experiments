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

#sys.path.insert(os.path.join(os.path.dirname(file), '../datapipeline'))
#from datapipeline import *

#from load_data import get_x_y_floor, gen_for_serialisation, get_data_for_test

# we use data pipeline to retrieve RSSID values
train = [[-999, 34, 36], [16, -999, 43], [26, -999, 43], [36, -999, 43], [46, -999, 43], [56, -999, 43], [66, -999, 43], [76, -999, 43], [86, -999, 43], [96, -999, 43], [106, -999, 43]]
# we use data pipeline to then retrieve the ground truth data corresponding to the train array
# format: [x,y, floor]
ground_truth = [[1, 2, 3], [11, 4, 7], [2, 4, 7], [3, 4, 7], [4, 4, 7], [5, 4, 7], [6, 4, 7], [7, 4, 7], [8, 4, 7], [9, 4, 7], [10, 4, 7]]

def cross_validation(train_set, n_folds, ground_truth, num_neighbors):
	kfold = KFold(n_splits = n_folds, shuffle=True)
	scores = list()

	for train, test in kfold.split(train_set, ground_truth): 
		
		prediction = predict_regression(train_set[train], test, num_neighbors, ground_truth[train])
		actual = [row[-1] for row in n_folds]
		accuracy = accuracy_metric(actual, prediction)
		scores.append(accuracy)
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

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def predict_regression(train, row, num_neighbors, ground_truth):
	neighbors = closest_neighbors(5, train, ground_truth, train[0])
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
	return neighbors

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (int(row1[i]) - int(row2[i]))**2
    return math.sqrt(distance)

def minkowski_distance(row1, row2):
    distance = 0.0 
    for i in range(len(row1)-1):
        distance += (abs(row1[i] - row1[j]) + abs(row2[i] - row2[j]))
    return distance


f = open('./5a0546857ecc773753327266.pickle', "rb")
site, train, ground = pickle.load(f)
f.close
#train_dict = {}


#train_dict = dict(enumerate(train.flatten(), enumerate(ground.flatten()))
prediction = predict_regression(train, train[1], 3, ground)
scores = cross_validation(train, 10, predict_regression, ground)
print(prediction)
#evaluate_algorithm(train, )
#print('Expected %s, Got %s.' % (train[0], predict))
n_folds = 5
#num_neighbors = 5
#scores = evaluate_algorithm(train, 5, n_folds, num_neighbors)