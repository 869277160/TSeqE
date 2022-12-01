#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from CONFIG import *
from scipy.spatial import distance as dist
from scipy import stats

def KLbased(previous, post):
	mean = (previous + post) / 2
	distance = stats.entropy(previous, mean) + stats.entropy(post, mean)

	return distance

def distance(mat1, mat2, metric):
	num_user = mat1.shape[0]
	dist_matrix = np.zeros((num_user, num_user))
	if metric == 'cosine':
		for i in range(num_user):
			for j in range(num_user):
				dist_matrix[i][j] = dist.cosine(mat1[i], mat2[j])

		return dist_matrix

	elif metric == 'euclidean':
		for i in range(num_user):
			for j in range(num_user):
				dist_matrix[i][j] = dist.euclidean(mat1[i], mat2[j])

		return dist_matrix

	else:
		for i in range(num_user):
			for j in range(num_user):
				dist_matrix[i][j] = KLbased(mat1[i], mat2[j])

		return dist_matrix

def accuracy(dist_mat, k = 1):
	num_user = dist_mat.shape[0]
	correct_num = 0

	count = 0
	for i in range(num_user):
		num_wrong = 0
		for j in range(num_user):
			if dist_mat[i][j] < dist_mat[i][i]:
				num_wrong += 1

		if num_wrong < k:
			correct_num += 1

	accuracy = correct_num / num_user
	return accuracy


if __name__ == '__main__':
	disMat1_path = 'ml_data/distribution_matrix_previous_0.7_50.txt'
	disMat2_path = 'ml_data/distribution_matrix_post_0.7_50.txt'

	disMat1 = np.loadtxt(disMat1_path)
	disMat2 = np.loadtxt(disMat2_path)
	for i in range(10000, 60000, 10000):
		print('start iterations: %d'%(i))
		start = time.time()
		path = 'ml_data/embedding/embedding_{}.txt'.format(i)
		v = np.loadtxt(path)

		test_pattern1 = np.dot(disMat1, v)
		test_pattern2 = np.dot(disMat2, v)

		for j, metric in enumerate(ML_DISTANCE_METRIC):
			distance_matrix = distance(test_pattern1, test_pattern2, metric)
			acc = accuracy(distance_matrix, TOP_K)
			print('accuracy: %.3f'%acc)
