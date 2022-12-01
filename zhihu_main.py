#!/usr/bin/python3
# -*- coding:utf-8 -*-

from embedding_learner import *
from CONFIG import *
import validation
import numpy as np
import random

def rand_div_list(ls, n):
	ls_len = len(ls)
	j = ls_len // n
	k = ls_len % n

	ls_return = []

	for i in range(n):
		ls_part = random.sample(ls, j)
		ls_return.append(ls_part)
		ls = list(set(ls) - set(ls_part))

	ls_return[-1].extend(ls)

	return ls_return


def dataReader(topic_threshold, user_threshold):
	disMat1_path = 'zhihu_data/distribution_matrix_previous_{0}_{1}.txt'.format(topic_threshold, user_threshold)
	disMat2_path = 'zhihu_data/distribution_matrix_post_{0}_{1}.txt'.format(topic_threshold, user_threshold)
	covMat_path = 'zhihu_data/covMat_{0}_{1}.txt'.format(topic_threshold, user_threshold)

	disMat1 = np.loadtxt(disMat1_path)
	disMat2 = np.loadtxt(disMat2_path)
	covMat = np.loadtxt(covMat_path)

	topic_sample_path = 'zhihu_data/topic_train_samples_{0}_1.txt'.format(topic_threshold)
	sample_list = []
	with open(topic_sample_path, 'r') as samples:
		for line in samples:
			line = line.strip('\n')
			sample_list.append(line)

	return disMat1, disMat2, covMat, sample_list


def exp(d, gamma):
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		for user_threshold in ZHIHU_USER_THRESHOLD:
			print('topic_threshold:%d  user_threshold:%d'%(topic_threshold, user_threshold))
			disMat1, disMat2, covMat, topic_samples = dataReader(topic_threshold, user_threshold)
			num_user = disMat1.shape[0]

			user_list = [i for i in range(num_user)]

			cross_valid_list = rand_div_list(user_list, CROSS_VALID_K)

			cross_valid_path = 'zhihu_data/cross_valid/user_partition.txt'
			with open(cross_valid_path, 'w') as cross:
				cross.write(str(cross_valid_list))

			num_metric = len(ZHIHU_DISTANCE_METRIC)
			accuracies = np.zeros((num_metric))

			for i, train_list in enumerate(cross_valid_list):
				seed_path = 'zhihu_data/seed_user/seed.txt'
				with open(seed_path, 'w') as seed_out:
					for j in train_list:
						seed_out.write(str(j) + '\r')

				print('start cross validation iter: %d'%(i+1))
				start = time.time()
				test_list = list(set(user_list) - set(train_list))

				train_disMat1 = disMat1[train_list]
				train_disMat2 = disMat2[train_list]
				test_disMat1 = disMat1[test_list]
				test_disMat2 = disMat2[test_list]

				model = Embedding_Learner(d = 50, gamma = gamma )
				model.fit(train_disMat1, train_disMat2, covMat, topic_samples)
				path = 'zhihu_data/embedding/em_{0}_{1}_{2}_{3}_{4}'.format(topic_threshold, user_threshold, 50, gamma, i+1)
				np.savetxt(path, model.v)

				test_pattern1 = np.dot(test_disMat1, model.v)
				test_pattern2 = np.dot(test_disMat2, model.v)

				for j, metric in enumerate(ZHIHU_DISTANCE_METRIC):
					distance_matrix = validation.distance(test_pattern1, test_pattern2, metric)
					path = 'zhihu_data/distance/em_{0}_{1}_{2}_{3}_{4}_{5}'.format(topic_threshold, user_threshold, d, gamma, i+1, metric)
					np.savetxt(path, distance_matrix)
					accuracy = validation.accuracy(distance_matrix, TOP_K)
					print('accuracy: %.3f'%accuracy)
					accuracies[j] += accuracy

				time_cost = (time.time() - start) / 60
				print('finished cross valid iter: %d    time cost:%d min'%(i+1, time_cost))

				if i == 0:
					break


			result_path = 'zhihu_result.txt'
			with open(result_path, 'a+') as result:
				result.write('topic_threshold:%d  user_threshold:%d\n'%(topic_threshold, user_threshold))
				for i in range(num_metric):
					result.write('%s:%.3f\n'%(ZHIHU_DISTANCE_METRIC[i], accuracies[i]/CROSS_VALID_K))

			print('finished  topic_threshold:%d  user_threshold:%d\n'%(topic_threshold, user_threshold))

if __name__ == '__main__':
	ds = [50]
	gamma = 0
	for d in ds:
		print('D: %.2f'%d)
		result_path = 'zhihu_result.txt'
		with open(result_path, 'a+') as result:
			result.write('d: %.2f \n'%d)

		exp(d, gamma)
