#!/usr/bin/python3
# -*- coding:utf-8 -*-

from etc.utils.CONFIG import *
import numpy as np
import random

'''
根据被选择的话题构造”话题-索引”字典
INPUT: topic_selected.txt
OUTPUT: ”话题-索引”字典
'''
def makeTopicDict():
	topic_path = 'ml_data/topic_selected.txt'
	with open(topic_path, 'r') as topics:
		topic_dict = {}
		for i, topic in enumerate(topics):
			topic = topic.strip("\n")
			topic_dict[topic] = i

	return topic_dict

'''
根据被选择的用户构造”用户-索引”字典
PARAMETERS: 用户行为数阈值
INPUT: user_id_(threshold).txt
OUTPUT: ”用户-索引”字典
'''
def makeUserDict(user_threshold):
	user_path = 'ml_data/user_id_{0}.txt'.format(user_threshold)
	with open(user_path, 'r') as users:

		user_dict = {}

		for i, user in enumerate(users):
			user = user.strip("\r\n")
			user_dict[user] = i

	return user_dict

'''
打印日志装饰器
'''
def log(text):
	def decorator(func):
		def wrapper(*args, **kw):
			print('Start %s'% (text))
			ret = func(*args, **kw)
			print('Finished\n')
			return ret
		return wrapper
	return decorator

'''
根据相关性阈值得到每部电影相关的话题，并抽取出所有的话题对
INPUT: genome_score.txt
OUTPUT: movie_tag_(threshold).txt, cooccur_selected_(threshold).txt
'''
@log('Topic Selection')
def topic_selection():
	for relevance_threshold in ML_RELEVANCE_THRESHOLD:
		print("relevance threshold: %.1f"%relevance_threshold)
		#构造话题索引字典
		topic_dict = makeTopicDict()
		num_topic = len(topic_dict)
		freq_matrix = np.zeros((num_topic, num_topic))

		#生成给定列表中的所有两两话题对
		def generate_pair(topic_list):
			pair_list = []
			for i in range(len(topic_list)):
				for j in range(i + 1, len(topic_list)):
					pair_list.append([topic_list[i], topic_list[j]])
			return pair_list

		score_path = 'ml_data/genome_score.txt'
		movie_tag_path = 'ml_data/movie_tag_{0}.txt'.format(relevance_threshold)
		co_path = 'ml_data/cooccur_selected_{0}.txt'.format(relevance_threshold)

		cur_movieId = '-1'
		tags = []
		relevances = []

		with open(score_path, 'r') as origin_scores, open(movie_tag_path, 'w') as out, open(co_path, 'w') as co_occur:
			for item in origin_scores:
				movieId, tagId, relevance = item.strip('\n').split('\t')
				#读到新电影
				if movieId != cur_movieId:
					if cur_movieId != '-1':
						out.write(cur_movieId + '\t')

						#写出电影相关话题
						sum_rel = 0
						for i in range(len(tags)):
							relevances[i] = float(relevances[i])
							sum_rel += relevances[i]

						for i in range(len(tags)):
							if i != len(tags) - 1:
								out.write(tags[i] + ':%.5f'%(relevances[i] / sum_rel) + ',')
							else:
								out.write(tags[i] + ':%.5f'%(relevances[i] / sum_rel))

						out.write('\n')

						#写出所有话题对
						pair_list = generate_pair(tags)
						for pair in pair_list:
							co_occur.write(pair[0] + ',' + pair[1] + '\n')

					cur_movieId = movieId
					tags = []
					relevances = []

				#选择相关系数大于阈值的话题
				if float(relevance) >= relevance_threshold:
					tags.append(tagId)
					relevances.append(relevance)

			out.write(cur_movieId + '\t')

			sum_rel = 0
			for i in range(len(tags)):
				relevances[i] = float(relevances[i])
				sum_rel += relevances[i]

			for i in range(len(tags)):
				if i != len(tags) - 1:
					out.write(tags[i] + ':%.5f'%(relevances[i] / sum_rel) + ',')
				else:
					out.write(tags[i] + ':%.5f'%(relevances[i] / sum_rel) + '\n')

			pair_list = generate_pair(tags)
			for pair in pair_list:
				co_occur.write(pair[0] + ',' + pair[1] + '\n')


'''
读入话题的出现频次
INPUT: tags.dat
OUTPUT: topic_distribution.txt
'''
@log('Reading Global Topic Distribution')
def global_topic_distribution():
	topic_count = []

	normal_path = 'ml_data/tags.dat'
	with open(normal_path, 'r') as items:
		for item in items:
			tagId, tag, freq = item.strip('\n').split('\t')
			topic_count.append(int(freq))

	topic_distribution = np.array(topic_count)

	out_path = 'ml_data/topic_distribution.txt'
	np.savetxt(out_path, topic_distribution, fmt = '%d')

'''
计算前后两个时段以及全时段的用户-话题分布矩阵
INPUT: user_movie_previous.txt, user_movie_post.txt
OUTPUT: distribution_matrix_previous_(relevance_threshold)_(user_threshold).txt, distribution_matrix_post_(relevance_threshold)_(user_threshold).txt， distribution_matrix_(topic_threshold)_(user_threshold).txt
'''
@log('Calculating User-Topic Distribution Matrix')
def calDistributionMatrix():
	#构造话题索引字典
	topic_dict = makeTopicDict()
	num_topic = len(topic_dict)
	for relevance_threshold in ML_RELEVANCE_THRESHOLD:
		movie_tag_dict = {}
		movie_tag_path = 'ml_data/movie_tag_{0}.txt'.format(relevance_threshold)
		#构造电影-话题字典
		with open(movie_tag_path, 'r') as movie_tags:
			for movie_tag in movie_tags:
				movie_tag = movie_tag.strip('\n')
				movie_id, tags = movie_tag.split('\t')
				movie_tag_dict[movie_id] = [i for i in tags.split(',')]
		for user_threshold in ML_USER_THRESHOLD:
			print('relevance_threshold:%.1f   user_threshold:%d'%(relevance_threshold, user_threshold))

			#构造用户索引字典
			user_dict = makeUserDict(user_threshold)
			num_user = len(user_dict)

			source_set = ['previous', 'post']

			D = np.zeros((2 * num_user, num_topic,))

			#统计在各个时段每个用户在每个话题上的频次
			for i, source in enumerate(source_set):
				D_u = np.zeros((num_user, num_topic,))
				activity_path = 'ml_data/user_movie_{0}.txt'.format(source)
				with open(activity_path, 'r') as activities:
					for activity in activities:
						activity = activity.strip('\r\n').split('\t')
						if activity[0] not in user_dict:
							continue
						user_index = user_dict[activity[0]]
						if activity[1] in movie_tag_dict and movie_tag_dict[activity[1]] != ['']:
							topics = movie_tag_dict[activity[1]]
							for topic in topics:
								topic, weight = topic.split(':')
								topic_index = topic_dict[topic]
								D_u[user_index][topic_index] += float(weight)
								D[user_index + i * num_user][topic_index] += float(weight)

				#标准化频次
				for i in range(num_user):
					if sum(D_u[i]) != 0:
						D_u[i] = D_u[i] / sum(D_u[i])
					else:
						D_u[i] = D_u[i] + (1 / len(D_u[i]))

				out_path = 'ml_data/distribution_matrix_{0}_{1}_{2}.txt'.format(source, relevance_threshold, user_threshold)
				np.savetxt(out_path, D_u, fmt = '%.18f')

			D_path = 'ml_data/distribution_matrix_{0}_{1}.txt'.format(relevance_threshold, user_threshold)
			np.savetxt(D_path, D, fmt = '%d')


'''
计算话题的皮尔森相关系数矩阵(经sigmod处理)
INPUT: distribution_matrix_(relevance_threshold)_(user_threshold).txt
OUTPUT: covMat_(relevance_threshold)_(user_threshold).txt
'''
@log('Calculating Cov-Matrix of Topics')
def calCov():
	for relevance_threshold in ML_RELEVANCE_THRESHOLD:
		for user_threshold in ML_USER_THRESHOLD:
			print('relevance_threshold:%.1f   user_threshold:%d'%(relevance_threshold, user_threshold))

			D_path = 'ml_data/distribution_matrix_{0}_{1}.txt'.format(relevance_threshold, user_threshold)
			D = np.loadtxt(D_path)

			def sigmod(x):
				return 1 / (1 + np.exp(-x))

			#协方差矩阵
			covMat=np.cov(D,rowvar=0)

			#计算各话题的标准差
			num_col = D.shape[1]
			std = np.zeros((num_col))
			for i in range(num_col):
				std[i]= np.std(D[:,i])

			#根据协方差和标准差计算皮尔森相关系数
			for i in range(covMat.shape[0]):
				for j in range(covMat.shape[1]):
					multi_std = std[i] * std[j]
					if multi_std != 0:
						covMat[i][j] = covMat[i][j] / multi_std
					covMat[i][j] = sigmod(covMat[i][j])

			cov_path = 'ml_data/covMat_{0}_{1}.txt'.format(relevance_threshold, user_threshold)

			np.savetxt(cov_path, covMat)

'''
生成优化话题兼容性的样本（一正样本对应k个负样本）
INPUT: distribution_matrix_(relevance_threshold)_(user_threshold).txt
OUTPUT: covMat_(relevance_threshold)_(user_threshold).txt
'''
@log('Topic Train Samples Generation')
def sample_generation():
	#构造话题-索引字典
	topic_dict = makeTopicDict()
	num_topic = len(topic_dict)
	#读入话题分布
	p_path = 'ml_data/topic_distribution.txt'
	p = np.loadtxt(p_path)

	#将频次标准化成概率分布
	p = p / np.sum(p)

	for relevance_threshold in ML_RELEVANCE_THRESHOLD:
		for k in ML_TOPIC_SAMPLE_K:
			print('relevance_threshold:%.1f   k:%d'%(relevance_threshold, k))

			#根据概率抽样
			def random_pick(some_list, probabilities):
				x = random.random()
				cumulative_probability = 0
				for item, probability in zip(some_list, probabilities):
					cumulative_probability += probability
					if x < cumulative_probability:
						break
				return item

			#返回列表中的所有两两话题对
			def generate_pair(topic_list):
				pair_list = []
				for i in range(len(topic_list)):
					for j in range(i + 1, len(topic_list)):
						pair_list.append([topic_list[i], topic_list[j]])
				return pair_list

			#替换正样本中的所有话题
			def replace_all(sample_size, k, p):
				noise_tuple_list = []
				for i in range(k):
					topic_list = []
					probability = 1
					while len(topic_list) != sample_size:
						noise = random_pick(range(len(p)), p)
						if str(noise) not in topic_list:
							topic_list.append(str(noise))
							probability *= p[noise]
					# for j in range(sample_size):
					# 	noise = random_pick(range(len(p)), p)
					# 	topic_list.append(str(noise))
					# 	probability *= p[noise]

					sample_tuple = ','.join(topic_list) + ';' + str(probability)

					noise_tuple_list.append(sample_tuple)

				return noise_tuple_list

			normal_path = 'ml_data/cooccur_selected_{0}.txt'.format(relevance_threshold)
			out_path = 'ml_data/topic_train_samples_{0}_{1}.txt'.format(relevance_threshold, k)
			with open(normal_path, 'r') as cooccurs, open(out_path, 'w') as output:
				i = 0
				for cooccur in cooccurs:
					cooccur = cooccur.strip('\n')
					topics = cooccur.split(',')

					topic_indexes = []
					for topic in topics:
						topic_indexes.append(topic_dict[topic])

					#返回所有话题对
					pair_list = generate_pair(topic_indexes)
					for pair in pair_list:
						sample_list = []
						noise_probability = 1

						for j, topic in enumerate(pair):
							#正样本的噪声概率
							noise_probability *= p[topic]
							pair[j] = str(topic)

						sample_tuple = ','.join(pair) + ';' + str(noise_probability)

						sample_list.append(sample_tuple)
						#加入负样本
						sample_list.extend(replace_all(2, k, p))
						output.write(':'.join(sample_list) + '\n')

					i += 1
					if i % 100000 == 0:
						print("Generated %d samples"%i)


if __name__ == '__main__':
	topic_selection()
	global_topic_distribution()
	calDistributionMatrix()
	calCov()
	sample_generation()
