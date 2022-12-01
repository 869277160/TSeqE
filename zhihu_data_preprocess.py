#!/usr/bin/python3
# -*- coding:utf-8 -*-

from etc.utils.CONFIG import *
import numpy as np
import random

'''
根据被选择的话题构造”话题-索引”字典
PARAMETERS: 话题出现次数阈值
INPUT: topic_selected_(threshold).txt
OUTPUT: ”话题-索引”字典
'''
def makeTopicDict(topic_threshold):
	topic_path = ZHIHU_DATA_PATH+'topic_selected_{0}.txt'.format(topic_threshold)
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
	user_path = ZHIHU_DATA_PATH+'user_id_{0}.txt'.format(user_threshold)
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
计算topic出现的次数， 并挑选出次数超过阈值的topic
INPUT: topic_id.txt, topic_co-occur.txt
OUTPUT: topic_selected_(threshold).txt
'''
@log('Topic Selection')
def topic_selection():
	topic_dict = {}
	topic_path = ZHIHU_DATA_PATH+'topic_id.txt'

	#构造topic全集的字典
	with open(topic_path, 'r') as topics:
		for topic in topics:
			topic = topic.strip('\r\n')
			topic_dict[topic] = 0

	#分割topic对方法
	def split_topic(co_occur):
		co_occur = co_occur.split(',')
		topic_list = []
		for topic in co_occur:
			if topic.startswith(' '):
				topic_list[-1] = '{0},{1}'.format(topic_list[-1],topic)
			else:
				topic_list.append(topic)

		return topic_list

	#topic计数
	co_path = ZHIHU_DATA_PATH+'topic_co-occur.txt'
	with open(co_path, 'r') as co_occurs:
		for co_occur in co_occurs:
			co_occur = co_occur.strip('\r\n')
			topics = co_occur.split(',')

			if len(topics) > 1:
				for topic in topics:
					try:
						topic_dict[topic] += 1
					except Exception as e:
						print(topic)
						raise e

	#选择出现次数大于阈值的topic写入文件
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		print('topic threshold: %d'%topic_threshold)
		out_path = ZHIHU_DATA_PATH+'topic_selected_{0}.txt'.format(topic_threshold)
		with open(out_path, 'w') as out_file:
			for key in topic_dict:
				if topic_dict[key] >= topic_threshold:
					out_file.write('{0}\n'.format(key))

'''
从话题共同出现组中去除未被选择的话题
INPUT:topic_selected_(threshold).txt, topic_co-occur.txt
OUTPUT:cooccur_selected_(threshold).txt
'''
@log('Unselected Topic Removal')
def unselected_topic_removal():
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		print('topic threshold: %d'%topic_threshold)
		#构造被选择的话题字典
		topic_path = ZHIHU_DATA_PATH+'topic_selected_{0}.txt'.format(topic_threshold)
		topic_dict = {}
		with open(topic_path, 'r') as topics:
			for topic in topics:
				topic = topic.strip('\r\n')
				topic_dict[topic] = None

		#”话题-索引”字典
		topic_index = makeTopicDict(topic_threshold)

		#将只包含被选择话题的话题组写入新文件
		cooccur_path = ZHIHU_DATA_PATH+'topic_co-occur.txt'
		out_path = ZHIHU_DATA_PATH+'cooccur_selected_{0}.txt'.format(topic_threshold)
		with open(cooccur_path, 'r') as origins, open(out_path, 'w') as changed:
			for origin in origins:
				origin_list = origin.strip('\r\n').split(',')
				new_list = []
				for topic in origin_list:
					if topic in topic_dict:
						new_list.append(topic)

				if len(new_list) > 1:
					origin_str = ','.join(new_list)
					changed.write(origin_str + '\n')


'''
计算被选择话题的出现频次
INPUT: cooccur_selected_(threshold).txt
OUTPUT: topic_distribution_(topic_threshold).txt
'''
@log('Calculating Global Topic Distribution')
def global_topic_distribution():
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		print('topic threshold:%d'%topic_threshold)

		topic_dict = makeTopicDict(topic_threshold)
		num_topic = len(topic_dict)
		topic_distribution = np.zeros((num_topic,))
		# appearance_probability = np.zeros((num_topic,))

		#统计话题频率
		normal_path = ZHIHU_DATA_PATH+'cooccur_selected_{0}.txt'.format(topic_threshold)
		with open(normal_path, 'r') as cooccurs:
			for cooccur in cooccurs:
				cooccur = cooccur.strip('\n')
				topics = cooccur.split(',')
				for topic in topics:
					index = topic_dict[topic]
					topic_distribution[index] += 1

		out_path = ZHIHU_DATA_PATH+'topic_distribution_{0}.txt'.format(topic_threshold)
		np.savetxt(out_path, topic_distribution, fmt = '%d')

'''
计算前后两个时段以及全时段的用户-话题分布矩阵
INPUT: user_topic_previous.txt, user_topic_post.txt
OUTPUT: distribution_matrix_previous_(topic_threshold)_(user_threshold).txt, distribution_matrix_post_(topic_threshold)_(user_threshold).txt， distribution_matrix_(topic_threshold)_(user_threshold).txt
'''
@log('Calculating User-Topic Distribution Matrix')
def calDistributionMatrix():
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		for user_threshold in ZHIHU_USER_THRESHOLD:
			print('topic_threshold:%d   user_threshold:%d'%(topic_threshold, user_threshold))
			#构造话题索引字典
			topic_dict = makeTopicDict(topic_threshold)
			num_topic = len(topic_dict)

			#构造用户索引字典
			user_dict = makeUserDict(user_threshold)
			num_user = len(user_dict)

			source_set = ['previous', 'post']

			D = np.zeros((2 * num_user, num_topic,))

			#统计在各个时段每个用户在每个话题上的频次
			for i, source in enumerate(source_set):
				D_u = np.zeros((num_user, num_topic,))
				activity_path = ZHIHU_DATA_PATH+'user_topic_{0}.txt'.format(source)
				with open(activity_path, 'r') as activities:
					for activity in activities:
						activity = activity.strip('\r\n').split('\t')
						if activity[0] not in user_dict:
							continue
						user_index = user_dict[activity[0]]
						for topic in activity[1].split(','):
							if topic not in topic_dict:
								continue
							topic_index = topic_dict[topic]
							D_u[user_index][topic_index] += 1
							D[user_index + i * num_user][topic_index] += 1

				#标准化频次
				for i in range(num_user):
					if sum(D_u[i]) != 0:
						D_u[i] = D_u[i] / sum(D_u[i])
					else:
						D_u[i] = D_u[i] + (1 / len(D_u[i]))

				out_path = ZHIHU_DATA_PATH+'distribution_matrix_{0}_{1}_{2}.txt'.format(source, topic_threshold, user_threshold)
				np.savetxt(out_path, D_u, fmt = '%.18f')

			D_path = ZHIHU_DATA_PATH+'distribution_matrix_{0}_{1}.txt'.format(topic_threshold, user_threshold)
			np.savetxt(D_path, D, fmt = '%d')


'''
计算话题的皮尔森相关系数矩阵(经sigmod处理)
INPUT: distribution_matrix_(topic_threshold)_(user_threshold).txt
OUTPUT: covMat_(topic_threshold)_(user_threshold).txt
'''
@log('Calculating Cov-Matrix of Topics')
def calCov():
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		for user_threshold in ZHIHU_USER_THRESHOLD:
			print('topic_threshold:%d   user_threshold:%d'%(topic_threshold, user_threshold))

			D_path = ZHIHU_DATA_PATH+'distribution_matrix_{0}_{1}.txt'.format(topic_threshold, user_threshold)
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

			cov_path = ZHIHU_DATA_PATH+'covMat_{0}_{1}.txt'.format(topic_threshold, user_threshold)

			np.savetxt(cov_path, covMat)

'''
生成优化话题兼容性的样本（一正样本对应k个负样本）
INPUT: distribution_matrix_(topic_threshold)_(user_threshold).txt
OUTPUT: covMat_(topic_threshold)_(user_threshold).txt
'''
@log('Topic Train Samples Generation')
def sample_generation():
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		for k in ZHIHU_TOPIC_SAMPLE_K:
			print('topic_threshold:%d   k:%d'%(topic_threshold, k))
			topic_dict = makeTopicDict(topic_threshold)
			num_topic = len(topic_dict)
			p_path = ZHIHU_DATA_PATH+'topic_distribution_{0}.txt'.format(topic_threshold)
			p = np.loadtxt(p_path)

			#将频次标准化成概率分布
			p = p / np.sum(p)

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

			normal_path = ZHIHU_DATA_PATH+'cooccur_selected_{0}.txt'.format(topic_threshold)
			out_path = ZHIHU_DATA_PATH+'topic_train_samples_{0}_{1}.txt'.format(topic_threshold, k)
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

'''
对每个用户采集30个行为样本，每个样本由30条行为记录组成
INPUT: user_topic_previous.txt, user_topic_post.txt
OUTPUT: behavior_samples_previous_(topic_threshold)_(user_threshold).txt, behavior_samples_post_(topic_threshold)_(user_threshold).txt
'''
@log('Behavior Sampling')
def behavior_sample(k, b):
	for topic_threshold in ZHIHU_TOPIC_THRESHOLD:
		for user_threshold in ZHIHU_USER_THRESHOLD:
			print('topic_threshold:%d   user_threshold:%d'%(topic_threshold, user_threshold))
			#构造话题索引字典
			topic_dict = makeTopicDict(topic_threshold)
			num_topic = len(topic_dict)

			#构造用户索引字典
			user_dict = makeUserDict(user_threshold)
			num_user = len(user_dict)

			source_set = ['previous', 'post']

			for i, source in enumerate(source_set):
				act_list = [[] for _ in range(num_user)]
				activity_path = ZHIHU_DATA_PATH+'user_topic_{0}.txt'.format(source)
				with open(activity_path, 'r') as activities:
					for activity in activities:
						activity = activity.strip('\r\n').split('\t')
						if activity[0] not in user_dict:
							continue
						user_index = user_dict[activity[0]]
						topic_list = []
						for topic in activity[1].split(','):
							if topic in topic_dict:
								topic_list.append(topic_dict[topic])
						if topic_list != []:
							act_list[user_index].append(topic_list)

				D = np.zeros((num_user, b, num_topic))
				for i in range(num_user):
					num_act = len(act_list[i])
					for j in range(b):
						sampled_index = random.sample(range(num_act), k)
						for act in [act_list[i][index] for index in sampled_index]:
							for topic in act:
								D[i,j,topic] += 1
							D[i,j,:] = D[i,j,:] / np.sum(D[i,j,:])

				D = D.reshape((num_user*b, num_topic))

				sampled_path = ZHIHU_DATA_PATH+'sampled_behavior_{0}.txt'.format(source)
				np.savetxt(sampled_path, D, fmt = '%.5f')

'''
统计用户的行为次数
INPUT: user_topic_previous.txt, user_topic_post.txt
OUTPUT: behavior_count_previous_(topic_threshold)_(user_threshold).txt, behavior_count_post_(topic_threshold)_(user_threshold).txt
'''
def countBehavior():
	for topic_threshold in [200]:
		for user_threshold in [200]:
			print('topic_threshold:%d   user_threshold:%d'%(topic_threshold, user_threshold))
			#构造话题索引字典
			topic_dict = makeTopicDict(topic_threshold)
			num_topic = len(topic_dict)

			#构造用户索引字典
			user_dict = makeUserDict(user_threshold)
			num_user = len(user_dict)

			source_set = ['previous', 'post']

			D = np.zeros((2 * num_user, num_topic,))

			#统计在各个时段每个用户在每个话题上的频次
			for i, source in enumerate(source_set):
				D_u = np.zeros((num_user,))
				activity_path = ZHIHU_DATA_PATH+'user_topic_{0}.txt'.format(source)
				with open(activity_path, 'r') as activities:
					for activity in activities:
						activity = activity.strip('\r\n').split('\t')
						if activity[0] not in user_dict:
							continue
						user_index = user_dict[activity[0]]
						D_u[user_index] += 1


				out_path = ZHIHU_DATA_PATH+'behavior_count_{0}_{1}_{2}.txt'.format(source, topic_threshold, user_threshold)
				np.savetxt(out_path, D_u, fmt = '%d')


if __name__ == '__main__':
	topic_selection()
	unselected_topic_removal()
	global_topic_distribution()
	calDistributionMatrix()
	calCov()
	sample_generation()
	countBehavior()
