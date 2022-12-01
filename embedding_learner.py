# -*- coding:utf-8 -*-

import numpy as np
import random, time, traceback, itertools
import threading, multiprocessing
import process_pool as pp
import numpy.linalg as LA
import random


class Embedding_Learner:
	"""一个使用Adam学习embedding的类"""
	def __init__(self, d = 50, gamma = 1, iters = 4e4, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, \
	topic_batch_size = 1024, user_batch_size = 256, epsilon = 1e-8,  margin = 0.1, lambda_ = 1e-5):
		self.d = d #embedding 维数
		self.gamma = gamma
		self.iters = iters
		self.alpha = alpha
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.topic_batch_size = topic_batch_size
		self.user_batch_size = user_batch_size
		self.epsilon = epsilon
		self.margin = margin
		self.lambda_ = lambda_

	#数据初始化
	def __dataInit(self):
		row = self.num_topic
		col = self.d
		domain = 0.01
		self.v = np.zeros((row, col,))
		for i in range(row):
			for j in range(col):
				self.v[i][j] = domain * random.random()

		self.c = - domain * random.random() - 10

		self.grad_v = np.zeros((row, col))
		self.grad_c = 0

		self.m_v = np.zeros((row, col,))
		self.v_v = np.zeros((row, col,))

		self.m_v_d = np.zeros((row, col,))
		self.v_v_d = np.zeros((row, col,))

		self.m_c = 0
		self.v_c = 0

		self.beta_1_pow_cv = 1
		self.beta_2_pow_cv = 1

		self.beta_1_pow_c = 1
		self.beta_2_pow_c = 1

		self.beta_1_pow_dv = 1
		self.beta_2_pow_dv = 1

		if self.gamma != 0:
			self.num_sample = len(self.topic_samples)

			self.observed_samples = []
			self.noise_samples = []
			self.observed_probabilities = []
			self.noise_probabilities = []
			for i in range(self.num_sample):
				sample_tuple = self.topic_samples[i]
				samples = sample_tuple.split(':')
				for j,sample in enumerate(samples):
					sample = sample.split(';')
					sample_topic = sample[0].split(',')
					for k in range(len(sample_topic)):
						sample_topic[k] = int(sample_topic[k])

					if j == 0:
						self.observed_samples.append(sample_topic)
						self.observed_probabilities.append(float(sample[1]))
					else:
						self.noise_samples.append(sample_topic)
						self.noise_probabilities.append(float(sample[1]))

			self.k = int(len(self.noise_samples) / self.num_sample)

	#计算话题对出现概率
	def __theta_probability(self, sample, tag):
		i,j = sample[0], sample[1]
		compatibility = np.dot(self.v[i], self.v[j])
		if tag == 1:
			score = compatibility / self.covMat[i][j] + self.c
		else:
			score = compatibility + self.c

		probability = np.exp(score)

		return probability

	#计算正样本梯度
	def __observed_gradient(self, observed_samples, p):
		grad_c = 0
		JC_grad_v = np.zeros(self.v.shape)
		num_sample = len(observed_samples)
		for i, observed_sample in enumerate(observed_samples):
			p_theta_a = self.__theta_probability(observed_sample, 1)
			p_noise_a = p[i] ** 0.75
			pre_part = self.k * p_noise_a + p_theta_a
			if pre_part == float('inf'):
				pre_part = self.k * p_noise_a
			elif pre_part != 0:
				pre_part = self.k * p_noise_a / pre_part

			grad_c += pre_part
			cov = self.covMat[observed_sample[0]][observed_sample[1]]
			JC_grad_v[observed_sample[0]] += pre_part * self.v[observed_sample[1]] / cov
			JC_grad_v[observed_sample[1]] += pre_part * self.v[observed_sample[0]] / cov

		return [grad_c, JC_grad_v]

	#计算负样本梯度
	def __noise_gradient(self, noise_samples, p):
		grad_c = 0
		JC_grad_v = np.zeros(self.v.shape)
		num_sample = len(noise_samples)
		for i, noise_sample in enumerate(noise_samples):
			p_theta_n = self.__theta_probability(noise_sample, 1)
			p_noise_n = p[i] ** 0.75

			pre_part = self.k * p_noise_n + p_theta_n
			JC_delta = np.log(self.k * p_noise_n / pre_part)
			pre_part = p_theta_n / pre_part

			cov = self.covMat[noise_sample[0]][noise_sample[1]]
			grad_c -= pre_part
			JC_grad_v[noise_sample[0]] -= pre_part * self.v[noise_sample[1]] / cov
			JC_grad_v[noise_sample[1]] -= pre_part * self.v[noise_sample[0]] / cov

		return [grad_c, JC_grad_v]

	#计算用户对梯度
	def __user_gradient(self, pair_samples):
		row = self.num_topic

		i = pair_samples[0]

		JD_grad_v = np.zeros((row,row))

		p1 = np.dot(self.disMat1, self.v)
		p2 = np.dot(self.disMat2, self.v)

		sii = np.dot(p1[i], p2[i])

		candidates = []
		for j in pair_samples[1:]:
			sij = np.dot(p1[i], p2[j])
			JD = sii - sij
			#如果损失函数小于边界值，用于更新梯度
			if JD < self.margin:
				candidates.append(j)

		num_can = len(candidates)

		JD_grad_v += np.dot(self.disMat1[i].reshape(-1,1), (num_can * self.disMat2[i] - np.sum(self.disMat2[candidates], axis=0)).reshape(1, -1))
		JD_grad_v += np.dot((num_can * self.disMat2[i] - np.sum(self.disMat2[candidates], axis=0)).reshape(-1,1), self.disMat1[i].reshape(1, -1))

		JD_grad_v = np.dot(JD_grad_v, self.v)
		JD_grad_v = JD_grad_v - self.lambda_ * self.v

		return JD_grad_v

	#计算梯度
	def __gradient(self):
		row, col = self.num_topic, self.d

		self.JC_grad_v = np.zeros((row, col))
		self.JD_grad_v = np.zeros((row, col))
		self.grad_c = 0

		rand = random.random()
		if rand < self.gamma:
			self.tag = 0
			#采样话题对样本
			observed_samples, noise_samples, observed_probabilities, noise_probabilities = self.__cooccur_sample(self.topic_batch_size)
			#使用正样本更新梯度
			grad_c, JC_grad_v = self.__observed_gradient(observed_samples, observed_probabilities)
			self.JC_grad_v += JC_grad_v
			self.grad_c += grad_c

			#使用负样本更新梯度
			grad_c, JC_grad_v = self.__noise_gradient(noise_samples, noise_probabilities)
			self.JC_grad_v += JC_grad_v
			self.grad_c += grad_c
		else:
			self.tag = 1
			#采样用户对样本
			pair_samples = self.__user_pair_sample()

			#计算用户对样本的梯度
			self.JD_grad_v = self.__user_gradient(pair_samples)

	#采样话题对
	def __cooccur_sample(self, batch_size):
		num_samples = self.num_sample

		observed_samples = []
		noise_samples = []
		observed_probabilities = []
		noise_probabilities = []
		if batch_size == 0:
			return self.observed_samples, self.noise_samples, self.observed_probabilities, \
			self.noise_probabilities
		else:
			sampled_index = random.sample(range(num_samples), batch_size)
			for i in sampled_index:
				observed_samples.append(self.observed_samples[i])
				observed_probabilities.append(self.observed_probabilities[i])
				index_h = int(i * self.k)
				index_t = int(index_h + self.k)
				noise_samples.extend(self.noise_samples[index_h : index_t])
				noise_probabilities.extend(self.noise_probabilities[index_h : index_t])

		return observed_samples, noise_samples, observed_probabilities, \
		noise_probabilities

	#采样用户对
	def __user_pair_sample(self):
		num_user = self.num_user
		target_user = random.sample(range(num_user), 1)
		candidates = [i for i in range(num_user)]
		candidates.pop(target_user[0])
		# target_user.extend(random.sample(candidates, self.user_batch_size))
		target_user.extend(candidates)

		return target_user

	#基于Adam更新参数
	def __adamUpdate(self):
		if self.tag == 0:
			self.m_v = self.beta_1 * self.m_v + (1 - self.beta_1) * self.JC_grad_v
			self.v_v = self.beta_2 * self.v_v + (1 - self.beta_2) * (self.JC_grad_v ** 2)

			self.beta_1_pow_cv *= self.beta_1
			self.m_bar_v = self.m_v / (1 - self.beta_1_pow_cv)

			self.beta_2_pow_cv *= self.beta_2
			self.v_bar_v = self.v_v / (1 - self.beta_2_pow_cv)
			self.grad_v = self.m_bar_v / (np.sqrt(self.v_bar_v) + self.epsilon)
		else:
			self.m_v_d = self.beta_1 * self.m_v_d + (1 - self.beta_1) * self.JD_grad_v
			self.v_v_d = self.beta_2 * self.v_v_d + (1 - self.beta_2) * (self.JD_grad_v ** 2)

			self.beta_1_pow_dv *= self.beta_1
			self.m_bar_v_d = self.m_v_d / (1 - self.beta_1_pow_dv)

			self.beta_2_pow_dv *= self.beta_2
			self.v_bar_v_d = self.v_v_d / (1 - self.beta_2_pow_dv)

			self.grad_v = self.m_bar_v_d / (np.sqrt(self.v_bar_v_d) + self.epsilon)

		self.delta_v = self.alpha * self.grad_v

		self.m_c = self.beta_1 * self.m_c + (1 - self.beta_1) * self.grad_c
		self.v_c = self.beta_2 * self.v_c + (1 - self.beta_2) * (self.grad_c ** 2)

		self.beta_1_pow_c *= self.beta_1
		self.m_bar_c = self.m_c / (1 - self.beta_1_pow_c)

		self.beta_2_pow_c *= self.beta_2
		self.v_bar_c = self.v_c / (1 - self.beta_2_pow_c)

		self.delta_c = self.alpha * self.m_bar_c / (np.sqrt(self.v_bar_c) + self.epsilon)

		self.v = self.v + self.delta_v
		self.c = self.c + self.delta_c

	#计算话题兼容性损失目标
	def __topic_loss(self):
		observed_samples, noise_samples, observed_probabilities, noise_probabilities = self.__cooccur_sample(0)
		def observed_process(observed_samples, p):
			JC = 0
			num_sample = len(observed_samples)
			for p_noise_a, observed_sample in zip(p,observed_samples):
				p_theta_a = self.__theta_probability(observed_sample, 1)
				pre_part = self.k * p_noise_a + p_theta_a
				if pre_part == float('inf'):
					JC_delta = 0
				elif pre_part != 0:
					JC_delta = np.log(p_theta_a / pre_part)

				JC += JC_delta

			return JC
		def noise_process(noise_samples, p):
			JC = 0
			for p_noise_n, noise_sample in zip(p, noise_samples):
				p_theta_n = self.__theta_probability(noise_sample, 1)
				pre_part = self.k * p_noise_n + p_theta_n
				JC_delta = np.log(self.k * p_noise_n / pre_part)
				JC += JC_delta

			return JC

		return -observed_process(observed_samples, observed_probabilities) - noise_process(noise_samples, noise_probabilities)

	#计算用户一致性和差异性目标
	def __user_loss(self):
		JD = 0
		p1 = np.dot(self.disMat1, self.v)
		p2 = np.dot(self.disMat2, self.v)
		for i in range(self.num_user):
			sii = np.dot(p1[i], p2[i])
			sij = np.dot(p1[i].reshape(1,-1), p2.T)

			for j in range(self.num_user):
				delta = sii - sij[0][j]
				JD += min(self.margin - delta, 0)

		return JD

	#训练函数入口
	def fit(self, disMat1, disMat2, covMat, topic_samples):
		self.num_user, self.num_topic = disMat2.shape[0], disMat1.shape[1]
		self.disMat1 = disMat1
		self.disMat2 = disMat2
		self.covMat = covMat
		self.topic_samples = topic_samples

		#初始化参数
		self.__dataInit()

		iterations = 0
		start = time.time()
		while iterations < self.iters:
			#计算梯度
			self.__gradient()
			# self.grad_c = -1024
			# print(self.grad_c)
			#使用Adam更新参数
			self.__adamUpdate()
			# print(self.c)



			JU = 2e31
			#计算目标函数值
			if iterations % 1000 == 0:
				JC, JD = 0, 0
				if self.gamma != 0:
					JC = self.__topic_loss()
				if self.gamma != 1:
					JD = self.__user_loss()

				JU = self.gamma * JC + (1 - self.gamma) * JD
				print('iterations<%d>  objective:%.2f'%(iterations, JU))
				print(time.time() - start)
				start = time.time()
			iterations += 1

		print('learning process finished')
