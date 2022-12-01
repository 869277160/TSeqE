# -*- coding: utf-8 -*-

#知乎数据集配置
ZHIHU_TOPIC_THRESHOLD = [200] #知乎话题出现次数阈值，列表形式
ZHIHU_USER_THRESHOLD = [100] #用户行为数阈值，列表形式
ZHIHU_TOPIC_SAMPLE_K = [1] #对每一个正样本采样负样本的个数
ZHIHU_DISTANCE_METRIC = ['cosine']

#知乎数据集配置
ML_RELEVANCE_THRESHOLD = [0.7]
ML_USER_THRESHOLD = [50] #用户行为数阈值，列表形式
ML_TOPIC_SAMPLE_K = [1] #对每一个正样本采样负样本的个数
ML_DISTANCE_METRIC = ['cosine']

D = 50
CROSS_VALID_K = 5
TOP_K = 1
