# import os, sys
# import warnings # warnings.filterwarnings("ignore")
from scipy.stats import norm
from argparse import Namespace

import numpy as np
"""
1D Kalman Filter
"""
class KF1D:
    def __init__(self, args:Namespace):
        super().__init__()
        print(f"KF1D: {args}")
        self.args = args
        self.dim = 1

        # 噪声参数
        # 这里取对数是为了进行非负约束. 标准差一定要是正值, 可以通过exp{sig_log} > 0来实现非负约束
        self.state_noise_sig_log = np.ones((self.dim, self.dim))*args.state_noise_init
        self.obs_noise_sig_log = np.ones((self.dim, self.dim))*args.obs_noise_init
        # inoise即系统固有的噪声, 防止过拟合, 同时保持一定的系统适应性(不过分信任预测值)
        self.state_inoise_sig_log = np.ones((self.dim, self.dim))*args.state_noise_init # 手动设置一个下界
        
        # kf系统模型参数
        self.trans_model = np.array([[1.0]]) # F
        self.obs_model = np.array([[1.0]]) # H

        self.score_max = args.score_max # 非有效值, 单纯初始化


    def encode(self, obs):
        '''
        将观察值转换为1*1的二阶矩阵。
        
        eg. obs=1, return [[1]]
        
        :param obs: 观察值
        '''
        return np.ones((self.dim, self.dim))*obs
    
        
    def init_state(self, obs):
        '''
        初始化状态。使用第一次的观测值来初始化。记录第一次观测值，并为状态均值和状态方差赋值，分别为观察值和单位矩阵
        
        :param obs: 第一次的观察值
        '''
        # 冗余的变量, 删除
        # self.obs = obs
        self.state_mu = self.encode(obs)
        self.state_cov = np.ones((self.dim, self.dim)) # 非有效值, 单纯初始化
        
        
    def predict_nextstate(self):
        '''
        预测的当前值。
        使用的公式为
        mu|next = trans_model * mu|now
        cov|next = trans_model * cov|now * T(trans_model) + {state_noise_sig}_2
        '''
        state_mu_pred = self.trans_model @ self.state_mu
        state_cov_pred = self.trans_model @ self.state_cov @ self.trans_model.T + np.power(np.exp(self.state_noise_sig_log), 2)

        # print(f'[KF, pred] mu = {state_mu_pred.item():.4f}, sig = {state_cov_pred.sqrt().item():.4f}')

        return {'mu': state_mu_pred, 'cov': state_cov_pred}

    
    def predict(self):
        '''
        注意: 这里不是观测值
        将预测值从状态空间映射到观测空间
        z = H * x + v
        '''
        # 预测值
        state_pred = self.predict_nextstate()
        state_mu_pred = state_pred['mu']
        state_cov_pred = state_pred['cov']
        
        # 转换到观测空间
        obs_mu_pred = self.obs_model @ state_mu_pred
        obs_cov_pred = self.obs_model @ state_cov_pred @ self.obs_model.T + np.power(np.exp(self.obs_noise_sig_log), 2)
        
        return {'mu': obs_mu_pred, 'cov': obs_cov_pred}


    def update_score_max(self):
        '''
        记录概率密度函数(中心)最大值的两倍
        '''
        pred = self.predict()
        mu = np.squeeze(pred['mu'])
        sig = np.squeeze(np.sqrt(pred['cov']))

        self.score_max = norm.pdf(mu, loc=mu, scale=sig) * 2 # 这里为什么要乘以2呀, 提供安全余量
        #print(f'score_max = {self.score_max:.4f}, mu = {mu:.4f}, sig = {sig:.4f}')
        
    
    def update_noise(self, obs):
        '''
        更新预测模型
        求出当前时刻的预测值和观察值差的分布, 梯度下降进行学习 
        '''
        obs = self.encode(obs)
        assert(obs.shape == (self.dim, self.dim))

        # # update noise parameters
        # y = obs
        # mu = self.state_mu
        # cov = self.state_cov
        # wbar = self.state_noise_sig_log
        # vbar = self.obs_noise_sig_log
        # w = np.exp(wbar)
        # v = np.exp(vbar)

        
        # # 预测偏差的指标. 假设也服从高斯分布
        # xi = np.sqrt(cov + np.power(np.exp(wbar), 2) + np.power(np.exp(vbar), 2))
        # diff = y - mu
        obs_pred = self.predict()
        obs_mu_pred = obs_pred['mu']
        obs_cov_pred = obs_pred['cov']
        
        diff = obs - obs_mu_pred # diff~(0, xi)
        xi = np.sqrt(obs_cov_pred) # xi = xi = np.sqrt(cov + np.power(np.exp(wbar), 2) + np.power(np.exp(vbar), 2))b
        
        # 在线学习噪声参数 
        # L = -log(p) p服从正态分布 忽略常数项
        # grad_state_noise_sig_log = ( 1/xi - np.power(diff, 2)/np.power(xi, 3) ) * ( w/xi ) * np.exp(wbar) 
        # grad_obs_noise_sig_log = ( 1/xi - np.power(diff, 2)/np.power(xi, 3) ) * ( v/xi ) * np.exp(vbar)
        Q = np.power(np.exp(self.state_noise_sig_log), 2)
        R = np.power(np.exp(self.obs_noise_sig_log), 2)
        dL_dxi = 1/xi - np.power(diff, 2) / np.power(xi, 3)
        grad_state_noise_sig_log = dL_dxi * (Q / xi) # dL_dxi * dxi_dwar
        grad_obs_noise_sig_log = dL_dxi * (R / xi)

        self.state_noise_sig_log = self.state_noise_sig_log - self.args.lr * grad_state_noise_sig_log
        self.obs_noise_sig_log = self.obs_noise_sig_log - self.args.lr * grad_obs_noise_sig_log

        # noise clipping
        self.state_noise_sig_log = np.maximum(self.state_noise_sig_log, self.state_inoise_sig_log)
        
        
        
    def update_state(self, state_pred, obs):
        '''
        结合预测值和观测值, 更新状态并返回更新的状态值
        
        :param state_pred: 输入的预测值
        :param obs: 输入的观测值
        '''
        obs = self.encode(obs)
        assert(obs.shape == (self.dim, self.dim))

        # state prediction
        state_mu_pred, state_cov_pred = state_pred['mu'], state_pred['cov']

        # update a state
        obs_inno = obs - self.obs_model @ state_mu_pred
        cov_inno = self.obs_model @ state_cov_pred @ self.obs_model.T + np.power(np.exp(self.obs_noise_sig_log), 2)
        # 计算卡尔曼增益
        opt_kalman_gain = state_cov_pred @ self.obs_model.T @ np.linalg.inv(cov_inno)

        # 综合预测值和观测值
        self.state_mu = state_mu_pred + opt_kalman_gain @ obs_inno
        self.state_cov = (np.eye(self.dim, self.dim) - opt_kalman_gain @ self.obs_model) @ state_cov_pred
        
        return {'mu': self.state_mu, 'cov': self.state_cov}


    def update(self, obs, update_noise=True, update_state=True, update_max=True):
        '''
        这里的可以看到时序
        先先在线学习, 再更新状态
        '''
        # update noise variance
        if update_noise:
            self.update_noise(obs)
        
        # update the current state
        if update_state:
            state_pred = self.predict_nextstate()
            state_updated = self.update_state(state_pred, obs)

        # update the max score
        if update_max:
            self.update_score_max()

    
    
    def score(self, obs):
        obs = self.encode(obs)
        # score on the predicted observation
        obs_pred = self.predict()
        assert((obs_pred['mu'].shape == (self.dim, self.dim)) and (obs_pred['cov'].shape == (self.dim, self.dim)))
        score = norm.pdf(np.squeeze(obs), loc=np.squeeze(obs_pred['mu']), scale=np.sqrt(np.squeeze(obs_pred['cov'])))
        score = score / self.score_max # 非归一化
        return score


    def superlevelset(self, t):
        '''
        求解 score(x) >= t
        用于反代出对应的概率密度值

        :param t: 给定的阈值
        '''
        # 反归一下出概率密度
        t = t * self.score_max
        
        # 概率密度为零, 整个空间都满足
        if t == 0:
            interval = [-np.inf, np.inf]
            return interval

        obs_pred = self.predict()
        mu = np.squeeze(obs_pred['mu'])
        sig = np.sqrt(np.squeeze(obs_pred['cov']))
        assert(sig > 0)
        # score(x) >= t, 这里求出的c是偏离均值多少个标准差
        c = - 2*np.log(t) - 2*np.log(sig) - np.log(2*np.pi) + 1e-8 # avoid numerical error
        # an empty prediction set
        if c <= 0:
            # return an invalid interval for the empty set
            interval = [np.inf, -np.inf]            
        else:
            interval = [mu - sig * np.sqrt(c), mu + sig * np.sqrt(c)]
        assert not any(np.isnan(interval)), f't = {t}, mu = {mu}, sig = {sig}, c = {c}'
        return interval