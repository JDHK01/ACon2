# import os, sys
# import warnings
import numpy as np

# redundant
from argparse import Namespace
from .kf import KF1D

from typing import Optional
class OneSigma:
    # def __init__(self, args: Namespace, model_base:KF1D):
    def __init__(self, args: Optional[Namespace]=None, model_base:Optional[KF1D]=None):
        super().__init__()
        if model_base is None:
            raise ValueError("model_base is required")
        # self.args = args # 根本用不到啊, 全部放到model_base里面了
        self.base = model_base
        self.reset()


    def reset(self):
        self.n_err = 0
        self.n_obs = 0
        self.initialized = False
        self.ps = None

                
    def error(self, label, itv): # label实际的价格值
        # itv = self.predict()
        if itv[0] <= label <= itv[1]:
            return 0.0
        else:
            return 1.0


    def predict(self):
        obs_pred = self.base.predict()
        mu, sig = np.squeeze(obs_pred['mu']), np.sqrt(np.squeeze(obs_pred['cov']))
        interval = [mu - sig, mu + sig]
        return interval

    
    def init_or_update(self, label):
        if label is None:
            return

        if not self.initialized:
            self.base.init_state(label)
            self.initialized = True
        else:
            # predict
            self.ps = self.predict() 

            # check error before update
            err = self.error(label=label, itv=self.ps)
            self.n_err += err
            self.n_obs += 1
            # print(f'MVP: error = {self.n_err}, n = {self.n_obs}')           
            
            print(f'[OneSigma] size = {self.ps[1] - self.ps[0]:.4f}, '
                  f'interval = [{self.ps[0]:.4f}, {self.ps[1]:.4f}], obs = {label:.4f}, '
                  f'error_cur = {err}, '
                  f'error = {self.n_err / self.n_obs:.4f}'
            )

            # update the base model
            self.base_out = self.base.update(label, update_max=False)
            self.label = label

            
    def summary(self):
        return {
            'obs': self.label,
            'ps': self.ps,
            'n_err': self.n_err,
            'n_obs': self.n_obs,
        }
