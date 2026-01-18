# question

- `kf.py`
  - 这里的
  > diff = obs - obs_mu_pred # diff~(0, sqrt(xi)) 很理想化
 
  - 最后的 `suoerlevelset` 反接求区间还没细看
  - `update_score_max` 方法中归一化的分母为什么要乘以2
  - 为什么要取出先验估计(预测值)值而不是后面验估计(最优估计), 一个可能的解释是: 把当前观测信息掺进“预测”，产生信息泄露，区间/误差评估会被“作弊”
  - 里面的`score`方法没有用到呀, 很多余