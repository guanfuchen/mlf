# 逻辑回归

## 逻辑回归问题

soft binary classification:
$$f(x)=P(+1|x) \in [0,1]$$

理想的（noiseless）数据
$$(x_1, y_1^{\prime}=0.9 =P(+1|x_1))$$
$$(x_2, y_2^{\prime}=0.2 =P(+1|x_2))$$
$$(x_N, y_N^{\prime}=0.6 =P(+1|x_N))$$

实际的（noise）数据
$$(x_1, y_1^{\prime}=1 =P(y|x_1))$$
$$(x_2, y_2^{\prime}=0 =P(y|x_2))$$
$$(x_N, y_N^{\prime}=0 =P(y|x_N))$$

逻辑斯蒂假设H
$$s=\sum_{i=0}^{d}{w_i x_i}$$
通过使用逻辑斯蒂函数$\theta(s)$转换score为估计概率

逻辑斯蒂假设：$h(x)=\theta(w^T x)$

$$\theta(S)=\frac{1}{1+e^{-s}}$$

特殊的几个逻辑斯蒂值：$\theta(-\infty)=0$、$\theta(0)=0.5$、$\theta(\infty)=1$

将逻辑斯蒂函数带入得到：
$$h(x)=\frac{1}{1+exp(-w^T x)}$$

## 逻辑回归误差

在线性模型中误差函数有使用0/1误差（PLA感知器模型）、均方误差（线性回归），这里逻辑回归模型误差函数使用

目标函数$f(x)=P(+1|x)$

由于$h(-x)=1-h(x)$，那么对于样本$y$取值为+1和-1：
$$likelihood(logistic h) \propto \prod_{n=1}^{N}{h(y_n x_n)}$$

cross entropy误差（交叉熵误差）
$$\max_w \ likelihood(w) \propto \prod_{n=1}^{N}{\theta(y_n w^Tx_n)}$$

$$\max_w \ likelihood(w) \propto \ln{\prod_{n=1}^{N}{\theta(y_n w^Tx_n)}}$$

$$\max_w \ likelihood(w) \propto \sum_{n=1}^{N}\ln{\theta(y_n w^Tx_n)}$$

$$\min_w \frac{1}{N} \sum_{n=1}^{N}-\ln{\theta(y_n w^Tx_n)}$$

$$\min_w \frac{1}{N} \sum_{n=1}^{N}\ln{\theta(1+\exp(-y_n w^Tx_n))}$$

## 逻辑回归误差的梯度

$$\min_w E_in(w)=\frac{1}{N}\sum_{n=1}^{N}{\ln(1+\exp(-y_n w^T x_n))}$$
