#coding:utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


'''构造训练数据 h(x) = thata0 * x0 + thata1 * x'''
x = np.arange(0., 10., 0.2)
m = len(x)                          
x0 = np.full(m, 1.0)                 
train_data = np.vstack([x0, x]).T   # 通过矩阵变化得到测试集 [x0 x1]
y = 4 * x + 1 + np.random.randn(m)  # 通过随机增减一定值构造'标准'答案 h(x)=4*x+1

def Bgd(alpha, loops, epsilon):
    '''
    [批量梯度下降]
    alpha:步长, 
    loops:循环次数,
    epsilon:收敛精度
    '''
    count = 0 # loop的次数
    thata = np.random.randn(2) # 随机thata向量初始的值,也就是随机起点位置
    err = np.zeros(2)  # 上次thata的值，初始为0向量
    finish = 0 # 完成标志位

    while count < loops:
        count += 1
        # 所有训练数据的期望更新一次thata
        sum = np.zeros(2) # 初始化thata更新总和
        for i in xrange(m):
            cost = (np.dot(thata, train_data[i]) - y[i]) * train_data[i]
            sum += cost
        thata = thata - alpha * sum
        if np.linalg.norm(thata - err) < epsilon: # 判断是否收敛
            finish = 1
            break
        else:
            err = thata # 没有则将当前thata向量赋值给err,作为下次判断收敛的参数之一
    print u'Bgd结果:\tloop_counts: %d\tthata[%f, %f]' % (count, thata[0], thata[1])
    return thata

def Sgd(alpha, loops, epsilon):
    '''
    [增量梯度下降]
    alpha:步长, 
    loops:循环次数,
    epsilon:收敛精度
    '''
    count = 0 # loop的次数
    thata = np.random.randn(2) # 随机thata向量初始的值,也就是随机起点位置
    err = np.zeros(2)  # 上次thata的值，初始为0向量
    finish = 0 # 完成标志位

    while count < loops:
        count += 1
        # 每组训练数据都会更新thata
        for i in xrange(m):
            cost = (np.dot(thata, train_data[i]) - y[i]) * train_data[i]
            thata = thata - alpha * cost
        if np.linalg.norm(thata - err) < epsilon: # 判断是否收敛
            finish = 1
            break
        else:
            err = thata # 没有则将当前thata向量赋值给err,作为下次判断收敛的参数之一
    print u'Sgd结果:\tloop_counts: %d\tthata[%f, %f]' % (count, thata[0], thata[1])
    return thata

if __name__ == '__main__':
    # thata = Sgd(alpha=0.001, loops=10000, epsilon=1e-4)
    thata = Bgd(alpha=0.0005, loops=10000, epsilon=1e-4)

    # 将训练数据导入stats的线性回归算法，以作验证
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
    print u'Stats结果:\tintercept(截距):%s\tslope(斜率):%s' % (intercept, slope)
        
    plt.plot(x, y, 'k+')
    plt.plot(x, thata[1] * x + thata[0], 'r')
    plt.show()