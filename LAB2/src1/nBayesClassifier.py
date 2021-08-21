import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''
    def fit(self,traindata,trainlabel,featuretype):
        for i in np.unique(trainlabel):  # 对所有的可能分类结果循环
            self.Pc[i] = (trainlabel[np.where(trainlabel == i)].shape[0] + 1)/(trainlabel.shape[0] + np.unique(trainlabel).shape[0])
            tmp = traindata[np.where(trainlabel == i),:][0]  # 取出数据集中对应分类的所有组
            for j in range(traindata.shape[1]):  # 对所有的属性循环
                if featuretype[j] == 0:  # 属性为离散的情况
                    for k in np.unique(traindata[:,j]):  # 对离散属性的每一种取值可能循环
                        self.Pxc[(i, j, k)] = (tmp[np.where(tmp[:,0] == k)].shape[0] + 1)/(tmp.shape[0] + np.unique(traindata[:,0]).shape[0])
                else:  # 属性为连续的情况
                    self.Pxc[(i, j)] = (np.average(tmp.T[j]), np.sqrt(np.var(tmp.T[j])))


    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        prediction = []
        test_num = features.shape[0]
        for k in range(test_num):
            probabilities = []  # 存放概率的集合
            for c in range(1, 4):  # 对所有分类可能循环
                temp = math.log(self.Pc[c])  # 利用temp的值累乘计算特定分类的相对概率值
                for i in range(features.shape[1]):
                    if featuretype[i] == 0:  # 属性为离散的情况
                        temp += math.log(self.Pxc[(c, i, int(features[k][i]))])
                    else:  # 属性为连续的情况
                        (mean, sigma) = self.Pxc[(c, i)]  # 取出高斯分布的两个参数
                        temp += math.log(math.exp(-((features[k][i] - mean) ** 2 / 2) / (sigma ** 2)) /(((2 * math.pi) ** 0.5) * sigma))
                probabilities.append(temp)
            prediction.append(np.argmax(probabilities) + 1)  # 求出最大概率对应的分类
        return np.array(prediction).reshape(test_num, 1)


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()