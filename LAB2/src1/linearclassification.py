from process_data import load_and_process_data
from evaluation import get_macro_F1,get_micro_F1,get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:

    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,lr=0.000005,Lambda= 0.001,epochs = 1000):
        self.lr=lr
        self.Lambda=Lambda
        self.epochs =epochs

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''
    def fit(self,train_features,train_labels):
        # 采用梯度下降算法， x 是数据集扩展来的 X 矩阵， w 是扩展后的被迭代的矩阵
        x = np.c_[np.ones(train_features.shape[0]),train_features]
        w = np.zeros(train_features.shape[1] + 1)
        # 开始执行 epochs 次迭代
        for t in range(self.epochs):
            for row in range(train_features.shape[0]):
                grad = 2 * np.dot(x[row].T, (np.dot(x[row], w.T) - train_labels.T[0][row]).T).T + 2 * self.Lambda * w.T  # 计算梯度
                w = w - self.lr * grad  # 根据梯度下降
        self.w = w

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''
    def predict(self,test_features):
        test_num = test_features.shape[0]  # 测试数据的数目
        X = np.c_[np.ones(test_features.shape[0]), test_features]  # X 作为测试数据
        prediction = []  # 预测结果
        for temp in X:
            y_predict = np.dot(temp, self.w)
            # 将各个数据分到最近的一个类
            if y_predict < 1.5:
                prediction.append(1)
            elif y_predict < 2.5:
                prediction.append(2)
            else:
                prediction.append(3)
        prediction = np.array(prediction).reshape(test_num, 1)  # 封装结果
        return prediction


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    lR=LinearClassification()
    lR.fit(train_data,train_label) # 训练模型
    pred=lR.predict(test_data) # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
