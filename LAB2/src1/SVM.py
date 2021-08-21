import numpy as np
import cvxopt #用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

# cvxopt.solvers.options['show_progress'] = False

#根据指定类别main_class生成1/-1标签
def svm_label(labels,main_class):
    new_label=[]
    for i in range(len(labels)):
        if labels[i]==main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)

# 实现线性回归
class SupportVectorMachine:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,kernel,C,Epsilon):
        self.kernel=kernel
        self.C = C
        self.Epsilon=Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''
    def KERNEL(self, x1, x2, kernel='Gauss', d=2, sigma=1):
        #d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1,x2)
        elif kernel == 'Poly':
            K = np.dot(x1,x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''
    def fit(self,train_data,train_label,test_data):
        # 构造 b
        b = cvxopt.matrix([0.])
        # 构造 q
        q = cvxopt.matrix(-1 * np.ones((train_data.shape[0], 1)))
        # 构造 h
        temp = np.ones((train_data.shape[0], 1))
        h = cvxopt.matrix(np.r_[self.C * temp, 0 * temp])
        # 构造 A
        A = cvxopt.matrix(train_label.reshape(1, -1).astype(np.double))
        # 构造 P
        P = np.zeros((train_data.shape[0], train_data.shape[0]))
        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[0]):
                P[i][j] = train_label[i] * train_label[j] * self.KERNEL(train_data[i], train_data[j], self.kernel)
        P = cvxopt.matrix(P)
        # 构造 G
        temp = np.eye(train_data.shape[0])
        G = cvxopt.matrix(np.r_[temp, -1 * temp])

        # 计算线性最优化问题
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x'])

        # 计算超平面和预测结果
        SV = set(np.where(alpha > self.Epsilon)[0])
        b_star = np.mean([train_label[i] - sum(
            [alpha[j] * train_label[j] * self.KERNEL(train_data[i], train_data[j], self.kernel) for j in SV]) for i in
                          SV])
        prediction = []
        for j in range(test_data.shape[0]):
            y_star = b_star + sum(
                [alpha[i] * train_label[i] * self.KERNEL(train_data[i], test_data[j], self.kernel) for i in SV])
            prediction.append(y_star)
        return np.array(prediction).reshape(-1, 1)


def main():
    # 加载训练集和测试集
    Train_data,Train_label,Test_data,Test_label=load_and_process_data()
    Train_label=[label[0] for label in Train_label]
    Test_label=[label[0] for label in Test_label]
    train_data=np.array(Train_data)
    test_data=np.array(Test_data)
    test_label=np.array(Test_label).reshape(-1,1)
    #类别个数
    num_class=len(set(Train_label))


    #kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    #C为软间隔参数；
    #Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    kernel='Gauss'
    C = 1
    Epsilon=10e-5
    #生成SVM分类器
    SVM=SupportVectorMachine(kernel,C,Epsilon)

    predictions = []
    #one-vs-all方法训练num_class个二分类器
    for k in range(1,num_class+1):
        #将第k类样本label置为1，其余类别置为-1
        train_label=svm_label(Train_label,k)
        # 训练模型，并得到测试集上的预测结果
        prediction=SVM.fit(train_data,train_label,test_data)
        predictions.append(prediction)
    predictions=np.array(predictions)
    #one-vs-all, 最终分类结果选择最大score对应的类别
    pred=np.argmax(predictions,axis=0)+1

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
