import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MLP:
    def __init__(self, lr=0.05, epochs=200):
        self.LLayer1_omega = torch.normal(0, 0.1, (5, 4), requires_grad=True)  # 第一个线性层
        self.LLayer1_b = torch.normal(0, 0.1, (1, 4), requires_grad=True)
        self.LLayer2_omega = torch.normal(0, 0.1, (4, 4), requires_grad=True)  # 第二个线性层
        self.LLayer2_b = torch.normal(0, 0.1, (1, 4), requires_grad=True)
        self.LLayer3_omega = torch.normal(0, 0.1, (4, 3), requires_grad=True)  # 第三个线性层
        self.LLayer3_b = torch.normal(0, 0.1, (1, 3), requires_grad=True)
        self.Sigmoid1 = None  # 第一个激活函数
        self.Sigmoid2 = None  # 第二个激活函数
        self.lr = lr
        self.epochs = epochs

    def train(self, train_data, train_label, test_epoch=0, test_line=0):
        loss_avg = []
        for j in range(self.epochs):
            loss_sum = []
            for i in range(train_data.shape[0]):
                # 前向计算结果
                X = torch.tensor([train_data[i].tolist()], requires_grad=True)  # 源数据
                out1 = torch.matmul(X, self.LLayer1_omega) + self.LLayer1_b  # 经过第一层
                self.Sigmoid1 = 1 / (1 + torch.exp(-out1))  # 经过第一个激活函数
                out2 = torch.matmul(self.Sigmoid1, self.LLayer2_omega) + self.LLayer2_b  # 经过第二层
                self.Sigmoid2 = 1 / (1 + torch.exp(-out2))  # 经过第二个激活函数
                out3 = torch.matmul(self.Sigmoid2, self.LLayer3_omega) + self.LLayer3_b  # 经过第三层
                y = torch.exp(out3) / torch.sum(torch.exp(out3))
                # 反向计算梯度
                loss = -torch.log(y[0][train_label[i]-1])
                loss_sum.append(float(loss))
                loss.backward()
                grad = y
                grad[0][train_label[i]-1] -= 1
                temp3_grad_omega = torch.matmul(self.Sigmoid2.T, grad)  # 反向传播第三层
                temp3_grad_b = torch.matmul(grad.T, torch.ones((self.Sigmoid2.shape[0], 1), requires_grad=True)).T
                grad = torch.matmul(grad, self.LLayer3_omega.T)
                grad = grad * self.Sigmoid2 * (1 - self.Sigmoid2)  # 反向传播第二个激活函数
                temp2_grad_omega = torch.matmul(self.Sigmoid1.T, grad)  # 反向传播第二层
                temp2_grad_b = torch.matmul(grad.T, torch.ones((self.Sigmoid1.shape[0], 1), requires_grad=True)).T
                grad = torch.matmul(grad, self.LLayer2_omega.T)
                grad = grad * self.Sigmoid1 * (1 - self.Sigmoid1)  # 反向传播第一个激活函数
                temp1_grad_omega = torch.matmul(X.T, grad)
                temp1_grad_b = torch.matmul(grad.T, torch.ones((X.shape[0], 1), requires_grad=True)).T
                grad = torch.matmul(grad, self.LLayer1_omega.T)  # 反向传播第一层
                if i == test_line and j == test_epoch:  # 取出一个特例验证求导是否正确
                    print("第", (j+1), "次迭代 第", (i+1), "个数据：")
                    print("线性层3:")
                    print("我的 omega 梯度:\n", temp3_grad_omega)
                    print("自动算的 omega 梯度:\n", self.LLayer3_omega.grad)
                    print("我的 b 梯度:", temp3_grad_b)
                    print("自动算的 b 梯度:", self.LLayer3_b.grad)
                    print("线性层2:")
                    print("我的 omega 梯度:\n", temp2_grad_omega)
                    print("自动算的 omega 梯度:\n", self.LLayer2_omega.grad)
                    print("我的 b 梯度:", temp2_grad_b)
                    print("自动算的 b 梯度:", self.LLayer2_b.grad)
                    print("线性层1:")
                    print("我的 omega 梯度:\n", temp1_grad_omega)
                    print("自动算的 omega 梯度:\n", self.LLayer1_omega.grad)
                    print("我的 b 梯度:", temp1_grad_b)
                    print("自动算的 b 梯度:", self.LLayer1_b.grad)
                # 更新网络
                self.LLayer1_omega = torch.tensor((self.LLayer1_omega - self.lr * temp1_grad_omega).tolist(),
                                                  requires_grad=True)  # 更新第一层
                self.LLayer1_b = torch.tensor((self.LLayer1_b - self.lr * temp1_grad_b).tolist(), requires_grad=True)
                self.LLayer2_omega = torch.tensor((self.LLayer2_omega - self.lr * temp2_grad_omega).tolist(),
                                                  requires_grad=True)  # 更新第二层
                self.LLayer2_b = torch.tensor((self.LLayer2_b - self.lr * temp2_grad_b).tolist(), requires_grad=True)
                self.LLayer3_omega = torch.tensor((self.LLayer3_omega - self.lr * temp3_grad_omega).tolist(),
                                                  requires_grad=True)  # 更新第三层
                self.LLayer3_b = torch.tensor((self.LLayer3_b - self.lr * temp3_grad_b).tolist(), requires_grad=True)
            loss_avg.append(sum(loss_sum) / train_data.shape[0])
        plt.plot(loss_avg)
        plt.show()
        print('Ends...')


if __name__ == '__main__':
    lr = 0.05
    epochs = 1000
    test_epoch = 3
    test_line = 8
    train_data = torch.zeros(100, 5)
    train_label = torch.randint(1, 4, (100, 1))
    for i in range(train_data.shape[0]):
        train_data[i] = torch.normal(0, train_label[i][0], (1, 5))
    MLP = MLP(lr, epochs)
    MLP.train(train_data, train_label, test_epoch, test_line)

