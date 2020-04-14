import csv
import numpy as np
import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

data = pd.read_csv('./train.csv', encoding='big5')  # 'big5'编码：繁体中文
# print(data)

# 截取训练所需要的数据
data = data.iloc[:, 3:]
# 将所有值为'NR'的数据修改为0
data[data == 'NR'] = 0
data = data.to_numpy()
# print(raw_data)

month_data = {}
# 按月份生成数据
for month in range(12):
    # 每个小时有18种空气成分，每个月共20*24=480个小时
    temp_data = np.empty([18, 480])
    for day in range(20):
        # 向temp_data中加入第day天的数据，每天有24个小时即24列，因此列的范围为24*day~24(day+1)；
        # 加入的数据为第20*month+day这一天的数据，由于每天有18中空气成分，因此行的范围为18*(20*month+day)~18*(20*month+day+1)
        temp_data[:, 24 * day: 24 * (day + 1)] = data[18 * (20 * month + day):18 * (20 * month + day + 1)]
    month_data[month] = temp_data
# print(month_data[0])

# x为input，共12*471个data，每个data是一个18*9的feature vector
x = np.empty([12 * 471, 18 * 9])
# y为output，共12*471个data，每个data是一个一维的vector
y = np.empty([12 * 471, 1])

# 取出input和output
for month in range(12):
    # 每个月共471个data，依次遍历
    for i in range(471):
        # 将每个data的前9列数据reshape为一个18*9的行向量并储存到x里
        x[471 * month + i, :] = month_data[month][:, i:i + 9].reshape([1, -1])
        # 将每个data的第9行第10列的数据（即第10天的PM2.5值）储存到y里
        y[471 * month + i] = month_data[month][9, i + 9]
# print(x)
# print(y)

# 特征缩放（标准化）
# # 方式一（按列标准化）：
# x_mean = np.mean(x, axis=0)  # 按列求均值
# x_std = np.std(x, axis=0)  # 按列求标准差
# for i in range(len(x)):
#     for j in range(len(x[i])):
#         if x_std[j] != 0:
#             x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]
# # print(x)
# 方式二（按属性标准化）：
x_mean = np.zeros(18)
x_std = np.zeros(18)
for i in range(18):
    x_mean[i] = np.mean(x[:, 9 * i:9 * (i + 1)])  # 每次求一个属性所有值的均值
    x_std[i] = np.std(x[:, 9 * i:9 * (i + 1)])  # 每次求一个属性所有值的标准差
for i in range(18):
    if x_std[i] != 0:
        x[:, 9 * i:9 * (i + 1)] = (x[:, 9 * i:9 * (i + 1)] - x_mean[i]) / x_std[i]
# print(x)

# 整合模型数据
# 由于存在偏置项b，dimension需要在18*9的基础上再加上一维，初始化每一维对应的weight为0
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
# 在原先x内的每个样本上加上一个偏置项bias=1
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1)
# print(x)

# 设置模型参数
# 学习率
lr = 0.1
# 迭代次数
iter_time = 1000000
# adagrad表达式为：w'=w-lr*gd/sqrt(Σgd^2)，为了避免分母为0，在分母上加上一个很小的epsilon(ε)
eps = 0.00000000001
adagrad = np.zeros([dim, 1])

# 训练回归模型：y=w_1*x+w_2*x+...+b
for i in range(iter_time):
    # 损失函数（均方误差）：L = Σ(y_预测-y_真实)^2/m/2
    loss = np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12 / 2

    if i % 100 == 0:
        print("iter: {0}, loss: {1}".format(i, loss))

    grad = np.dot(x.transpose(), np.dot(x, w) - y) / 471 / 12
    adagrad += np.power(grad, 2)
    # 使用adagrad更新w
    w = w - lr * grad / np.sqrt(adagrad + eps)

# 保存模型参数
np.save('weight.npy', w)

# 载入测试数据并处理
test_data = pd.read_csv('./test.csv', header=None, encoding='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
# print(test_data)

test_x = np.empty([240, 18 * 9])
for i in range(240):
    test_x[i, :] = test_data[18 * i:18 * (i + 1), :].reshape([1, -1])
# for i in range(len(test_x)):
#     for j in range(len(test_x[i])):
#         if x_std[j] != 0:
#             test_x[i][j] = (test_x[i][j] - x_mean[j]) / x_std[j]
for i in range(18):
    if x_std[i] != 0:
        test_x[:, 9 * i:9 * (i + 1)] = (test_x[:, 9 * i:9 * (i + 1)] - x_mean[i]) / x_std[i]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1)
# print(test_x)

# 载入训练好的模型参数
w = np.load('./weight.npy')
# 预测测试集数据
pred = np.dot(test_x, w)
# print(pred)

# 将预测数据保存至csv文件中
with open('submission.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'value'])
    for i in range(240):
        row = ['id_' + str(i), pred[i][0]]
        writer.writerow(row)
