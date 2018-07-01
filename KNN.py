from __future__ import print_function


import numpy as np
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier

# 最大词数
maxlen = 20
# 每个词映射的维度
len_wv = 50


def Cross_validation(index, x_train, y_train, x_dev, y_dev, hidden_layer_size, \
                     max_iter, alpha, solver):
    print("model running...")
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=max_iter, alpha=alpha, solver=solver,
                          )
    model.fit(x_train, y_train)
    accuracy = model.score(x_dev, y_dev)
    return accuracy


if __name__ == '__main__':
    output = open("MLP.csv", "a")
    output.write("acc,index,hidden_layer_size,,max_iter,alpha,solver\n")
    hidden_layer_sizes = [[100, 100], [100, 150], [100, 200], [150, 100], [200, 100],
                          [150, 150], [200, 150], [200, 200]]
    max_iters = [300]
    solvers = ["adam"]
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
    t = 0
    for solver in solvers:
        for hidden_layer_size in hidden_layer_sizes:
            for max_iter in max_iters:
                for alpha in alphas:
                    t = t + 1
                    for i in range(0, 5):
                        index = str(t) + '.' + str(i)
                        print("running"+index+'....')
                        x_train = np.load("SMP2017/smp2017_5fold/" + str(i) +
                                          "_train_test_x_train_kf_smp_bow.npy")
                        x_dev = np.load("SMP2017/smp2017_5fold/" + str(i) +
                                          "_train_test_x_test_kf_smp_bow.npy")
                        # x_test = np.load("H:/research/data/SMP2017/smp2017_5fold/" + str(i) +
                        #                  "develop_x_test_smp_bow.npy")
                        # y_test = np.load("H:/research/data/SMP2017/smp2017_5fold/" + str(i) +
                        #                   "develop_y_test_smp_bow.npy")
                        y_train = np.load("SMP2017/smp2017_5fold/" + str(i) +
                                          "_train_test_y_train_kf_smp_bow.npy")
                        y_dev = np.load("SMP2017/smp2017_5fold/" + str(i) +
                                          "_train_test_y_test_kf_smp_bow.npy")


                        # y_dev = np_utils.to_categorical(y_dev, 31)  # 必须使用固定格式表示标签
                        # y_train = np_utils.to_categorical(y_train, 31)  # 必须使用固定格式表示标签 一共 31分类

                        acc = Cross_validation(index, x_train, y_train, x_dev, y_dev, hidden_layer_size, \
                                               max_iter, alpha, solver)
                        output.write( str(acc) +',' + str(index) + ',' + str(hidden_layer_size) + ',' + str(max_iter) + ',' + str(alpha)
                                     + ',' + str(solver) + "\n")
                        print('Test accuracy:', acc)
