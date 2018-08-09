# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:53:19 2018

@author: 佛山
"""


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
import random
from MLP_MODEL import MLPClassifier
import pickle as pk
import os
import load

hidden_layer_sizes = [[100, 100], [100, 150], [100, 200], [150, 100], [200, 100],
                          [150, 150], [200, 150], [200, 200]]
max_iters = [300]
alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
index = int(random.random()*10000)

def get_test():
    with open("data/smp_test_x_bow.pkl", 'rb')as f:
        test_x = pk.load(f)
    with open("data/smp_test_y.pkl",'rb')as f:
        test_y = pk.load(f)
    return test_x, test_y

def find_best(index):
    files = [f for f in os.listdir("saved_model/"+str(index))]
    best = 0
    for f in files:
        if float(f[19:-5]) > best:
            best = float(f[19:-5])
    return best

if __name__ == "__main__":
    with open("joint_predict.csv", 'a', encoding='utf-8')as f:
        f.write("avg" + ',' + "index - 1" + ',' + "hidden_layer_size" + ',' + "max_iter" + ',' + "alpha" + ',' + "predict_score" + '\n')
    output = open("MLP.csv",'a')
    output.write("acc,index,hidden_layer_size,,max_iter,alpha\n")
    parameter_lstm_list = list()
    for hidden_layer_size in hidden_layer_sizes:
        for max_iter in max_iters:
            for alpha in alphas:
                    
                param = dict()
                param['hidden_layer_size'] = hidden_layer_size
                param['max_iter'] = max_iter
                param['alpha'] = alpha
                    
                parameter_lstm_list.append(param)
    test_x, test_y = get_test()

    for param in parameter_lstm_list:
        hidden_layer_size = param['hidden_layer_size']
        max_iter = param['max_iter']
        alpha = param['alpha']

        best_bag = []
        avg = 0
        # joint_test = np.array()
        for i in range(0, 5):
            filename_x_train = "data/" + str(i) + "train_x_fold.npy"
            filename_y_train = "data/" + str(i) + "train_y_fold.npy"
            filename_x_test = "data/" + str(i) + "val_x_fold.npy"
            filename_y_test = "data/" + str(i) + "val_y_fold.npy"
            
            x_train = np.load(filename_x_train)
            y_train = np.load(filename_y_train)
            x_test = np.load(filename_x_test)
            y_test = np.load(filename_y_test)
            
            model_now = MLPClassifier(x_train, y_train, x_test, y_test, index, hidden_layer_size, max_iter, alpha)
            test_x = np.array(test_x)
            result = model_now.predict(test_x)
            if i == 0:
                joint_test = result
            else:
                joint_test += result
            now_best = find_best(index)
            best_bag.append(now_best)
            index += 0.2


        predict_id = load.get_label(joint_test)
        predict_score = load.get_score(predict_id, test_y)
        with open("joint_predict.csv", 'a', encoding='utf-8')as f:
            f.write(str(avg) + ',' + str(index-1) + ',' + str(hidden_layer_size) + ',' + str(max_iter) + ',' + str(alpha)+','+str(predict_score)+'\n')

        total = 0
        for each_best in best_bag:
            total += each_best
        avg = total/5
        output.write(str(avg) +',' + str(index-1) + ',' + str(hidden_layer_size) + ',' + str(max_iter) + ',' + str(alpha)
                                     + "\n")
    output.close()
        
                    