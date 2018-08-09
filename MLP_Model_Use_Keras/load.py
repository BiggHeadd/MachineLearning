from keras.models import load_model
import pickle
import numpy as np

def load_test():
    with open("data/smp_test_x_bow.pkl", 'rb')as f:
        test_x = pickle.load(f)
    with open("data/smp_test_y.pkl", 'rb')as f:
        test_y = pickle.load(f)
    return test_x, test_y

def get_label(predict_array):
    predict_ = []
    for predict in predict_array:
        index = 0
        max = 0
        max_index = 0
        for each in predict:
            if each > max:
                max = each
                max_index = index
            index += 1
        predict_.append(max_index)
    return predict_

def get_score(predict, array):
    point = 0
    for p, a in zip(predict, array):
        if p == a:
            point+=1
    return point / len(predict)

if __name__ == "__main__":
    model = load_model("saved_model/6614/-epoch-86-val_acc_0.7017.hdf5")
    print(model.summary())

    x_train = np.load("data/0train_x_fold.npy", 'r')

    x, label = load_test()
    x = np.array(x)
    print(type(x))
    print(type(x_train))
    result = model.predict(x, batch_size=176, verbose=0)
    predict_ = get_label(result)
    print(type(result))
    print(result)
    result += result
    print('----------------------------')
    print(result)
    print('----------------------------')
    print(get_score(predict_,label))