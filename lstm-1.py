# coding=utf-8
# https://qiita.com/kenta1984/items/c2f3b2609071717dcf71 を参考
# https://qiita.com/tomov3/items/039d4271ed30490edf7b 交差検証法
# ave,0.9678487830987523,0.96625,0.9661256343917134,0.9662499999999999

import csv
import gc
from telnetlib import SE
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import neural_network
from sklearn.model_selection import cross_val_score
from sklearn import svm, datasets, model_selection
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.optimizers import adam_v2


jp = ["dai","inu","huto","take","hi","otto","ten","man","hu","bun","ki","hon","ta","husu","jou","tuki","yuu","aru","en","tan","me","niti","you","dou","syou","kai","gu","miru","hada","mei"]
csv_path = "./適合・再現・F値.csv"
x = ["set1","set2","set3","set4"]

BATCHSIZE = 128
EPOCHS = 100
UNITNUM = 256
LABELNUM = 30
LR = 0.0001

def plot_history(history, dir):
    plt.figure()
    # 精度の履歴をプロット
    plt.plot(history.history['accuracy'], "o-", label="accuracy")
    plt.plot(history.history['val_accuracy'], "o-", label="val_accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig('weights/day'+ str(dir) + '/lstm-1-acc.png')
    # plt.show()
    plt.figure()
    # 損失の履歴をプロット
    plt.plot(history.history['loss'], "o-", label="loss", )
    plt.plot(history.history['val_loss'], "o-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('weights/day'+ str(dir) + '/lstm-1-loss.png')
    # plt.show()

# ScoreCalculationの結果をCSVファイルに出力
def OutputScoreCalculation(pre_score, re_score, f1_sco, csv_path):
    """Parameters
    pre_score : 適合率
    re_score : 再現率
    f1_sco : F値
    """
    # my_makedir(csv_path.rsplit(‘/’, 1)[0])
    with open(csv_path, mode='w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Data", "Precision", "Recall", "F-measure"])
        #i = 0
        for l,p,r,f in zip(x, pre_score, re_score, f1_sco):
        #print(i)
            writer.writerow([l, p, r, f])
        #i += 1


day1_data = np.loadtxt('./day1.csv', delimiter=',', dtype=float)
day2_data = np.loadtxt('./day2.csv', delimiter=',', dtype=float)
day3_data = np.loadtxt('./day3.csv', delimiter=',', dtype=float)
day4_data = np.loadtxt('./day4.csv', delimiter=',', dtype=float)

set1_train = np.concatenate([day3_data,day4_data])
set2_train = np.concatenate([day1_data,day4_data])
set3_train = np.concatenate([day1_data,day2_data])
set4_train = np.concatenate([day2_data,day3_data])

traindata = []
testdata = []
valdata = []

traindata.append(set1_train)
traindata.append(set2_train)
traindata.append(set3_train)
traindata.append(set4_train)

testdata.append(day1_data)
testdata.append(day2_data)
testdata.append(day3_data)
testdata.append(day4_data)

valdata.append(day2_data)
valdata.append(day3_data)
valdata.append(day4_data)
valdata.append(day1_data)


x_train = []
x_test = []
x_val = []
y_train = []
y_test = []
y_val = []

y_test_backup = []



for i in range(4):
    x_train.append(traindata[i][:, 1:])
    x_test.append(testdata[i][:, 1:])
    x_val.append(valdata[i][:, 1:])
    y_train.append(traindata[i][:, 0:1])
    y_test.append(testdata[i][:, 0:1])
    y_val.append(valdata[i][:, 0:1])

    y_test_backup.append(y_test[i])

    x_train[i] = np.reshape(x_train[i], [x_train[i].shape[0], 3, -1])
    x_test[i] = np.reshape(x_test[i], [x_test[i].shape[0], 3, -1])
    x_val[i] = np.reshape(x_val[i], [x_val[i].shape[0], 3, -1])

    y_train[i] = np.reshape(np.array(np_utils.to_categorical(y_train[i])), [y_train[i].shape[0], LABELNUM])
    y_test[i] = np.reshape(np.array(np_utils.to_categorical(y_test[i])), [y_test[i].shape[0], LABELNUM])
    y_val[i] = np.reshape(np.array(np_utils.to_categorical(y_val[i])), [y_val[i].shape[0], LABELNUM])



data_len = 365

for i in range(4):
    model = Sequential()
    model.add(LSTM(UNITNUM, input_shape=(3, data_len), return_sequences=True))
    # model.add(LSTM(UNITNUM, input_shape=(3, data_len), return_sequences=True))
    model.add(LSTM(int(UNITNUM / 2), input_shape=(3, data_len), return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(UNITNUM, input_shape=(3, data_len), return_sequences=True))
    # model.add(LSTM(int(UNITNUM / 4), input_shape=(3, data_len), return_sequences=True))
    # model.add(LSTM(int(UNITNUM / 8), input_shape=(3, data_len), return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(int(UNITNUM / 16), input_shape=(3, data_len), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(LABELNUM, activation='softmax'))

    modelCheckpoint = ModelCheckpoint(
        filepath='./weights/' + 'day' + str(i+1) + '/best_model.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        period=1)

    model.compile(
        # loss='mean_squared_error', 
        loss='categorical_crossentropy', 
        # optimizer=adam_v2.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=LR/EPOCHS), 
        optimizer='adam', 
        metrics=['accuracy'])
        
    # 層構成をコンソール出力
    model.summary()

    # 層構成を画像ファイルで保存
    plot_model(model, to_file="lstm-1.png", show_shapes=True)


    history = model.fit(x_train[i], y_train[i], 
                        batch_size=BATCHSIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(x_val[i], y_val[i]),
                        callbacks = [modelCheckpoint])

    plot_history(history, i + 1)

    # メモリ解放
    keras.backend.clear_session()
    del model, history
    gc.collect()