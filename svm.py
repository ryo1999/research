# coding=utf-8
# https://qiita.com/kenta1984/items/c2f3b2609071717dcf71 を参考
# https://qiita.com/tomov3/items/039d4271ed30490edf7b 交差検証法

import csv
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

jp = ["dai","inu","huto","take","hi","otto","ten","man","hu","bun","ki","hon","ta","husu","jou","tuki","yuu","aru","en","tan","me","niti","you","dou","syou","kai","gu","miru","hada","mei"]
csv_path = "./適合・再現・F値.csv"
x = ["set1","set2","set3","set4"]

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

traindata = []
testdata = []

#data_len = 416

traindata.append(np.loadtxt('mix'+ '/mix1.csv', delimiter=',', dtype=float))
traindata.append(np.loadtxt('mix'+ '/mix2.csv', delimiter=',', dtype=float))
traindata.append(np.loadtxt('mix'+ '/mix3.csv', delimiter=',', dtype=float))
traindata.append(np.loadtxt('mix'+ '/mix4.csv', delimiter=',', dtype=float))

testdata.append(np.loadtxt('mix'+ '/mix1_test.csv', delimiter=',', dtype=float))
testdata.append(np.loadtxt('mix'+ '/mix2_test.csv', delimiter=',', dtype=float))
testdata.append(np.loadtxt('mix'+ '/mix3_test.csv', delimiter=',', dtype=float))
testdata.append(np.loadtxt('mix'+ '/mix4_test.csv', delimiter=',', dtype=float))

x_train = []
x_test = []
y_train = []
y_test = []

best_parameters = {'C': 0, 'gamma': 0}

for i in range(4):
    x_train.append(preprocessing.minmax_scale(traindata[i][:, 1:]))
    x_test.append(preprocessing.minmax_scale(testdata[i][:, 1:]))
    y_train.append(traindata[i][:, 0:1])
    y_test.append(testdata[i][:, 0:1])

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

best_score = 0

pre_ave = []
rec_ave = []
f_ave = []
count = 0
moji = 0

# 手動グリッドサーチを行って、最適なパラメータを探索
for gamma in [0.001,0.01,0,1,1]:
    for C in [0.1,1,10,100]:
        print("C: ", C)
        print("gamma: ", gamma)
        scores = []
        ave = 0
        for i in range(len(x_train)):
            svm = SVC(gamma=gamma, C=C)
            # svm.fit(x_train[i], y_train[i])
            svm.fit(x_train[i], np.ravel(y_train[i],order='C'))

            predicted_label = svm.predict(x_test[i])
            print("predicted_label: ", predicted_label)
            print("0")
            #print("y_test[i]: ", y_test[i])
            y_test_corrected = []
            predicted_label_corrected = []
            for j in y_test[i]:
                y_test_corrected.append(jp[int(j[0])])
            for j in predicted_label:
                predicted_label_corrected.append(jp[int(j)])
            for q in range(len(y_test_corrected)):
                if y_test_corrected[q] != predicted_label_corrected[q]:
                    print(predicted_label_corrected[q])
                count += 1
                if count%40 == 0:
                    moji += 1
                    print(moji)
                    if moji == 29:
                        moji = 0
                        break
            print("次の日！")
            # print("y_test_corrected: ", y_test_corrected)
            # print("predicted_labrel_corrected: ", predicted_label_corrected)
            # print("y_test_corrected length", len(y_test_corrected))
            # print("predicted_label length", len(predicted_label))
            # print("type predicted_label[0]", type(predicted_label_corrected[0]))
            # print("type y_test_corrected[0]", type(y_test_corrected[0]))
            pre_score = precision_score(y_test_corrected, predicted_label_corrected, average="macro", labels=jp)
            rec_score = recall_score(y_test_corrected, predicted_label_corrected, average="macro", labels=jp)
            # f_score = f1_score(y_test_corrected, predicted_label_corrected, average="macro", labels=jp)
            f_score = 2 * (pre_score*rec_score) / (pre_score+rec_score)
            pre_ave.append(pre_score)
            rec_ave.append(rec_score)
            f_ave.append(f_score)

            OutputScoreCalculation(pre_ave, rec_ave, f_ave, csv_path)

            scores.append(svm.score(x_test[i], y_test[i]))
            ave = sum(scores) / len(scores)
        if ave > best_score:
            best_score = ave
            best_parameters = {'C': C, 'gamma': gamma}
            print("-----------BEST------------")
            print("C : " + str(C))
            print("gamma : " + str(gamma))
            print("scores: ", scores)
            print("score_ave : " + str(ave))
            print("---------------------------")
            print("precision_average=" , sum(pre_ave)/len(pre_ave))
            print("recall_average=" , sum(rec_ave)/len(rec_ave))
            print("f_average=" , sum(f_ave)/len(f_ave))
            # print("precision=" , pre_ave)
            # print("recall=" , rec_ave)
            # print("f=" , f_ave)
            # print(scores)
