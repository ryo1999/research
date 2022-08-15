# coding=utf-8
import numpy as np
from scipy import signal, interpolate
from sklearn import preprocessing
import csv
import pandas as pd
from glob import glob
from operator import mul
import os
import time

# 最大データ長を保持する変数
"""length_max = -1
# 前処理完了データから最長データ長のものを見つけてくる
files = glob("./data/O/*.csv")
for file_name in files:
    print(file_name)
    continue
    df = pd.read_csv(file_name)  # file_nameはファイルパスを表す
    if length_max < len(df):
        length_max = len(df)  # 最長のデータ長を取得
length_max = length_max + 1
"""
# データ
out_file = "allday.csv"
folder = "./minmax_linear-interpolation/data/all/linear_interpolation/day4"
files = glob(folder+"/*.csv")

jp = ["dai","inu","huto","take","hi","otto","ten","man","hu","bun","ki","hon","ta","husu","jou","tuki","yuu","aru","en","tan","me","niti","you","dou","syou","kai","gu","miru","hada","mei"]
file_list = dict()

for key in jp:
    file_list[key] = list()

for file_name in files:
    newFileName = file_name.split("/")[-1]
    for key in jp:
        if "_"+key+"_" in newFileName:
            file_list[key].append(newFileName)

for key in jp:
    file_list[key].sort()

with open("./mix/"+out_file, "a") as csv_all:
    writer = csv.writer(csv_all)
    for key in jp:
        for file in file_list[key]:
            # 生データ
            csv_file = open(folder+"/"+file, "r", encoding="ms932", errors="", newline="" )
            # リスト形式
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            header = next(f)
            pre_data = []
            pre_data.append(header)
            for row in f:
                pre_data.append(row)
            
            csv_file.close()
            data = [str(jp.index(key))] + pre_data[0] + pre_data[1] + pre_data[2]
            writer.writerow(data)

print("===== OWARI PEKO =====")