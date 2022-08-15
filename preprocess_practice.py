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
length_max = -1
saityou = ""

# 全データから最長データ長のものを見つけてくる
files = glob("./minmax_linear-interpolation/data/all/*.csv")
for file_name in files:
    print(file_name)
    df = pd.read_csv(file_name)  # file_nameはファイルパスを表す
    if length_max < len(df):
        length_max = len(df)  # 最長のデータ長を取得
        saityou = file_name
length_max = length_max + 1

# 生データ
files = glob("./minmax_linear-interpolation/data/all/*.csv")

for file_name in files:
    newFileName = file_name.split("/")[-1]
    print(newFileName)

    # 生データ
    csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
    # リスト形式
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    # 開始点
    header = next(f)

    # rowListに生データの情報を追加していく
    rowList = []
    rowList.append(header)
    for row in f:
        rowList.append(row)

    # 各要素の抽出
    before_data_x = [x[0] for x in rowList]
    before_data_y = [x[1] for x in rowList]
    before_data_z = [x[2] for x in rowList]

    # ①minmax ###################################################
    after_data_x_minmax = preprocessing.minmax_scale(before_data_x)
    after_data_y_minmax = preprocessing.minmax_scale(before_data_y)
    after_data_z_minmax = preprocessing.minmax_scale(before_data_z)

    newRow = []
    newRow.append(after_data_x_minmax)
    newRow.append(after_data_y_minmax)
    newRow.append(after_data_z_minmax)

    # ディレクトリが存在しなければ新規作成
    if not (os.path.isdir("./minmax_linear-interpolation/data/all/minmax/day1")):
        os.mkdir("./minmax_linear-interpolation/data/all/minmax/day1")

    # 書き込み
    nf = open("./minmax_linear-interpolation/data/all/minmax/day1/minmax_" + newFileName,'w')  # r...ファイルを読み込み専用で開く/w...ファイルに書き込みをするために新規作成する。既存なら中身を消去して更新/a...書き込みをするために開く
    dataWriter = csv.writer(nf)
    dataWriter.writerows(newRow)
    #####################################################

    # ②線形補間 ####################################################
    before_data_x = after_data_x_minmax
    before_data_y = after_data_y_minmax
    before_data_z = after_data_z_minmax

    t = np.linspace(0, len(before_data_x), len(before_data_x))  # 補間前データの等比数列 = x座標
    tt = np.linspace(0, len(before_data_x), length_max)  # 補間後データの等比数列 = x座標

    f1 = interpolate.interp1d(t, before_data_x)  # 線形補間
    after_data_x = f1(tt)  # 補間後データのリスト

    f1 = interpolate.interp1d(t, before_data_y)  # 線形補間
    after_data_y = f1(tt)  # 補間後データのリスト

    f1 = interpolate.interp1d(t, before_data_z)  # 線形補間
    after_data_z = f1(tt)  # 補間後データのリスト

    newRow = []
    newRow.append(after_data_x)
    newRow.append(after_data_y)
    newRow.append(after_data_z)

    # ディレクトリがなければ作成
    if not (os.path.isdir("./minmax_linear-interpolation/data/all/linear_interpolation/day1")):
        os.mkdir("./data/all/linear_interpolation/day1")

    # 書き込み
    nf = open("./minmax_linear-interpolation/data/all/linear_interpolation/day1/linear_interpolation_" + newFileName, 'w')  # r...ファイルを読み込み専用で開く/w...ファイルに書き込みをするために新規作成する。既存なら中身を消去して更新/a...書き込みをするために開く
    dataWriter = csv.writer(nf)
    dataWriter.writerows(newRow)
    #####################################################

    nf.close()
print("最長は" + saityou + "です")
print("===== DONE =====")