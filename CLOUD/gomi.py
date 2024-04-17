from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM

# import keras
# from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as dates
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model


# from  scr_py.weather import  scr_weather

import sys
import os

# 現在のファイルのあるディレクトリの親パスをsys.pathに追加
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from scr_py import weather





# # 現在の日付を取得し、指定のフォーマットに変換
# specified_date = datetime.now().strftime("%Y/%#m/%#d")  # Windows以外では"%Y/%#m/%#d"を"%Y/%-m/%-d"に変更する

# 'predict_data.csv'を読み込む
pred_df = pd.read_csv('predict_csv/predict_data.csv')

# 今日の日付から7日間の日付でdatetime列を更新

# `kousuiryou_weather`列が存在する場合、そのデータで`kousuiryou`列を更新
pred_df['hour'] = pred_df['hour'].str.split(':').str[0]
# print(pred_df['hour_only'])
# 不要な列を削除
# 日付のみを抽出し、新しい列に格納

# # print(merged_df)
pred_df.to_csv('predict_csv/predict_data.csv', index=False)




# 強制的な日付指定
#specified_date = '2024/4/2'

# specified_date = datetime.now().strftime("%Y/%#m/%#d")
# predict_temperature(model,specified_date,weather_df)  # 例として指定した日の予測を実行

