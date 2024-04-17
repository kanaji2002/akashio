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


#from  scr_py.weather import  scr_weather

import sys
import os

# 現在のファイルのあるディレクトリの親パスをsys.pathに追加
def kousuiryou():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from scr_py import weather



    weather_df=weather.scr_weather()

    weather_df=weather_df[['datetime','kousuiryou']]


    # # 現在の日付を取得し、指定のフォーマットに変換
    # specified_date = datetime.now().strftime("%Y/%#m/%#d")  # Windows以外では"%Y/%#m/%#d"を"%Y/%-m/%-d"に変更する

    # 'predict_data.csv'を読み込む
    pred_df = pd.read_csv('predict_csv/predict_data.csv')

    # 今日の日付から7日間の日付でdatetime列を更新
    for i in range(7):
        # 現在の日付にi日を加算
        new_date = (datetime.now() + timedelta(days=i)).strftime("%Y/%#m/%#d")  # Windows以外ではフォーマットを調整
        # 'datetime'列を新しい日付で更新
        weather_df.at[i, 'datetime'] = new_date

    # 'datetime'を基にして`predict_data_df`と`weather_df`をマージ
    # `weather_df`のデータで`predict_data_df`のデータを更新
    merged_df = pd.merge(pred_df, weather_df, on='datetime', how='left', suffixes=('', '_weather'))

    # `kousuiryou_weather`列が存在する場合、そのデータで`kousuiryou`列を更新
    merged_df['kousuiryou'] = merged_df['kousuiryou_weather'].fillna(merged_df['kousuiryou'])

    # 不要な列を削除
    merged_df.drop(columns=['kousuiryou_weather'], inplace=True)





    # print(merged_df)
    merged_df.to_csv('predict_csv\predict_data.csv', index=False)
    print("kousuiryou yosoku OK!")

# 引数なしで実行
kousuiryou()

# 強制的な日付指定
#specified_date = '2024/4/2'

# specified_date = datetime.now().strftime("%Y/%#m/%#d")
# predict_temperature(model,specified_date,weather_df)  # 例として指定した日の予測を実行

