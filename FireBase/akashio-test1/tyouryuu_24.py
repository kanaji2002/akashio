from datetime import datetime, timedelta
import pandas as pd
#FireBase\akashio-test1\tyouryuu_24.py
##tyouryuu_24.pyは，現在の時刻から24時間以内のdelta_latとdelta_lngを返す関数
##JSONにする．
## 最終的には，deltaの値を求めてFireBaseに定期的に送る関数をセットにして，all_to_csv.pyから一括実行

def tyouryuu_24():
    # CSVファイルを読み込む
    df = pd.read_csv("FireBase/akashio-test1/one_year_sabunn.csv")
    df=df.drop(columns=['u', 'v', 'uv','u_sabunn_m','v_sabunn_m'])
    
    # 日付列をdatetime型に変換
    df['date'] = pd.to_datetime(df['date'])

    # 現在の日付と時刻を取得
    current_datetime = datetime.now()

    # 現在の日時から24時間分のデータを選択
    future_datetime = current_datetime + timedelta(hours=24)
    mask = (df['date'] >= current_datetime) & (df['date'] <= future_datetime)
    df24 = df.loc[mask]
    df24['date']=df24['date'].dt.strftime('%H:%M')
    
    
    
    print(df24)
    # 新しいDataFrameを作成しCSVに保存
    # df24.to_csv('FireBase/akashio-test1/next_24_hours.csv', index=False)
    df24.to_json('FireBase/akashio-test1/next_24_hours.json',orient='records' ,indent=4,index=False)

    
    


    return df24

tyouryuu_24()
