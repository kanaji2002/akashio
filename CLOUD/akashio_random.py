from sklearn import  svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pytz

import time

## akashio_random.py
def yosoku2():
    # CSVファイルを読み込む
    df = pd.read_csv("HIU_data_all_data2.csv")

    # 空白をNaNに変換
    df.replace(' ', pd.NA, inplace=True)

    # NaNを含む行を削除
    df.dropna(inplace=True)

    # Chl.aを3つのクラスに分割
    df['label_class'] = pd.cut(df['Chl.a'], bins=[-np.inf, 0.6 ,1,2,6,np.inf,], labels=[0,1,2,3,4])

    # 説明変数として使用する列を選択
    selected_columns = ['hour','nissyaryou','kousuiryou','suionn', 'tyouryuu']
    


# 説明変数として使用する列を選択
#selected_columns = ['DO', 'nissyaryou','kousuiryou_1pre','Sal','hour','Tem']

    # 正解データの列を指定
    label_column = 'label_class'

    # データの前処理
    labels = df[label_column]
    data = df[selected_columns]

    # 訓練データとテストデータに分割
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.1, random_state=0)

    # ランダムフォレストのアルゴリズムを利用して学習
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(data_train, label_train)

    # テストデータで予測
    label_pred = clf.predict(data_test)

    # 混同マトリックスを取得
    conf_matrix = confusion_matrix(label_test, label_pred)

    # 分類精度を表示
    accuracy = accuracy_score(label_test, label_pred)
    
    
    
    data_li = []
    hour_li=[]
    nissyaryou_li=[]
    kousuiryou_li=[]
    suionn_li=[]
    tyouryuu_li=[]
    pred_df=pd.read_csv('predict_csv/predict_data.csv')
    from datetime import datetime
# 現在の日付と時刻を取得
    now = datetime.now()
    current_date = now.strftime('%Y/%#m/%#d')  # 'YYYY-MM-DD'形式
    current_hour = now.hour  # 時間（0-23）
    matched_rows = pred_df[(pred_df['datetime'] == current_date) & (pred_df['hour'] == current_hour)]
# 6日後の日付と23時に一致する行を検索
    end_date = (now + timedelta(days=6)).strftime('%Y/%#m/%#d')
    end_rows = pred_df[(pred_df['datetime'] == end_date) & (pred_df['hour'] == 23)]   
    if not matched_rows.empty and not end_rows.empty:
        start_index = matched_rows.index[0]  # 現在の行のインデックス
        end_index = end_rows.index[0]  # 6日後の23時の行のインデックス
        
    # 現在の行から6日後の23時の行まで繰り返し
        for i in range(start_index, end_index + 1):
            # print(i)
            data_li.append(pred_df.iloc[i]['datetime'])
            hour_li.append(pred_df.iloc[i]['hour'])
            nissyaryou_li.append((pred_df.iloc[i]['nissyaryou']).round(3))
            kousuiryou_li.append(pred_df.iloc[i]['kousuiryou'])
            suionn_li.append((pred_df.iloc[i]['suionn']).round(3))
            tyouryuu_li.append(pred_df.iloc[i]['tyouryuu'])
    else:
        print("No matching rows found.")
        
    week_df = pd.DataFrame({
    'datetime': data_li,
    'hour':hour_li,
    'nissyaryou':nissyaryou_li,
    'kousuiryou':kousuiryou_li,
    'suionn':suionn_li,
    'tyouryuu':tyouryuu_li
    })


    # # 混同マトリックスを可視化
    # labels = df['label_class'].cat.categories

# 予測を行う
    
    string0="かなり安全"
    string1="安全"
    string2="注意が必要"
    string3="危険"
    string4="かなり危険"
    
    pred_df['answer']=None
    
    rows, cols = week_df.shape
    for i in range(rows):
        X_test = week_df.iloc[[i]][['hour', 'nissyaryou', 'kousuiryou', 'suionn', 'tyouryuu']]

    # 予測を実行
        predicted_label = clf.predict(X_test)   
        
        if predicted_label==0:
            week_df.loc[i,'answer']=string0
        elif predicted_label==1:
            week_df.loc[i,'answer']=string1
        elif predicted_label==2:
            week_df.loc[i,'answer']=string2
        elif predicted_label==3:
            week_df.loc[i,'answer']=string3
        elif predicted_label==4:
            week_df.loc[i,'answer']=string4
            
    # week_df['datetime']=datetime.strftime(week_df['datetime'],)        
    # week_df['datetime']=week_df['datetime']+week_df['hour']
    week_df['datetime'] = pd.to_datetime(week_df['datetime'])
    # hour列を文字列型に変換
    week_df['hour'] = week_df['hour'].astype(str)

    # datetime列とhour列を結合して新しい列を作成
    week_df['datetime_with_hour'] = week_df['datetime'].dt.strftime('%Y-%m-%d') + ' ' + week_df['hour']

    
    week_df.to_csv('predict_csv/week_df.csv',index=False)
    
        
        

        # 日本時間のタイムゾーンを取得

        

    return week_df
        # print(f"Accuracy: {accuracy:.3f}")
def add_data_to_firestore_one_week(df):
    # Firebaseの認証情報を使用して初期化
    # cred = credentials.Certificate('akshio-test1-firebase-adminsdk-kdl6f-8e7fe843c3.json')
    cred = credentials.Certificate('mic-kagawa-dx-murakamilab-firebase-adminsdk-5vxj6-9f29dca6a9.json')

    firebase_admin.initialize_app(cred)

    # Firestoreデータベースのインスタンスを取得
    db = firestore.client()
    for index, row in df.iterrows():
        doc_ref = db.collection('one_week_predict').document()
        doc_ref.set(row.to_dict())
        
    return 0
        
        
week_df=yosoku2()
add_data_to_firestore_one_week(week_df)





# print(yosoku2())