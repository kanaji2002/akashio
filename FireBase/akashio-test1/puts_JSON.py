import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import schedule
import time

# Firebaseの認証情報を使用して初期化
cred = credentials.Certificate('FireBase/akashio-test1/akshio-test1-firebase-adminsdk-kdl6f-980dde07b2.json')
firebase_admin.initialize_app(cred)

# Firestoreデータベースのインスタンスを取得
db = firestore.client()

def add_data_to_firestore():
    doc_ref = db.collection('your_collection').document()
    doc_ref.set({
        'data': 3
    })

add_data_to_firestore()
def send_data_to_firestore():
    # CSVファイルを読み込む
    df = pd.read_csv('tyouryuu/sabunn/201601_sabunn.csv')
    
    # DataFrameの各行をJSON形式に変換してFirestoreに送信
    for index, row in df.iterrows():
        data = row.to_dict()
        # ここでドキュメントIDを生成していますが、必要に応じてカスタマイズしてください
        doc_id = f'document_{index}'
        db.collection('your_collection').document(doc_id).set(data)

# send_data_to_firestore()

# # スケジューラーにジョブを追加
# schedule.every().hour.do(send_data_to_firestore)

# # 無限ループでスケジューラーを実行
# while True:
#     schedule.run_pending()
#     time.sleep(1)
