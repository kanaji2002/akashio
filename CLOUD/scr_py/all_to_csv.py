import kousuiryou_to_csv 
import nissyaryou_to_csv
import suionn_to_csv
import time
# 定期実行を行う．
import schedule

import subprocess
# scr_py/all_to_csv.py
from datetime import datetime,timedelta
from datetime import datetime

def job():
    print("Job started at:", datetime.now())
    subprocess.run(["python", "predict_py/kousuiryou_yosoku.py"])
    subprocess.run(["python", "predict_py/nissyaryou_xday_yosoku.py"])
    subprocess.run(["python", "predict_py/suionn_yosoku.py"])

def akashio_random():
    print("one week akashio started at:", datetime.now())
    subprocess.run(["python", "akashio_random.py"])

time1=datetime.strptime("18:26","%H:%M")
time2=time1+timedelta(minutes=1)
time3=time2+timedelta(minutes=1)
time4=time3+timedelta(minutes=1)
time5=time4+timedelta(minutes=1)

# #毎日01:30に実行
# ## 変数のスクレイピング
# schedule.every().day.at(time1.strftime("%H:%M")).do(kousuiryou_to_csv.write_scr_kousuiryou)
# schedule.every().day.at(time2.strftime("%H:%M")).do(nissyaryou_to_csv.write_scr_nissyaryou)
# schedule.every().day.at(time3.strftime("%H:%M")).do(suionn_to_csv.write_scr_suionn)

## それぞれの変数の1週間後までを予測
schedule.every().day.at(time4.strftime("%H:%M")).do(job)

## 一週間以内の赤潮発生の予測
schedule.every().day.at(time5.strftime("%H:%M")).do(akashio_random)





while True:
    # スケジュールに登録されたタスクを実行
    schedule.run_pending()
    # 1分ごとに確認
    time.sleep(10)