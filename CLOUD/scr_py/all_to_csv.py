import kousuiryou_to_csv 
import nissyaryou_to_csv
import suionn_to_csv
import time
# 定期実行を行う．
import schedule

import subprocess
# scr_py/all_to_csv.py

#毎日01:30に実行
schedule.every().day.at("13:11").do(kousuiryou_to_csv.write_scr_kousuiryou)
schedule.every().day.at("13:19").do(nissyaryou_to_csv.write_scr_nissyaryou)
schedule.every().day.at("13:20").do(suionn_to_csv.write_scr_suionn)

from datetime import datetime

def job():
    print("Job started at:", datetime.now())
    subprocess.run(["python", "predict_py/kousuiryou_yosoku.py"])
    subprocess.run(["python", "predict_py/nissyaryou_xday_yosoku.py"])
    subprocess.run(["python", "predict_py/suionn_yosoku.py"])


# 毎日特定の時刻にjob関数を実行するスケジュールを設定
schedule.every().day.at("11:49").do(job)

while True:
    # スケジュールに登録されたタスクを実行
    schedule.run_pending()
    # 1分ごとに確認
    time.sleep(10)