import pandas as pd
import numpy as np
#tyouryuu\sabunn\sabunn1.py
# CSVファイルを読み込む

for i in range(1,13):
    read_file=f'tyouryuu\get_uv_250m_csv\get_uv_{i:02}.csv'
    df = pd.read_csv(read_file)
    
    df['u_sabunn_m']=(df['u']*36).round(6)
    df['v_sabunn_m']=(df['v']*36).round(6)

    df['delta_lat']=(df['u_sabunn_m']/6371).round(6)
    df['delta_lng']=(df['v_sabunn_m']/(6371*np.cos(df['delta_lat']))).round(6)


    ##他の変数桁数を丸める
    df['u']=df['u'].round(6)
    df['v']=df['v'].round(6)
    df['uv']=df['uv'].round(3)
    


    save_file=f'tyouryuu/sabunn/2016{i:02}_sabunn.csv'

        
    # numpy配列に変換して保存
    df.to_csv(save_file)
