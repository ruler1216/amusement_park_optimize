import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import json
import os
from scipy.stats import norm
from datetime import datetime, time
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple, Optional
# 自作モジュール
from config import Config
from park_info import ParkInfo

"""関数定義"""

##### 満足度を可視化する関数(検証用)
def plot_satisfaction(
    parkinfo: ParkInfo,
    satisfaction: List[float],
    path: str=f'{Config.MODE}/検証', 
    filename: str='満足度.png'
    ) -> None:
    """
    Args:
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        satisfaction (List[float]): 満足度の二次元配列
        path (str, optional): ディレクトリパス(デフォルト:f'{Config.MODE}/検証')
        filename (str, optional): ファイル名(デフォルト:'満足度.png')

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    
    """
    W = [parkinfo.predict_granulity * i for i in range(1, 101)]
    
    for i in range(1, len(satisfaction)+1):
        #plt.subplot(2, 5, i)
        plt.plot(W, satisfaction[i-1], label=f'S={i}')
    plt.xlim(0, 300)
    plt.xlabel('アトラクション待ち時間予測値(分)')
    plt.ylabel('予測待ち時間を考慮した満足度')
    plt.legend(title='満足度', loc='upper right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    """
    """
    待ち時間の変化も満足度に含める
    例)10:00に100分で予測, 10:30に60分で予測⇒10:20くらいに移動しておけば有利となるが、10:05に移動してもあまり効果はない
    よって、
    (調整項) = α*(W_t - W_(t+1) / Δt)
    を加える。W_t=現時点での待ち時間予測値, W_(t+1)=次の待ち時間予測値, Δt=次の予測に切り替わるまでの残り時間, α=勾配の強さ
    ⇒
    分子と分母のスケールが異なるため、任意の関数でまとめて評価をすることは困難である。
    そのため、
    (調整項) = α*(W_t - W_(t+1))
    とし、正規分布によるスケーリングを行う。
    ※分母ΔtもReLUのような関数でスケーリングを行うこともできるが、予測値の時間粒度は30分単位なので、無理に数値化して計算に組み込む必要はないと思った。
    """
    W = [parkinfo.predict_granulity * i for i in range(1, 101)]
    
    # 分子
    numerator = [i for i in range(-50, 51, 5)]
    # 分子スケーリング
    #y_numerator = [1.5*(norm.pdf(n, 0, 10) / norm.pdf(0, 0, 10)) if n >= 0 else -1.5*(norm.pdf(n, 0, 10) / norm.pdf(0, 0, 10)) for n in numerator]
    y_numerator = [(-1)*1.5*(norm.pdf(n, 0, 10) / norm.pdf(0, 0, 10))+1.5 if n >= 0 else 1.5*(norm.pdf(n, 0, 10) / norm.pdf(0, 0, 10))-1.5 for n in numerator]
    # 調整
    #y_nume = [(-1)*y+1.5 for y in y_numerator]
    
    plt.plot(numerator, y_numerator, label='分子')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # 分母Δtは正規化可能(30で割ればいい)
    # speed_list = [i for i in range(-50, 51, 10)]
    # alpha = 1.5
    # for i in range(10, len(satisfaction) + 1):
    #     base = satisfaction[i-1]
        
    #     for speed in speed_list:
    #         s_total = [b + alpha*np.tanh(speed) for b in base]
            
    #         plt.plot(W, s_total, label=f'S={i}, Δ={speed}')
    
    # plt.xlim(0, 300)
    # plt.xlabel('アトラクション待ち時間予測値(分)')
    # plt.ylabel('予測待ち時間＋次回変化を考慮した満足度')
    # plt.legend(title='満足度×改善度', loc='upper right', fontsize=8)
    # plt.grid(True)
    # plt.savefig(save_path, dpi=300)
    # plt.close()

##### 乗車記録とアトラクション番号を作成する関数
def create_ride_data(columns: List[str]) -> Dict[str, int]:
    """
    Args:
        columns (List[str]): 予測済みアトラクションカラム名

    Returns:
        Dict[str, int]: {予測済みアトラクションカラム名:0(乗車回数)}
    """
    # 乗車記録
    ride_count = {col:0 for col in columns if col != 'Date'}
    
    return ride_count

##### 乗車回数に応じて満足度を減少させる関数
def reflect_ride_data(satisfaction: float, num: int) -> float:
    """
    Args:
        satisfaction (float): 乗車回数反映前の満足度
        num (int): 乗車回数

    Returns:
        float: 乗車回数反映後の満足度
    """
    # 2^(n-1)で除算(n≧1)
    return satisfaction / 2**(num - 1) if num >= 1 else satisfaction

##### 開園から閉園までの時間と現時点で使用した時間を分単位で計算し、プログラム実行日時を取得する関数
def calc_time(parkinfo: ParkInfo) -> Tuple[int, int, datetime]:
    """
    Args:
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        Tuple[int, int, int, datetime]: (開園から閉園までの時間, 現時点で使用した時間, プログラム実行日時)
    """
    # 開園から閉園までの時間
    total_time = (parkinfo.time_window[1] - parkinfo.time_window[0])*60
    
    # 使用した時間の計算
    now_dt = datetime.combine(datetime.today(), datetime.now().time()) # 現在時刻(yyyymmdd HHMMSS)
    base_dt = datetime.combine(datetime.today(), time(int(parkinfo.time_window[0]), (parkinfo.time_window[0] - int(parkinfo.time_window[0]))*60, 0)) # 開園時間(yyyymmdd HHMMSS)
    use_time = int((now_dt - base_dt).total_seconds() // 60) # 所要時間(分単位)
    
    return total_time, use_time, datetime.now()

##### 予測結果の前処理を行う関数
def preprocess_predict_df(predict_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Args:
        predict_df (pd.DataFrame): 前処理前の予測結果データフレーム

    Returns:
        Tuple[pd.DataFrame, int]: (前処理後の予測結果データフレーム, 予測間隔)
    """
    # データ型変更
    predict_df['Date'] = pd.to_datetime(predict_df['Date'])
    
    # 予測間隔
    predict_interval = [(val.seconds // 60) for val in predict_df['Date'].diff().unique() if pd.notna(val)][0]
    
    # 初回予測の満足度を適切に評価するためのデータ追加
    insert_data = {col:0 for col in predict_df.columns}
    insert_data['Date'] = predict_df.loc[0, 'Date'] - relativedelta(minutes=predict_interval)
    
    predict_df = pd.concat([pd.DataFrame([insert_data]), predict_df])
    predict_df = predict_df.sort_values('Date').reset_index(drop=True)
    
    return predict_df, predict_interval

##### 乗車回数を更新する関数(フォーマット)
def ride_overwrite_format(ride_count: Dict[str, int], ride_num: Dict[str, int]) -> Dict[str, int]:
    """
    Args:
        ride_count (Dict[str, int]): 乗車記録
        ride_num (Dict[str, int]): アトラクション番号

    Returns:
        Dict[str, int]: 乗車記録
    """
    flg = True
    while True:
        if flg:
            print('本日乗車したアトラクションの数字を入力してください(なければ-1を入力してください):')
            flg = False
        else:
            print('ほかに乗車したアトラクションの数字を入力してください(なければ-1を入力してください:)')
        
        attraction_num = int(input())
        if attraction_num == -1:
            break
        elif attraction_num in ride_num.keys():
            print(f'現在の{ride_num[attraction_num]}の乗車回数:{ride_count[ride_num[attraction_num]]}回')
            value = int(input('新しい乗車回数を入力してください:'))
            ride_count[ride_num[attraction_num]] = value
        else:
            print('無効な値です。もう一度入力してください。')
    
    return ride_count

##### 満足度の公式
def _formula(
    satisfaction: float,
    parkinfo: ParkInfo,
    predict_time: int,
    next_predict: int=0,
    alpha: float=1.5
    ) -> float:
    """
    Args:
        satisfaction (float): 満足度
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        predict_time (int): 予測待ち時間
        next_predict (int, optional): 次の予測待ち時間(デフォルト:0)
        alpha (float, optional): 調整項の勾配(デフォルト:1.5)

    Returns:
        float: 予測待ち時間及び調整項を考慮した満足度
    """
    # 正規分布用分散(満足度ごとに裾を広げるように定式化)
    v = 5 + 10*satisfaction
    # 予測待ち時間を考慮した満足度
    y_base = satisfaction * (norm.pdf(predict_time, parkinfo.predict_granulity, v) / norm.pdf(parkinfo.predict_granulity, parkinfo.predict_granulity, v))
    # 予測差分
    delta_predict = predict_time - next_predict
    # 調整項
    y_adjust = np.where(
        delta_predict > 0,
        (-1)*alpha*(norm.pdf(delta_predict, 0, 10) / norm.pdf(0, 0, 10)) + alpha,
        alpha*(norm.pdf(delta_predict, 0, 10) / norm.pdf(0, 0, 10)) - alpha
    )
    
    return (y_base + y_adjust) if next_predict != 0 else y_base

##### 予測待ち時間を考慮した実際の満足度の算出
def calc_satisfaction(
    satisfaction: float,
    predict_df: pd.DataFrame,
    column: str,
    results: Dict[str, float],
    parkinfo: ParkInfo,
    ) -> Dict[str, float]:
    """
    Args:
        satisfaction (float): 予測待ち時間未考慮の満足度
        predict_df (pd.DataFrame): 予測結果データフレーム
        column (str): 満足度算出対象のカラム
        results (Dict[str, float]): 結果を格納する辞書
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        Dict[str, float]: {満足度算出対象のカラム:予測待ち時間及び調整項を考慮した満足度}
    """
    if len(predict_df) == 1:
        results[column] = _formula(satisfaction, parkinfo, int(predict_df.iloc[0]))
    else:
        results[column] = _formula(satisfaction, parkinfo, int(predict_df.iloc[0]), int(predict_df.iloc[1]))
    
    return results

##### 満足度ランキングを表示する関数
def satisfaction_ranking(
    satisfaction_dict: Dict[str, float],
    ) -> pd.DataFrame:
    """
    Args:
        satisfaction_dict (Dict[str, float]): {'アトラクション名':満足度}

    Returns:
        pd.DataFrame: 満足度ランキング
    """
    # 満足度を降順ソートし、アトラクション名だけ抽出
    satisfaction_dict_sorted = sorted(satisfaction_dict.items(), key=lambda x:x[1], reverse=True)
    ranking = [satisfaction[0] for satisfaction in satisfaction_dict_sorted]
    
    # データフレーム化
    df_ranking = pd.DataFrame(ranking, columns=['アトラクション名'])
    
    # ランキングを明示的に追加
    df_ranking.index += 1
    df_ranking = df_ranking.reset_index(drop=False).rename(columns={'index':'満足度ランキング'})
    
    return df_ranking