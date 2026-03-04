import os
import pandas as pd
import numpy as np
import re
import jpholiday
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Tuple, List, Dict, Optional, Any
# =====　自作モジュール　=====
from park_info import ParkInfo
from config import Config

"""関数定義"""

##### 学習期間に対応するファイルの抽出を行う関数※後続処理のため予測日の月データも包含する
def extract_train_datafile(
    dir_path: str, 
    train_period: Tuple[str, str]=Config.TRAIN_PRIOD,
    forecast_date: str=Config.FORECAST_DATE) -> List[str]:
    """
    Args:
        dir_path (str): ディレクトリパス
        train_period (Tuple[str, str]): (学習開始日, 学習終了日) (デフォルト:config.pyのTRAIN_PERIOD)
        forecast_date (str): 予測対象日 (デフォルト:config.pyのFORECAST_DATE)

    Returns:
        List[str]: 学習期間に対応するファイルリスト(ない場合は空のリスト)
    """
    files = os.listdir(dir_path)
    
    # ファイル比較を行うための正規表現パターン
    pattern = re.compile(r'(\d{4}).+(\d{2}).+')
    
    ##### 文字列から整数型に変換する関数
    def to_yyyymm(s: str) -> int:
        """
        Args:
            s (str): 年月情報が入った文字列

        Returns:
            int: yyyymm形式の整数値
        """
        m = pattern.search(s)
        
        return int(m.group(1) + m.group(2))
    
    start = to_yyyymm(train_period[0])
    end = to_yyyymm(forecast_date)
    
    result = []
    
    for f in files:
        value = to_yyyymm(f)
        
        if start <= value <= end:
            result.append(os.path.join(dir_path, f))
    
    return result

##### アトラクション正式名称でカラム名のリネームを行う関数
def dataframe_columns_rename(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): カラムのリネーム前のデータフレーム
        file_path (str): アトラクション正式名称のパス

    Returns:
        pd.DataFrame: カラムのリネーム後のデータフレームもしくはリネームの必要がないデータフレーム
    """
    file_open = open(file_path, 'r')
    columns = json.load(file_open)
    
    df = df.rename(columns=columns)
    
    use_cols = [col for col in columns.values() if col in df.columns]
    return df[['Date'] + use_cols]

##### 気象情報の前処理を行う関数
def preprocess_weather(file_path: str) -> pd.DataFrame:
    """
    Args:
        file_path (str): 気象情報のファイルパス

    Returns:
        pd.DataFrame: 前処理後のデータフレーム
    """
    df = pd.read_json(file_path)
    # 使用データ
    df_idxs = df.index
    df_cols = ['hourly_units', 'hourly']
    
    result = pd.DataFrame()
    for idx in df_idxs:
        result[f'{idx}({df.loc[idx, df_cols[0]]})'] = df.loc[idx, df_cols[1]]
    
    # ISO8601からDatetimeに変換
    result['time(iso8601)'] = result['time(iso8601)'].apply(lambda x:datetime.fromisoformat(x))
    # 気象コードをカテゴリカルに変換
    result['weather_code(wmo code)'] = result['weather_code(wmo code)'].astype('category')
    
    # 必要カラム
    columns = [
        'time(iso8601)', # 日付
        'temperature_2m(°C)', # 気温
        'relative_humidity_2m(%)', # 相対湿度
        'apparent_temperature(°C)', # 体感温度
        'precipitation(mm)', # 降水量(前1時間合計)※雨,にわか雨, 雪が含まれる
        'cloud_cover(%)', # 雲量
        'direct_radiation(W/m²)', # 直達日射量(前1時間平均)
        'wind_speed_10m(m/s)', # 風速
        'weather_code(wmo code)' # 気象コード
    ]
    
    return result[columns]

##### 前処理を行う関数
def _preprocess(
    df: pd.DataFrame,
    parkinfo: ParkInfo
    ) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Args:
        df (pd.DataFrame): 前処理前のデータフレーム
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        Tuple[pd.DataFrame, List[str], int]: (特徴量作成前の前処理済データフレーム, アトラクションカラムリスト, 遊園地の1日当たりのレコード数)
    """
    # アトラクション待ち時間が数値以外の場合は、欠損として扱いつつ、データ型を変更
    num_cols = [column for column in df.columns if column != 'Date']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    
    # Dateカラムを型変更しつつ、気象情報マージ用のカラムを作成
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['time(iso8601)'] = df['Date'].dt.floor('1h')
    
    # 後続処理に必要なカラム作成
    df['時分'] = df['Date'].dt.strftime('%H:%M')
    df['年月日'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    slots = len(parkinfo.times)
    
    return df, num_cols, slots

##### 過去データ統計値を作成する関数
def _create_stats_features(
    df: pd.DataFrame,
    data: Dict[str, Any],
    columns: List[str],
    target_col: str
    ) -> Tuple[Dict[str, Any], List[str]]:
    """
    Args:
        df (pd.DataFrame): データフレーム
        data (Dict[str, Any]): 過去データ統計値作成前データ
        columns (List[str]): 過去データ統計値追加前カラムリスト
        target_col (str): 処理対象カラム

    Returns:
        Tuple[Dict[str, Any], List[str]]: (過去データ統計値作成後データ, 過去データ統計値追加後カラムリスト)
    """
    # 過去データ平均値(同時間帯)
    data[f'{target_col}_加工_平均値'] = (
        df.groupby('時分')[target_col]
        .transform(lambda x: x.expanding().mean().shift())
    )
    columns.append(f'{target_col}_加工_平均値')
    
    # 過去データ中央値(同時間帯)
    data[f'{target_col}_加工_中央値'] = (
        df.groupby('時分')[target_col]
        .transform(lambda x: x.expanding().median().shift())
    )
    columns.append(f'{target_col}_加工_中央値')
    
    # 過去データ標準偏差(同時間帯)
    data[f'{target_col}_加工_標準偏差'] = (
        df.groupby('時分')[target_col]
        .transform(lambda x: x.expanding().std().shift())
    )
    columns.append(f'{target_col}_加工_標準偏差')
    
    return data, columns

##### 移動平均を作成する関数
def _create_rolling_features(
    df: pd.DataFrame,
    data: Dict[str, Any],
    columns: List[str],
    slots: int,
    target_col: str
    ) -> Tuple[Dict[str, Any], List[str]]:
    """
    Args:
        df (pd.DataFrame): データフレーム
        data (Dict[str, Any]): 移動平均作成前データ
        columns (List[str]): 移動平均追加前カラムリスト
        slots (int): 遊園地の1日当たりのレコード数
        target_col (str): 処理対象カラム

    Returns:
        Tuple[Dict[str, Any], List[str]]: (移動平均作成後データ, 移動平均追加後カラムリスト)
    """
    # 前々日の移動平均
    data[f'{target_col}_加工_前々日移動平均'] = (
        df[target_col]
        .shift(2*slots)
        .rolling(window=slots, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    columns.append(f'{target_col}_加工_前々日移動平均')
    
    # 1週間前の移動平均
    data[f'{target_col}_加工_1週間移動平均'] = (
        df[target_col]
        .shift(7*slots)
        .rolling(window=slots, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    columns.append(f'{target_col}_加工_1週間移動平均')
    
    return data, columns

##### ラグを作成する関数
def _create_lag_features(
    df: pd.DataFrame,
    data: Dict[str, Any],
    columns: List[str],
    slots: int,
    target_col: str
    ) -> Tuple[Dict[str, Any], List[str]]:
    """
    Args:
        df (pd.DataFrame): データフレーム
        data (Dict[str, Any]): ラグ作成前データ
        columns (List[str]): ラグ追加前カラムリスト
        slots (int): 遊園地の1日当たりのレコード数
        target_col (str): 処理対象カラム

    Returns:
        Tuple[Dict[str, Any], List[str]]: (ラグ作成後データ, ラグ追加後カラムリスト)
    """
    # 前々日の待ち時間(同時間帯)
    data[f'{target_col}_加工_ラグ前々日'] = df[target_col].shift(2*slots).reset_index(drop=True)
    columns.append(f'{target_col}_加工_ラグ前々日')
    
    # 一週間前の待ち時間(同時間帯)
    data[f'{target_col}_加工_ラグ1週間'] = df[target_col].shift(7*slots).reset_index(drop=True)
    columns.append(f'{target_col}_加工_ラグ1週間')
    
    return data, columns

##### 横持ちから縦持ちに変換する関数
def _long_to_pivot(
    df: pd.DataFrame,
    num_cols: List[str],
    feature_cols: List[str]
    ) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): 特徴量加工済み横持ちデータフレーム
        num_cols (List[str]): アトラクションカラムリスト
        feature_cols (List[str]): 特徴量カラムリスト

    Returns:
        pd.DataFrame: 特徴量加工済み縦持ちデータフレーム
    """
    # 縦持ちデータに変換
    df_normal = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=num_cols, var_name='アトラクション名', value_name='待ち時間')
    df_feature = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=feature_cols, var_name='feature', value_name='value')
    df_feature['アトラクション名'] = df_feature['feature'].str.split('_').str.get(0)
    df_feature['stats'] = df_feature['feature'].str.split('_').str.get(2)
    
    df_feature = df_feature.pivot_table(
        index=['Date', 'time(iso8601)', 'アトラクション名'],
        columns=['stats'],
        values='value'
    ).reset_index()
    
    df = df_normal.merge(
        df_feature,
        on=['Date', 'time(iso8601)', 'アトラクション名'],
        how='left'
    )
    
    return df

##### アトラクション全体の統計値を作成する関数
def _create_all_stats_features(
    df: pd.DataFrame,
    num_cols: List[str]
    ) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): 特徴量加工済み縦持ちデータフレーム
        num_cols (List[str]): アトラクションカラムリスト

    Returns:
        pd.DataFrame: アトラクション全体の統計値データフレーム
    """
    # アトラクションごとの合計
    attraction_total = (
        df.groupby('年月日')[num_cols]
        .sum()
    )
    attraction_cols = attraction_total.columns
    # 遊園地全体の合計
    daily_total = attraction_total.sum(axis=1)
    
    # 年月日をカラムに戻す
    attraction_total = attraction_total.reset_index()
    daily_total = daily_total.reset_index()
    
    # アトラクション合計
    attraction_total_pivot = attraction_total.melt(
        id_vars=['年月日'],
        value_vars=attraction_cols,
        var_name='アトラクション名',
        value_name='合計待ち時間'
        )
    
    # アトラクション合計/遊園地全体の合計
    df_all_stats = pd.merge(
        attraction_total_pivot,
        daily_total,
        how='left',
        on='年月日'
    )
    df_all_stats['相対割合'] = df_all_stats['合計待ち時間']/df_all_stats[0]
    
    df_all_stats = df_all_stats.assign(**{
        '前々日全体待ち時間':df_all_stats[0].shift(2).values,
        '1週間全体待ち時間':df_all_stats[0].shift(7).values,
        '前々日相対割合':df_all_stats['相対割合'].shift(2).values,
        '1週間相対割合':df_all_stats['相対割合'].shift(7).values
    })
    
    """
    ver2.0
    # 前々日の全体待ち時間と1週間前の全体待ち時間
    df_all_stats = pd.DataFrame({
        '年月日':daily_total.index,
        '前々日全体待ち時間':daily_total.shift(2).values,
        '1週間全体待ち時間':daily_total.shift(7).values,
    })
    """
    
    return df_all_stats

##### 後処理を行う関数
def _post_process(
    df: pd.DataFrame,
    parkinfo: ParkInfo
    ) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): 全特徴量加工済み縦持ちデータフレーム
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        pd.DataFrame: 後処理後の縦持ちデータフレーム
    """
    # 特徴量作成
    df = df.assign(
        月_sin=np.sin(2 * np.pi * df["Date"].dt.month / 12),
        月_cos=np.cos(2 * np.pi * df["Date"].dt.month / 12),
        日_sin=np.sin(2 * np.pi * df["Date"].dt.day / 30), # 処理効率を意識して30日固定
        日_cos=np.cos(2 * np.pi * df["Date"].dt.day / 30), # 処理効率を意識して30日固定
        時間_sin=np.sin(2 * np.pi * df["Date"].dt.hour / 24),
        時間_cos=np.cos(2 * np.pi * df["Date"].dt.hour / 24),
        分_sin=np.sin(2 * np.pi * df["Date"].dt.minute / 60),
        分_cos=np.cos(2 * np.pi * df["Date"].dt.minute / 60),
        曜日_sin=np.sin(2 * np.pi * df['Date'].dt.day_of_week / 7),
        曜日_cos=np.cos(2 * np.pi * df['Date'].dt.day_of_week / 7),
        祝日フラグ=df['Date'].dt.date.apply(lambda d: int(jpholiday.is_holiday(d))).astype('category'), # 0:祝日でない, 1:祝日
        運転停止フラグ=df['待ち時間'].isnull().astype('int').astype('category') # 0:通常運転, 1:運転停止
    )
    
    # 遊園地の営業時間でデータを絞り込み
    df['時間']=df['Date'].dt.hour
    df = df[(df['時間'] >= parkinfo.time_window[0]) & (df['時間'] < parkinfo.time_window[1])].reset_index(drop=True)
    
    df['アトラクション名'] = df['アトラクション名'].astype('category')
    
    # 予測時間粒度で四捨五入を行うカラム
    rounding_columns = [
        '平均値',
        '中央値',
        #'標準偏差',
        'ラグ前々日',
        '前々日移動平均',
        '前々日全体待ち時間',
        'ラグ1週間',
        '1週間移動平均',
        '1週間全体待ち時間'
    ]
    
    for col in rounding_columns:
        df[col] = round(df[col] / parkinfo.predict_granulity) * parkinfo.predict_granulity
        
    
    columns = [
        'Date',
        'time(iso8601)',
        '年月日', # 予測対象日フィルタリング用
        'アトラクション名',
        '待ち時間',
        '平均値',
        '中央値',
        '標準偏差',
        'ラグ前々日',
        '前々日移動平均',
        #'前々日相対割合',
        '前々日全体待ち時間',
        'ラグ1週間',
        '1週間移動平均',
        #'1週間相対割合',
        '1週間全体待ち時間',
        '月_sin',
        '月_cos',
        '日_sin',
        '日_cos',
        '時間_sin',
        '時間_cos',
        '分_sin',
        '分_cos',
        '曜日_sin',
        '曜日_cos',
        '祝日フラグ',
        '運転停止フラグ',
    ]
    
    return df[columns]

##### データの前処理や特徴量の作成を行う関数
def preprocess_data(
    df: pd.DataFrame,
    parkinfo: ParkInfo,
    train_period: Tuple[str, str]=Config.TRAIN_PRIOD,
    forecast_date: str=Config.FORECAST_DATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Args:
        df (pd.DataFrame): 前処理前の全(学習+予測)データフレーム
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        train_period (Tuple[str, str]): (学習開始日, 学習終了日) (デフォルト:config.pyのTRAIN_PERIOD)
        forecast_date (str): 予測対象日(デフォルト:config.pyのFORECAST_DATE)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: (学習用データフレーム, 予測用データフレーム, {予測できないカラム名:予測できないカラム名})※Dict型はJSON形式で保存するため
    """
    train_start_dt = datetime.strptime(train_period[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')
    train_end_dt = datetime.strptime(train_period[1] + ' ' + '23:59:00', '%Y-%m-%d %H:%M:%S')
    
    df, num_cols, slots = _preprocess(df, parkinfo)
    
    null_cols = {}
    # 学習データにおけるアトラクションが全欠損なら削除
    for col in df.columns[df.isnull().all()]:
        null_cols[col] = col
    df = df.drop(columns=null_cols.keys())
    
    num_cols = list(filter(lambda x: x not in null_cols, num_cols))
    
    # 全特徴量カラム
    feature_cols = []
    # 過去統計値の作成
    stats_features = {}
    # 移動平均の作成
    rolling_features = {}
    # ラグ特徴量の作成
    lag_features = {}
    for col in num_cols:
        stats_features, feature_cols = _create_stats_features(df, stats_features, feature_cols, col)
        rolling_features, feature_cols = _create_rolling_features(df, rolling_features, feature_cols, slots, col)
        lag_features, feature_cols = _create_lag_features(df, lag_features, feature_cols, slots, col)
        
    df = pd.concat([
        df,
        pd.DataFrame(stats_features),
        pd.DataFrame(rolling_features),
        pd.DataFrame(lag_features)
        ], axis=1)
    
    # 横持ちから縦持ちに変換
    results = _long_to_pivot(df, num_cols, feature_cols)
    results['年月日'] = results['Date'].dt.strftime('%Y-%m-%d')
    
    # 全アトラクションの統計値を作成
    df_all_stats = _create_all_stats_features(df, num_cols)
    
    results = pd.merge(
            results, 
            df_all_stats,
            how='left',
            on=['年月日', 'アトラクション名']
        )
    
    # 後処理
    results = _post_process(results, parkinfo)
    
    # 学習期間データの抽出
    df_train = results[(results['Date'] >= train_start_dt)&(results['Date'] <= train_end_dt)].drop(columns='年月日').reset_index(drop=True)
    # 予測期間データの抽出
    df_fore = results[(results['年月日'] == forecast_date)].drop(columns='年月日').reset_index(drop=True)
    df_fore['運転停止フラグ'] = 0
    df_fore['運転停止フラグ'] = df_fore['運転停止フラグ'].astype('category')
    
    return df_train, df_fore, null_cols

##### 待ち時間の前処理を行う関数
# def _preprocess_wait_time(
#     df: pd.DataFrame,
#     parkinfo: ParkInfo,
#     null_cols: Dict[str, str] = {},
#     df_past: Optional[pd.DataFrame]=None,
#     train_period: Tuple[str, str]=Config.TRAIN_PRIOD,
#     ) -> Tuple[pd.DataFrame, Dict[str, str]]:
#     """
#     Args:
#         df (pd.DataFrame): 前処理前のデータフレーム
#         parkinfo (ParkInfo): 遊園地情報のインスタンス
#         data (Optional[pd.DataFrame]): 予測に使用するデータを作るための過去待ち時間データフレーム(デフォルト:None)
#         train_period (Tuple[str, str]): (学習開始日, 学習終了日)

#     Returns:
#         Tuple[pd.DataFrame, List[str]]: (前処理後のデータフレーム, {予測できないカラム名:予測できないカラム名})※Dict型はJSON形式で保存するため
#     """ 
#     """
#     学習データと予測データで処理を分割する
#     ※学習データから先に作ると学習データが縦持ちになり予測データとの処理を共通化できなくなる。
#     各特徴量の作成ごとに関数を作るなどして保守性を向上させる
#     特徴量を作る際には日時により全体特徴量から抜き出せるようにする
#     """
#     # アトラクション待ち時間が数値以外の場合は、欠損として扱いつつ、データ型を変更
#     num_cols = [column for column in df.columns if column != 'Date']
#     df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    
#     # Dateカラムを型変更しつつ、気象情報マージ用のカラムを作成
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     df['time(iso8601)'] = df['Date'].dt.floor('1h')
    
#     if df_past is None:
#         # 学習データにおけるアトラクションが全欠損なら削除
#         # 学習期間データの抽出
#         train_start_dt = datetime.strptime(train_period[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')
#         train_end_dt = datetime.strptime(train_period[1] + ' ' + '23:59:00', '%Y-%m-%d %H:%M:%S')
        
#         df = df[(df['Date'] >= train_start_dt)&(df['Date'] <= train_end_dt)].reset_index(drop=True)
        
#         for col in df.columns[df.isnull().all()]:
#             null_cols[col] = col
#         df = df.drop(columns=null_cols.keys())
        
#         num_cols = list(filter(lambda x: x not in null_cols, num_cols))
        
#         # アトラクションに関する特徴量カラムの作成
#         feature_cols = []
#         features = {}
#         df['時分'] = df['Date'].dt.strftime('%H:%M')
#         df['年月日'] = df['Date'].dt.strftime('%Y-%m-%d')
#         #df = df.sort_values(['時分', 'Date']).reset_index(drop=True)
#         df = df.sort_values('Date').reset_index(drop=True)
#         for col in num_cols:
#             # 過去データ平均値(同時間帯)
#             features[f'{col}_加工_平均値'] = (
#                 df.groupby('時分')[col]
#                 .transform(lambda x: x.expanding().mean().shift())
#             )
#             feature_cols.append(f'{col}_加工_平均値')
            
#             # 過去データ中央値(同時間帯)
#             features[f'{col}_加工_中央値'] = (
#                 df.groupby('時分')[col]
#                 .transform(lambda x: x.expanding().median().shift())
#             )
#             feature_cols.append(f'{col}_加工_中央値')
            
#             # 過去データ標準偏差(同時間帯)
#             features[f'{col}_加工_標準偏差'] = (
#                 df.groupby('時分')[col]
#                 .transform(lambda x: x.expanding().std().shift())
#             )
#             feature_cols.append(f'{col}_加工_標準偏差')
        
#         df = pd.concat([df, pd.DataFrame(features)], axis=1)
        
#         features = {}
#         slots = len(parkinfo.times)
#         for col in num_cols:
#             # 前々日の移動平均
#             features[f'{col}_加工_前々日移動平均'] = (
#                 df[col]
#                 .shift(2*slots)
#                 .rolling(window=slots, min_periods=1)
#                 .mean()
#                 .reset_index(drop=True)
#             )
#             feature_cols.append(f'{col}_加工_前々日移動平均')
            
#             # 1週間前の移動平均
#             features[f'{col}_加工_1週間移動平均'] = (
#                 df[col]
#                 .shift(7*slots)
#                 .rolling(window=slots, min_periods=1)
#                 .mean()
#                 .reset_index(drop=True)
#             )
#             feature_cols.append(f'{col}_加工_1週間移動平均')
        
#         df = pd.concat([df, pd.DataFrame(features)], axis=1)
        
#         features = {}
#         #df = df.sort_values(['年月日', 'Date']).reset_index(drop=True)
#         for col in num_cols:
#             # 前々日の待ち時間(同時間帯)
#             features[f'{col}_加工_ラグ前々日'] = df[col].shift(2*slots).reset_index(drop=True)
#             feature_cols.append(f'{col}_加工_ラグ前々日')
            
#             # 一週間前の待ち時間(同時間帯)
#             features[f'{col}_加工_ラグ1週間'] = df[col].shift(7*slots).reset_index(drop=True)
#             feature_cols.append(f'{col}_加工_ラグ1週間')
        
#         df = pd.concat([df, pd.DataFrame(features)], axis=1)
        
#         # 前々日の全体待ち時間と1週間前の全体待ち時間をマージ
#         daily_total = (
#             df.groupby('年月日')[num_cols]
#             .sum()
#             .sum(axis=1)
#         )
#         features_total = pd.DataFrame({
#             '年月日':daily_total.index,
#             '前々日全体待ち時間':daily_total.shift(2).values,
#             '1週間全体待ち時間':daily_total.shift(7).values,
#         })
        
#         # 縦持ちデータに変換
#         df_normal = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=num_cols, var_name='アトラクション名', value_name='待ち時間')
#         df_feature = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=feature_cols, var_name='feature', value_name='value')
#         df_feature['アトラクション名'] = df_feature['feature'].str.split('_').str.get(0)
#         df_feature['stats'] = df_feature['feature'].str.split('_').str.get(2)
        
#         df_feature = df_feature.pivot_table(
#             index=['Date', 'time(iso8601)', 'アトラクション名'],
#             columns=['stats'],
#             values='value'
#         ).reset_index()
        
#         df = df_normal.merge(
#             df_feature,
#             on=['Date', 'time(iso8601)', 'アトラクション名'],
#             how='left'
#         )
        
#         df = pd.merge(
#             df, 
#             features_total,
#             how='left',
#             on='年月日'
#         )
#     else:
#         df = df.drop(columns=null_cols.keys())
#         num_cols = list(filter(lambda x: x not in null_cols, num_cols))
        
#         df_past['時分'] = df_past['Date'].dt.strftime('%H:%M')
#         df_past['年月日'] = df_past['Date'].dt.strftime('%Y-%m-%d')
        
#         df = df.sort_values('Date').reset_index(drop=True)
        
#         two_days_ago = (df['Date'].unique()[0] - relativedelta(days=2)).strftime('%Y-%m-%d')
#         one_week_ago = (df['Date'].unique()[0] - relativedelta(days=7)).strftime('%Y-%m-%d')
        
#         df_two_days_ago = df_past[df_past['年月日'] == two_days_ago]
#         df_one_week_ago = df_past[df_past['年月日'] == one_week_ago]
        
#         slots = len(parkinfo.times)
#         for col in num_cols:
#             # 前々日の移動平均
#             features[f'{col}_加工_前々日移動平均'] = (
#                 df_two_days_ago[col]
#                 .rolling(window=slots, min_periods=1)
#                 .mean()
#                 .reset_index(drop=True)
#             )
#             feature_cols.append(f'{col}_加工_前々日移動平均')
            
#             # 1週間前の移動平均
#             features[f'{col}_加工_1週間移動平均'] = (
#                 df_one_week_ago[col]
#                 .rolling(window=slots, min_periods=1)
#                 .mean()
#                 .reset_index(drop=True)
#             )
#             feature_cols.append(f'{col}_加工_1週間移動平均')
        
#         df = pd.concat([df, pd.DataFrame(features)], axis=1)
        
#         features = {}
#         #df = df.sort_values(['年月日', 'Date']).reset_index(drop=True)
#         for col in num_cols:
#             # 前々日の待ち時間(同時間帯)
#             features[f'{col}_加工_ラグ前々日'] = df_two_days_ago[col].reset_index(drop=True)
#             feature_cols.append(f'{col}_加工_ラグ前々日')
            
#             # 一週間前の待ち時間(同時間帯)
#             features[f'{col}_加工_ラグ1週間'] = df_one_week_ago[col].reset_index(drop=True)
#             feature_cols.append(f'{col}_加工_ラグ1週間')
        
#         df = pd.concat([df, pd.DataFrame(features)], axis=1)
        
#         # 過去データから統計値テーブルを作成
#         stats_table = (
#             df_past
#             .groupby(['時分', 'アトラクション名'], observed=True)['待ち時間']
#             .agg(['mean', 'median', 'std'])
#             .reset_index()
#             .rename(columns={'mean':'平均値', 'median':'中央値', 'std':'標準偏差'})
#         )
        
#         df = df.merge(
#             stats_table,
#             on=['時分', 'アトラクション名'],
#             how='left'
#         )
        
#         df['運転停止フラグ'] = 0
#         df['運転停止フラグ'] = df['運転停止フラグ'].astype('category')
    
#     # 特徴量作成
#     df = df.assign(
#         月_sin=np.sin(2 * np.pi * df["Date"].dt.month / 12),
#         月_cos=np.cos(2 * np.pi * df["Date"].dt.month / 12),
#         日_sin=np.sin(2 * np.pi * df["Date"].dt.day / 30), # 処理効率を意識して30日固定
#         日_cos=np.cos(2 * np.pi * df["Date"].dt.day / 30), # 処理効率を意識して30日固定
#         時間_sin=np.sin(2 * np.pi * df["Date"].dt.hour / 24),
#         時間_cos=np.cos(2 * np.pi * df["Date"].dt.hour / 24),
#         分_sin=np.sin(2 * np.pi * df["Date"].dt.minute / 60),
#         分_cos=np.cos(2 * np.pi * df["Date"].dt.minute / 60),
#         曜日_sin=np.sin(2 * np.pi * df['Date'].dt.day_of_week / 7),
#         曜日_cos=np.cos(2 * np.pi * df['Date'].dt.day_of_week / 7),
#         祝日フラグ=df['Date'].dt.date.apply(lambda d: int(jpholiday.is_holiday(d))).astype('category'), # 0:祝日でない, 1:祝日
#         運転停止フラグ=df['待ち時間'].isnull().astype('int').astype('category') # 0:通常運転, 1:運転停止
#     )
    
#     # 遊園地の営業時間でデータを絞り込み
#     df['時間']=df['Date'].dt.hour
#     df = df[(df['時間'] >= parkinfo.time_window[0]) & (df['時間'] < parkinfo.time_window[1])].reset_index(drop=True)
    
#     df['アトラクション名'] = df['アトラクション名'].astype('category')
    
#     columns = [
#         'Date',
#         'time(iso8601)',
#         'アトラクション名',
#         '待ち時間',
#         '平均値',
#         '中央値',
#         '標準偏差',
#         'ラグ前々日',
#         '前々日移動平均',
#         '前々日全体待ち時間',
#         'ラグ1週間',
#         '1週間移動平均',
#         '1週間全体待ち時間',
#         '月_sin',
#         '月_cos',
#         '日_sin',
#         '日_cos',
#         '時間_sin',
#         '時間_cos',
#         '分_sin',
#         '分_cos',
#         '曜日_sin',
#         '曜日_cos',
#         '祝日フラグ',
#         '運転停止フラグ',
#     ]
    
#     return df[columns], null_cols

# 特徴量ver1.0(検証用)
# def preprocess_wait_time(
#     df: pd.DataFrame,
#     parkinfo: ParkInfo,
#     null_cols: Dict[str, str] = {},
#     df_past: Optional[pd.DataFrame]=None,
#     train_period: Tuple[str, str]=Config.TRAIN_PRIOD
#     ) -> Tuple[pd.DataFrame, Dict[str, str]]:
#     """
#     Args:
#         df (pd.DataFrame): 前処理前のデータフレーム
#         parkinfo (ParkInfo): 遊園地情報のインスタンス
#         data (Optional[pd.DataFrame]): 予測に使用するデータを作るための過去待ち時間データフレーム(デフォルト:None)
#         train_period (Tuple[str, str]): (学習開始日, 学習終了日)

#     Returns:
#         Tuple[pd.DataFrame, List[str]]: (前処理後のデータフレーム, {予測できないカラム名:予測できないカラム名})※Dict型はJSON形式で保存するため
#     """    
#     # アトラクション待ち時間が数値以外の場合は、欠損として扱いつつ、データ型を変更
#     num_cols = [column for column in df.columns if column != 'Date']
#     df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

#     # Dateカラムを型変更しつつ、気象情報マージ用のカラムを作成
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     df['time(iso8601)'] = df['Date'].dt.floor('1h')

#     df = df.sort_values('Date')

#     # 学習データにおけるアトラクションが全欠損なら削除
#     if df_past is None:
#         # 学習期間データの抽出
#         train_start_dt = datetime.strptime(train_period[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')
#         train_end_dt = datetime.strptime(train_period[1] + ' ' + '23:59:00', '%Y-%m-%d %H:%M:%S')

#         df = df[(df['Date'] >= train_start_dt)&(df['Date'] <= train_end_dt)].reset_index(drop=True)

#         for col in df.columns[df.isnull().all()]:
#             null_cols[col] = col
#         df = df.drop(columns=null_cols.keys())
#     else:
#         df = df.drop(columns=null_cols.keys())

#     num_cols = list(filter(lambda x: x not in null_cols, num_cols))

#     # 各アトラクションの1つ前の時間帯における待ち時間を作成
#     lag_cols = []
#     for col in num_cols:
#         # データの最初が欠損になるが08:15の時間帯なので対応不要
#         df[f'{col}_加工'] = df[col].shift()
#         lag_cols.append(f'{col}_加工')

#     # 縦持ちデータに変換
#     df_normal = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=num_cols, var_name='アトラクション名', value_name='待ち時間')
#     df_lag = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=lag_cols, var_name='アトラクション名', value_name='待ち時間_加工')

#     df_lag['アトラクション名'] = df_lag['アトラクション名'].str.replace('_加工', '', regex=True)

#     df = df_normal.merge(
#         df_lag,
#         on=['Date', 'time(iso8601)', 'アトラクション名'],
#         how='left'
#     )

#     # 特徴量作成
#     df = df.assign(
#         年=df['Date'].dt.year,
#         月=df['Date'].dt.month,
#         日=df['Date'].dt.day,
#         時間=df['Date'].dt.hour,
#         分=df['Date'].dt.minute,
#         曜日=df['Date'].dt.day_of_week.astype('category'),
#         祝日フラグ=df['Date'].dt.date.apply(lambda d: int(jpholiday.is_holiday(d))).astype('category'), # 0:祝日でない, 1:祝日
#         運転停止フラグ=df['待ち時間'].isnull().astype('int').astype('category') # 0:通常運転, 1:運転停止
#     )
    
#     df = df.assign(
#         月_sin=np.sin(2 * np.pi * df["Date"].dt.month / 12),
#         月_cos=np.cos(2 * np.pi * df["Date"].dt.month / 12),
#         日_sin=np.sin(2 * np.pi * df["Date"].dt.day / 30), # 処理効率を意識して30日固定
#         日_cos=np.cos(2 * np.pi * df["Date"].dt.day / 30), # 処理効率を意識して30日固定
#         時間_sin=np.sin(2 * np.pi * df["Date"].dt.hour / 24),
#         時間_cos=np.cos(2 * np.pi * df["Date"].dt.hour / 24),
#         分_sin=np.sin(2 * np.pi * df["Date"].dt.minute / 60),
#         分_cos=np.cos(2 * np.pi * df["Date"].dt.minute / 60),
#         曜日_sin=np.sin(2 * np.pi * df['Date'].dt.day_of_week / 7),
#         曜日_cos=np.cos(2 * np.pi * df['Date'].dt.day_of_week / 7),
#         #祝日フラグ=df['Date'].dt.date.apply(lambda d: int(jpholiday.is_holiday(d))).astype('category'), # 0:祝日でない, 1:祝日
#         #運転停止フラグ=df['待ち時間'].isnull().astype('int').astype('category') # 0:通常運転, 1:運転停止
#     )
    
#     # 遊園地の営業時間でデータを絞り込み
#     df = df[(df['時間'] >= parkinfo.time_window[0]) & (df['時間'] < parkinfo.time_window[1])].reset_index(drop=True)

#     # 予測データの前処理の場合
#     if df_past is not None:
#         df_past = df_past.assign(
#             時間=df_past['Date'].dt.hour,
#             分=df_past['Date'].dt.minute,
#         )
#         # 過去データから統計値テーブルを作成
#         stats_table = (
#             df_past
#             .groupby(['時間', '分', 'アトラクション名'], observed=True)['待ち時間']
#             .mean()
#             .reset_index()
#             .rename(columns={'待ち時間':'待ち時間_加工'})
#         )
#         # 統計値を予測時間粒度で四捨五入
#         stats_table['待ち時間_加工'] = round(stats_table['待ち時間_加工'] / parkinfo.predict_granulity) * parkinfo.predict_granulity
#         df = df.drop(columns='待ち時間_加工').merge(
#             stats_table,
#             on=['時間', '分', 'アトラクション名'],
#             how='left'
#         )

#         df['運転停止フラグ'] = 0
#         df['運転停止フラグ'] = df['運転停止フラグ'].astype('category')

#         # カラム順序変更
#         df = df.reindex(columns=df_past.columns)

#     df['アトラクション名'] = df['アトラクション名'].astype('category')

#     columns = [
#         'Date',
#         'time(iso8601)',
#         #'年月日', # 予測対象日フィルタリング用
#         'アトラクション名',
#         '待ち時間',
#         #'待ち時間_加工',
#         #'平均値',
#         #'中央値',
#         #'標準偏差',
#         #'ラグ前々日',
#         #'前々日移動平均',
#         #'前々日全体待ち時間',
#         #'ラグ1週間',
#         #'1週間移動平均',
#         #'1週間全体待ち時間',
#         '月_sin',
#         '月_cos',
#         '日_sin',
#         '日_cos',
#         '時間_sin',
#         '時間_cos',
#         '分_sin',
#         '分_cos',
#         '曜日_sin',
#         '曜日_cos',
#         '祝日フラグ',
#         '運転停止フラグ',
#     ]
    
#     return df[columns], null_cols

# # 特徴量ver1.0
# def preprocess_wait_time(
#     df: pd.DataFrame,
#     parkinfo: ParkInfo,
#     null_cols: Dict[str, str] = {},
#     df_past: Optional[pd.DataFrame]=None,
#     train_period: Tuple[str, str]=Config.TRAIN_PRIOD
#     ) -> Tuple[pd.DataFrame, Dict[str, str]]:
#     """
#     Args:
#         df (pd.DataFrame): 前処理前のデータフレーム
#         parkinfo (ParkInfo): 遊園地情報のインスタンス
#         data (Optional[pd.DataFrame]): 予測に使用するデータを作るための過去待ち時間データフレーム(デフォルト:None)
#         train_period (Tuple[str, str]): (学習開始日, 学習終了日)

#     Returns:
#         Tuple[pd.DataFrame, List[str]]: (前処理後のデータフレーム, {予測できないカラム名:予測できないカラム名})※Dict型はJSON形式で保存するため
#     """    
#     # アトラクション待ち時間が数値以外の場合は、欠損として扱いつつ、データ型を変更
#     num_cols = [column for column in df.columns if column != 'Date']
#     df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

#     # Dateカラムを型変更しつつ、気象情報マージ用のカラムを作成
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     df['time(iso8601)'] = df['Date'].dt.floor('1h')

#     df = df.sort_values('Date')

#     # 学習データにおけるアトラクションが全欠損なら削除
#     if df_past is None:
#         # 学習期間データの抽出
#         train_start_dt = datetime.strptime(train_period[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')
#         train_end_dt = datetime.strptime(train_period[1] + ' ' + '23:59:00', '%Y-%m-%d %H:%M:%S')

#         df = df[(df['Date'] >= train_start_dt)&(df['Date'] <= train_end_dt)].reset_index(drop=True)

#         for col in df.columns[df.isnull().all()]:
#             null_cols[col] = col
#         df = df.drop(columns=null_cols.keys())
#     else:
#         df = df.drop(columns=null_cols.keys())

#     num_cols = list(filter(lambda x: x not in null_cols, num_cols))

#     # 各アトラクションの1つ前の時間帯における待ち時間を作成
#     lag_cols = []
#     for col in num_cols:
#         # データの最初が欠損になるが08:15の時間帯なので対応不要
#         df[f'{col}_加工'] = df[col].shift()
#         lag_cols.append(f'{col}_加工')

#     # 縦持ちデータに変換
#     df_normal = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=num_cols, var_name='アトラクション名', value_name='待ち時間')
#     df_lag = df.melt(id_vars=['Date', 'time(iso8601)'], value_vars=lag_cols, var_name='アトラクション名', value_name='待ち時間_加工')

#     df_lag['アトラクション名'] = df_lag['アトラクション名'].str.replace('_加工', '', regex=True)

#     df = df_normal.merge(
#         df_lag,
#         on=['Date', 'time(iso8601)', 'アトラクション名'],
#         how='left'
#     )

#     # 特徴量作成
#     df = df.assign(
#         年=df['Date'].dt.year,
#         月=df['Date'].dt.month,
#         日=df['Date'].dt.day,
#         時間=df['Date'].dt.hour,
#         分=df['Date'].dt.minute,
#         曜日=df['Date'].dt.day_of_week.astype('category'),
#         祝日フラグ=df['Date'].dt.date.apply(lambda d: int(jpholiday.is_holiday(d))).astype('category'), # 0:祝日でない, 1:祝日
#         運転停止フラグ=df['待ち時間'].isnull().astype('int').astype('category') # 0:通常運転, 1:運転停止
#     )
    
#     # 遊園地の営業時間でデータを絞り込み
#     df = df[(df['時間'] >= parkinfo.time_window[0]) & (df['時間'] < parkinfo.time_window[1])].reset_index(drop=True)

#     # 予測データの前処理の場合
#     if df_past is not None:
#         # 過去データから統計値テーブルを作成
#         stats_table = (
#             df_past
#             .groupby(['時間', '分', 'アトラクション名'], observed=True)['待ち時間']
#             .mean()
#             .reset_index()
#             .rename(columns={'待ち時間':'待ち時間_加工'})
#         )
#         # 統計値を予測時間粒度で四捨五入
#         stats_table['待ち時間_加工'] = round(stats_table['待ち時間_加工'] / parkinfo.predict_granulity) * parkinfo.predict_granulity
#         df = df.drop(columns='待ち時間_加工').merge(
#             stats_table,
#             on=['時間', '分', 'アトラクション名'],
#             how='left'
#         )

#         df['運転停止フラグ'] = 0
#         df['運転停止フラグ'] = df['運転停止フラグ'].astype('category')

#         # カラム順序変更
#         df = df.reindex(columns=df_past.columns)

#     df['アトラクション名'] = df['アトラクション名'].astype('category')

#     return df, null_cols