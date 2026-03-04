import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import japanize_matplotlib
import optuna
import pickle
import os
import re
from datetime import datetime
from typing import Dict, Any, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
# 自作モジュール
from config import Config
from park_info import ParkInfo

"""関数定義"""

##### 特徴量重要度を保存する関数(検証用)
def save_feature_importance(
    model: lgb.LGBMRegressor, 
    path: str=f'{Config.MODE}/検証/特徴量重要度/', 
    filename: str=f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}.png'
    ):
    """
    Args:
        model (lgb.LGBMRegressor): 学習済みモデル
        path (str, optional): ディレクトリパス(デフォルト:f'{Config.MODE}/検証/特徴量重要度/')
        filename (str, optional): ファイル名(デフォルト:f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}.png')

    Returns:
        None
    """
    lgb.plot_importance(model, figsize=(8, 4))
    
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

##### 指定期間の各アトラクションの年月ごとのMAEを計算する関数(検証用)
def calc_mae_by_yyyymm(
    columns: List[str],
    df: pd.DataFrame,
    start_date: str,
    end_date: str
    ) -> pd.DataFrame:
    """
    Args:
        columns (List[str]): アトラクションカラム名
        df (pd.DataFrame): 予測結果
        start_date (str): 集計開始期間
        end_date (str): 集計終了期間

    Returns:
        pd.DataFrame: 集計結果
    """
    start_dt = datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_date + ' 23:59:00', '%Y-%m-%d %H:%M:%S')
    
    df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy()
    
    # 年月カラムの作成
    df['年月'] = df['Date'].dt.to_period('M').astype(str)
    
    mae_cols = []
    
    # アトラクションごとに絶対誤差を算出
    for col in columns:
        mae_col = f'{col}_絶対誤差'
        mae_cols.append(mae_col)
        
        df[mae_col] = (df[col] - df[f'{col}_待ち時間_予測値']).abs()
        
        # 実績が0or欠損は除外
        df.loc[(df[col].isna()) | (df[col] == 0), mae_col] = np.nan
    
    df = df[['年月'] + mae_cols]
    
    df_long = df.melt(
        id_vars=['年月'],
        value_vars=mae_cols,
        var_name='アトラクション',
        value_name='MAE'
    )
    
    df_long['アトラクション'] = df_long['アトラクション'].str.replace('_絶対誤差', '', regex=True)
    
    summary = df_long.groupby(['年月', 'アトラクション'], observed=True)['MAE'].mean().reset_index().sort_values(['年月', 'アトラクション'])
    summary = summary.dropna(subset=['MAE'])
    
    summary_all = summary.groupby(['アトラクション'], observed=True)['MAE'].mean().reset_index().sort_values(['アトラクション'])
    summary_all['年月'] = '全体'
    
    summary = pd.concat([summary, summary_all])

    return summary

##### 作成したモデルを保存する関数
def save_model(dirname: str, filename: str, model: lgb.LGBMRegressor, base_dirname: str=Config.MODE):
    """
    Args:
        dirname (str): ディレクトリ名
        filename (str): ファイル名
        model (lgb.LGBMRegressor): 学習モデル
        base_dirname (str, optional): 遊園地ごとのディレクトリ(デフォルト:config.pyのMODEで指定した文字列)

    Returns:
        None
    """
    # アウトプット元のパスを作成
    base_path = os.path.join(os.getcwd(), base_dirname, dirname)
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f'{filename}.pkl')
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

##### 時系列でクロスバリデーションを行う関数
def time_cv_mae(
    df: pd.DataFrame,
    params: Dict[str, Any],
    feature_cols: List[str],
    target_col: str,
    parkinfo: ParkInfo
    ) -> float:
    """
    Args:
        df (pd.DataFrame): 学習用データ
        params (Dict[str, Any]): {ハイパーパラメータ名: 値}
        feature_cols (List[str]): 説明変数リスト
        target_col (str): 目的変数(デフォルト:待ち時間)
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        float: 遊園地全体のMAE
    """
    # ユニークな時系列ソート, アトラクション名の抽出
    unique_date = df['Date'].sort_values().unique()
    attraction_names = df['アトラクション名'].unique()
    
    # 各アトラクションMAE
    scores = {name:[] for name in attraction_names}
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # リークにならないよう時系列を保ち、分割
    for train_idx, valid_idx in tscv.split(unique_date):
        train_times = unique_date[train_idx]
        valid_times = unique_date[valid_idx]
        
        # 学習/検証データ作成
        train = df[df['Date'].isin(train_times)]
        valid = df[df['Date'].isin(valid_times)]
        
        # データ分割
        train_x, train_y = train[feature_cols], train[target_col]
        valid_x = valid[feature_cols]
        
        # モデル作成/予測
        model = lgb.LGBMRegressor(**params)
        model.fit(train_x, train_y)
        pred = model.predict(valid_x)
        
        # 検証データに予測値結合
        valid_result = valid.copy()
        valid_result['pred'] = pred
        
        # 予測時間粒度で四捨五入
        valid_result['pred'] = round(valid_result['pred'] / parkinfo.predict_granulity) * parkinfo.predict_granulity
        
        # 各アトラクションごとにMAEを算出
        for name in attraction_names:
            mask = valid_result['アトラクション名'] == name
            
            # 実績値と予測値
            y_true = valid_result.loc[mask, target_col]
            y_pred = valid_result.loc[mask, 'pred']
            
            valid_mask = (~y_true.isna())&(~y_pred.isna())
            
            # 欠損ではないデータのみ抽出
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            
            if len(y_true) == 0:
                continue
            
            # MAE
            mae = mean_absolute_error(
                y_true,
                y_pred
            )
            
            # スコアにMAEを登録
            scores[name].append(mae)
    
    # 遊園地全体のMAE
    all_mae = []
    
    for name, values in scores.items():
        print(name, 'MAE:', np.mean(values))
        if not np.isnan(np.mean(values)):
            all_mae.append(np.mean(values))
    
    print('アトラクション全体のMAE:', np.mean(all_mae))
    
    return np.mean(all_mae)

##### Optunaのパラメータチューニングを行う関数
def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col:str,
    parkinfo: ParkInfo
    ) -> float:
    """
    Args:
        trial (optuna.Trial): ハイパーパラメータの提案, 取得や評価結果を記録するためのインターフェース(自動的に受け渡し)
        df (pd.DataFrame): 学習用データ
        feature_cols (List[str]): 説明変数リスト
        target_col (str): 目的変数
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        float: 遊園地全体のMAE
    """
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        'random_state': 42
    }
    
    mae = time_cv_mae(df, params, feature_cols, target_col, parkinfo)
    
    return mae

##### モデルを作成する関数
def create_model(
    df: pd.DataFrame, 
    parkinfo: ParkInfo,
    target_col: str='待ち時間', 
    filename: str=f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}'
    ) -> lgb.LGBMRegressor:
    """
    Args:
        df (pd.DataFrame): 学習データ
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        target_col (str, optional): 目的変数(デフォルト:待ち時間)
        filename (str, optional): 保存ファイル名(デフォルト:f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}')

    Returns:
        lgb.LGBMRegressor: 学習モデル
    """
    # 説明変数リスト
    feature_cols = [c for c in df.columns if c not in [target_col, 'Date']]
    
    # ハイパーパラメータの最適化
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, df, feature_cols, target_col, parkinfo),
        n_trials=100
        )
    best_params = study.best_params
    print(f'最適なパラメータ:{best_params}')
    
    train_x, train_y = df[feature_cols], df[target_col]
    
    # 学習
    model = lgb.LGBMRegressor(**best_params)
    model.fit(train_x, train_y)
    
    save_model('モデル', filename, model)
    
    return model

##### 予測を行う関数
def create_predict_data(
    df: pd.DataFrame,
    model: lgb.LGBMRegressor,
    parkinfo: ParkInfo,
    target_col: str='待ち時間',
    ) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): 予測用データ
        model (lgb.LGBMRegressor): 学習モデル
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        target_col (str, optional): 目的変数(デフォルト:待ち時間)

    Returns:
        pd.DataFrame: 予測結果
    """
    # 説明変数リスト
    feature_cols = [c for c in df.columns if c not in [target_col, 'Date']]
    
    # 予測
    pred = model.predict(df[feature_cols])
    
    result = df[['Date', 'アトラクション名']].copy()
    result['待ち時間_予測値'] = pred
    
    # 予測時間粒度で四捨五入
    result['待ち時間_予測値'] = round(result['待ち時間_予測値'] / parkinfo.predict_granulity) * parkinfo.predict_granulity
    
    # 横持ちデータに変換
    result_long = result.melt(
        id_vars=['Date', 'アトラクション名'],
        value_vars=['待ち時間_予測値'],
        var_name='予測結果',
        value_name='値'
    )
    
    result_pivot = result_long.pivot(
        index='Date',
        columns=['アトラクション名', '予測結果'],
        values='値'
    ).reset_index()
    
    result_pivot.columns = [
        f'{a}_{b}' for a, b in result_pivot.columns
    ]
    result_pivot = result_pivot.rename(columns={'Date_':'Date'})
    
    return result_pivot

##### カラム名のリネームを行う関数(後処理用)
def dataframe_columns_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): カラムのリネーム前のデータフレーム

    Returns:
        pd.DataFrame: カラムのリネーム後のデータフレームもしくはリネームの必要がないデータフレーム
    """
    pattern = re.compile(r'(.+)_待ち時間_予測値')
    
    rename_dict = {col:re.search(pattern, col).group(1) for col in df.columns if col != 'Date'}
    df = df.rename(columns=rename_dict)
    
    return df