import pandas as pd
import calendar
import os
import time
import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import lightgbm as lgb
import json
from typing import Dict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
# =====　自作モジュール　=====
import scraping
import cleansing
import predict
import forms
import route_optimize
from config import Config
from park_info import ParkInfo

"""関数定義"""
##### 各データパスを作成する関数
def create_data_path(mode: str=Config.MODE, train_file: str=f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}', forecast_file: str=Config.FORECAST_DATE) -> Dict[str, str]:
    """
    Args:
        mode (str, optional): 遊園地名(デフォルト:config.pyのMODEで指定した文字列)
        train_file (str, optional): 学習データのファイル名(デフォルト:config.pyのTRAIN_PERIODで作成した文字列)※学習データファイル名を指定できるディレクトリのみに使用
        forecast_file (str, optional): 予測データのファイル名(デフォルト:config.pyのFORECAST_DATEで指定した文字列)※予測データファイル名を指定できるディレクトリのみに使用

    Returns:
        Dict[str, str]: [ディレクトリ名:ディレクトリパスorファイルパス]
    """
    
    return {
        'アトラクション正式名称':f'./{mode}/アトラクション正式名称/{forecast_file}.json',
        #'アトラクション正式名称':f'./{mode}/2026-02-06.json', # 検証用
        '天気予報':f'./{mode}/天気予報/{forecast_file}.json',
        #'天気予報':f'./{mode}/2024-02-06_2026-02-04.json', # 検証用
        '過去待ち時間':f'./{mode}/過去待ち時間',
        '過去天気':f'./{mode}/過去天気/{train_file}.json',
        #'過去天気':f'./{mode}/2024-02-06_2026-02-04.json', # 検証用
        '学習データ':f'./{mode}/学習データ/{train_file}.csv',
        '予測データ':f'./{mode}/予測データ/{forecast_file}.csv',
        '予測不可データ':f'./{mode}/予測不可データ/{forecast_file}.json',
        'モデルファイル':f'./{mode}/モデル/{train_file}.pkl',
        '予測結果':f'./{mode}/予測結果/{forecast_file}.csv',
        'フォームID':f'./{mode}/フォームID/{forecast_file}.txt',
        '乗車記録':f'./{mode}/乗車記録/{forecast_file}.json'
    }

"""サンプルコード"""
if __name__ == '__main__':
    """
    ##### whileはgoogleformsの読み込みまで続く。検証に不要なもの(スクレイピング,forms作成/読み込み)はコメントアウトする
    # 検証用
    start_date = '2025-04-01'
    end_date = '2025-05-01'
    #end_date = '2026-02-04'
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    while True:
        # 検証用
        Config.FORECAST_DATE = start_dt.strftime('%Y-%m-%d')
        # 3か月学習
        Config.TRAIN_PRIOD= (
        (start_dt - relativedelta(years=1)).strftime('%Y-%m-%d'),
        (start_dt - relativedelta(days=2)).strftime('%Y-%m-%d')
        )
    """
    """
    # 最適化検証用
    Config.FORECAST_DATE = '2026-02-27'
    Config.TRAIN_PRIOD= (
    '2025-02-27',
    '2026-02-25'
    )
    """
    
    # 各データパス作成
    paths = create_data_path(mode=Config.MODE, train_file=f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}', forecast_file=Config.FORECAST_DATE)
    
    # 遊園地情報の読み込み
    try:
        parkinfo = Config.PARK_CONFIG[Config.MODE]
    except:
        ValueError('config.pyのMODEが不適切です。')
    
    ##### データ取得
    if (not os.path.isfile(paths['過去天気'])) or (not os.path.isfile(paths['天気予報'])):
        # =====スクレイピング周りの設定=====
        # Chromeドライバのパスとオプションを指定
        chrome_options = Options()
        chrome_options.binary_location = Config.CHROMIUM_PATH
        chrome_options.add_argument('--headless')
        
        # Chromeドライバを起動
        driver_path = ChromeDriverManager().install()
        service = Service(executable_path=Config.CHROME_PATH)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # =====スクレイピング実行=====        
        # 指定したurlを開く
        driver.get(parkinfo.url)
        scraping.generate_sleep()
        
        # ページのソースコード取得
        html = driver.page_source
        # BeautifulSoupでHTMLを解析
        soup = BeautifulSoup(html, 'html.parser')
        
        # ラジオボタン切り替えが存在する場合は切り替え
        scraping.switch_radio_needed(driver, parkinfo)
        
        # 最新日のアトラクション名を取得
        attractions_latest = scraping.get_attraction_name(driver, parkinfo)
        # 各アトラクションの正式名称を取得
        attractions_correct_name = scraping.get_attraction_correct_name(driver, parkinfo, forecast_date=Config.FORECAST_DATE)
        
        # 昨日の日付
        base_date = datetime.today() - relativedelta(days=1)
        # すでに取得済みデータの最新年月を取得
        latest_date = scraping.get_latest_date('過去待ち時間')
        # ループフラグ
        flg = True
        
        # アトラクションカラム名が異なるまでループの実施
        while flg:
            # 月ごとのスクレイピング結果を格納するDataFrame
            df_yyyymm = pd.DataFrame()
            # 日付リスト(逆順)
            date_list = scraping.get_date_list(driver, parkinfo, base_date)
            
            # 日付が遅い順にスクレイピング
            for dl in date_list:
                scraping.open_date_page(driver, parkinfo, dl)
                
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # 日付の取得
                date = scraping.get_target_date(driver, parkinfo, base_date, dl)
                
                # すでに取得済みデータの日付に一致したらループ終了
                if date == latest_date:
                    flg = False
                    # CSV出力
                    scraping.output_to_csv('過去待ち時間', base_date.strftime('%Y_%m'), df_yyyymm)
                    break
                
                # 時間帯の加工
                df_dt = scraping.build_date_dateframe(date, parkinfo)
                
                # アトラクションカラム名の取得
                attractions = scraping.get_attraction_name(driver, parkinfo)
                
                
                # 最新アトラクションに不足があった場合は終了
                #diff = set(attractions) - set(attractions_latest)
                #if diff:
                #    flg = False
                #    break
                
                # 待ち時間の取得
                wait_list = scraping.get_wait_times(driver, parkinfo)
                
                if wait_list:
                    df_wait = pd.DataFrame(wait_list, columns=attractions)
                    df_day = pd.concat([df_dt, df_wait], axis=1)
                    df_yyyymm = pd.concat([df_yyyymm, df_day])
            
            # 取得データが存在しないは終了
            if df_yyyymm.empty:
                flg = False
            
            # CSV出力
            scraping.output_to_csv('過去待ち時間', base_date.strftime('%Y_%m'), df_yyyymm)
            print(f'{base_date.strftime('%Y年%m月')}分を取得完了')
            
            # 月末日の作成
            base_date = base_date - relativedelta(months=1)
            last_day = calendar.monthrange(base_date.year, base_date.month)[1]
            base_date = base_date.replace(day=last_day)
            
            # 必要に応じて前月切り替え
            scraping.switch_previous_needed(driver, parkinfo)
        
        # ドライバを終了
        driver.quit()

        # =====天気情報取得=====
        past_weather = scraping.get_past_weather(url=Config.PAST_WEATHER, start_date=Config.TRAIN_PRIOD[0], end_date=Config.TRAIN_PRIOD[1])
        # 検証は過去データを使用するためurlに過去天気を取得するように指示
        forecast_weather = scraping.get_forecast_weather(url=Config.FORECAST_WEATHER, forecast_date=Config.FORECAST_DATE)
    
    ##### google forms作成
    if not os.path.isfile(paths['フォームID']):
        form_id = forms.create_form(f'./{Config.MODE}/アトラクション正式名称')
    else:
        with open(paths['フォームID'], 'r') as f:
            form_id = f.read()
    
    ##### 回答が存在するかを判定(回答がない場合は終了)
    if os.path.isfile(paths['フォームID']):
        res = forms.get_from_responses(form_id)
        # なければ終了
        if res is None:
            print('まだ回答がないため、処理を終了します')
            exit()
    
    ##### 前処理
    if (not os.path.isfile(paths['学習データ'])) or (not os.path.isfile(paths['予測データ'])) or (not os.path.isfile(paths['予測不可データ'])):
        # 過去待ち時間データ
        df_past_time = pd.concat(
            [pd.read_csv(f) for f in cleansing.extract_train_datafile(paths['過去待ち時間'], train_period=Config.TRAIN_PRIOD, forecast_date=Config.FORECAST_DATE)]
        )
        
        # 予測用データ
        df_fore_time = pd.DataFrame(columns=df_past_time.columns)
        df_fore_time['Date'] = [Config.FORECAST_DATE + ' ' + t for t in parkinfo.times]
        
        # 特徴量作成のためにマージ
        df_all = pd.concat([df_past_time, df_fore_time]).drop_duplicates(subset=['Date'], keep='last') # 重複削除は検証用
        
        # 過去天気
        df_past_weather = cleansing.preprocess_weather(paths['過去天気'])
        # 天気予報
        df_fore_weather = cleansing.preprocess_weather(paths['天気予報'])
        
        # リネーム処理
        df_all = cleansing.dataframe_columns_rename(df_all, paths['アトラクション正式名称'])
        # 検証用
        #df_past_time = cleansing.dataframe_columns_rename(df_past_time, paths['アトラクション正式名称'])
        #df_fore_time = cleansing.dataframe_columns_rename(df_fore_time, paths['アトラクション正式名称'])
        
        # 加工
        df_learn, df_fore, null_cols = cleansing.preprocess_data(df_all, parkinfo, train_period=Config.TRAIN_PRIOD, forecast_date=Config.FORECAST_DATE)
        # 検証用
        #df_learn, null_cols = cleansing.preprocess_wait_time(df_past_time, parkinfo, null_cols={}, df_past=None, train_period=Config.TRAIN_PRIOD)
        #df_fore, _= cleansing.preprocess_wait_time(df_fore_time, parkinfo, null_cols, df_learn, train_period=Config.TRAIN_PRIOD)
        
        df_learn = pd.merge(df_learn, df_past_weather, how='left', on='time(iso8601)')
        df_fore = pd.merge(df_fore, df_fore_weather, how='left', on='time(iso8601)')
        
        df_learn = df_learn.drop(columns='time(iso8601)')
        df_fore = df_fore.drop(columns='time(iso8601)')
        
        scraping.output_to_csv('学習データ', f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}', df_learn)
        scraping.output_to_csv('予測データ', f'{Config.FORECAST_DATE}', df_fore)
        scraping.output_to_json('予測不可データ', f'{Config.FORECAST_DATE}', null_cols)
    else:
        df_learn = pd.read_csv(paths['学習データ'])
        df_fore = pd.read_csv(paths['予測データ'])
        with open(paths['予測不可データ'], 'r') as f:
            null_cols = json.load(f)
    
    ##### モデル作成
    if (not os.path.isfile(paths['モデルファイル'])):
        model = predict.create_model(df_learn, parkinfo, filename=f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}')
        # 検証用
        #predict.save_feature_importance(model, filename=f'{Config.TRAIN_PRIOD[0]}_{Config.TRAIN_PRIOD[1]}.png')
    else:
        with open(paths['モデルファイル'], 'rb') as f:
            model = pickle.load(f)
    
    ##### 予測実行
    if (not os.path.isfile(paths['予測結果'])):
        result = predict.create_predict_data(df_fore, model, parkinfo)
        result = predict.dataframe_columns_rename(result)
        scraping.output_to_csv('予測結果', Config.FORECAST_DATE, result)
    else:
        result = pd.read_csv(paths['予測結果'])
    
    """
    # 検証用
    start_dt = start_dt + relativedelta(days=1)
    if start_dt == end_dt:
        break
    """
    
    ##### google forms読み込み
    if os.path.isfile(paths['フォームID']):
        # 予測不可データは削除
        res = res.drop(columns=null_cols.values())
        # 平均値算出
        satisfaction = res.mean(axis=0)
        satisfaction_dict = satisfaction.to_dict()
    
    ##### 乗車記録の作成/読み込み
    if os.path.isfile(paths['乗車記録']):
        with open(paths['乗車記録'], 'r') as f:
            ride_count = json.load(f)
    else:
        ride_count = route_optimize.create_ride_data(result.columns)
        scraping.output_to_json('乗車記録', Config.FORECAST_DATE, ride_count)
    
    # 各時間の計算
    total_time, use_time, executed_time = route_optimize.calc_time(parkinfo)
    
    # 予測結果の前処理
    result, predict_interval = route_optimize.preprocess_predict_df(result)
    
    # アトラクション番号の作成
    ride_num = {idx+1:col for idx, col in enumerate(ride_count.keys())}
    print('=====アトラクション番号:=====')
    for k, v in ride_num.items():
        print(f'{v}:{k}')
    
    # アトラクション乗車回数の設定・入力
    ride_count = route_optimize.ride_overwrite_format(ride_count, ride_num)
    # 上書き保存
    scraping.output_to_json('乗車記録', Config.FORECAST_DATE, ride_count)
    print('乗車記録を上書き保存しました。')
    
    # 乗車回数に応じてsatisfaction_dictを更新
    for key, value in satisfaction_dict.items():
        satisfaction_dict[key] = route_optimize.reflect_ride_data(value, ride_count[key])
    
    # 予測待ち時間と調整項を加味した実際の満足度
    real_satisfaction_dict = {}
    for column, satisfaction in satisfaction_dict.items():
        # 満足度計算用のデータフレーム定義
        to_calc = result[result['Date'] >= executed_time - relativedelta(minutes=predict_interval)][column]
        # 検証用
        #to_calc = result[result['Date'] >= datetime.strptime('2026-02-27 9:00:00', '%Y-%m-%d %H:%M:%S') - relativedelta(minutes=predict_interval)][column]
        real_satisfaction_dict = route_optimize.calc_satisfaction(satisfaction, to_calc, column, real_satisfaction_dict, parkinfo)
    # 検証用
    #route_optimize.plot_satisfaction(parkinfo, real_satisfaction)
    
    df_satisfaction = route_optimize.satisfaction_ranking(satisfaction_dict)
    df_real_satisfaction = route_optimize.satisfaction_ranking(real_satisfaction_dict)
    
    if executed_time < result.loc[1, 'Date']:
        print('=====単純な満足度ランキング=====')
        print(df_satisfaction.to_string(index=False))
    
    print('=====予測時間を考慮した満足度ランキング=====')
    print(df_real_satisfaction.to_string(index=False))
    
    """
    # 検証用
    predict_results = pd.concat(
        [pd.read_csv(os.path.join(f'./{Config.MODE}/予測結果', path)) for path in os.listdir(f'./{Config.MODE}/予測結果')]
        )
    actual_results = pd.concat(
        [pd.read_csv(os.path.join(f'./{Config.MODE}/過去待ち時間', path)) for path in os.listdir(f'./{Config.MODE}/過去待ち時間')]
    )
    actual_results = cleansing.dataframe_columns_rename(actual_results, paths['アトラクション正式名称'])
    num_cols = [column for column in actual_results.columns if column != 'Date']
    actual_results[num_cols] = actual_results[num_cols].apply(pd.to_numeric, errors='coerce')
    actual_results['Date'] = pd.to_datetime(actual_results['Date'], errors='coerce')
    predict_results['Date'] = pd.to_datetime(predict_results['Date'], errors='coerce')
    
    correct_name = open(paths['アトラクション正式名称'], 'r')
    all_columns = json.load(correct_name).values()
    not_predict = open(paths['予測不可データ'], 'r')
    not_use_columns = json.load(not_predict).values()
    columns = list(filter(lambda x: x not in not_use_columns, all_columns))
    
    results = pd.merge(predict_results, actual_results, how='left', on='Date')
    
    summary = predict.calc_mae_by_yyyymm(columns, results, start_date, end_date)
    scraping.output_to_csv('検証', f'{start_date}_{end_date}', summary)
    """