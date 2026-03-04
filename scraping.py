import pandas as pd
import urllib3
import os
import re
import requests
import random
import time
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Union, Dict, List, Any, Tuple
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from selenium.webdriver.common.by import By
# =====　自作モジュール　=====
from park_info import ParkInfo
from config import Config

"""関数定義"""

##### 待機時間をランダムに生成する関数
def generate_sleep():
    """
    Args:
        None
    Returns:
        None
    """
    # 待機時間を生成
    sleep_time = random.uniform(5, 10)
    time.sleep(sleep_time)

##### すでに取得済みの最新CSVから取得終了指定日を取得する関数
def get_latest_date(dirname: str, base_dirname: str=Config.MODE) -> Optional[str]:
    """
    Args:
        dirname (str): ディレクトリ名
        base_dirname (str): 遊園地ごとのディレクトリ(デフォルト:config.pyのMODEで指定した文字列)

    Returns:
        Optional[str]: 取得終了指定日(ない場合はNone)
    """
    
    file_map = {}
    
    # インプット元のパスを作成
    base_path = os.path.join(os.getcwd(), base_dirname, dirname)
    
    # パスが存在しない場合は作成
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return None
    
    for f in os.listdir(base_path):
        if not f.endswith('.csv'):
            continue
        
        yyyy_mm = re.search(r'(\d{4})_(\d{2})', f)
        
        if not yyyy_mm:
            continue
        
        year = int(yyyy_mm.group(1))
        month = int(yyyy_mm.group(2))
        
        key = year * 100 + month
        file_map[key] = f
    
    if not file_map:
        return None
    
    latest_key = max(file_map.keys())
    
    target_yyyy_mm = re.search(r'(\d{4})_(\d{2}).csv', file_map[latest_key])
    target_yyyy = int(target_yyyy_mm.group(1))
    target_mm = int(target_yyyy_mm.group(2))
    
    return f'{target_yyyy}年{target_mm}月1日'
    
##### 該当要素のオブジェクトを1つ取得する関数
def safe_element(driver: WebDriver, by: str, value: str) -> Optional[WebElement]:
    """
    Args:
        driver (WebDriver): ウェブドライバ
        by (str): HTMLの属性を指定する文字列
        value (str): 属性値を指定する文字列

    Returns:
        Optional[WebElement]: 該当要素のオブジェクト(ない場合はNone)
    """
    try:
        return driver.find_element(by, value)
    except NoSuchElementException:
        return None

##### 該当要素のオブジェクトをすべて取得する関数
def safe_elements(driver: Union[WebDriver, WebElement], by: str, value: str) -> List[WebElement]:
    """
    Args:
        driver (Union[WebDriver, WebElement]): ウェブドライバ
        by (str): HTMLの属性を指定する文字列
        value (str): 属性値を指定する文字列

    Returns:
        List[WebElement]: 該当要素のオブジェクトリスト(ない場合は空のリスト)
    """
    try:
        return driver.find_elements(by, value)
    except NoSuchElementException:
        return []

##### 該当要素のテキスト(生の文字列)をすべて取得する関数
def safe_attrubutes(driver: Union[WebDriver, WebElement], by: str, value: str) -> List[str]:
    """
    Args:
        driver (Union[WebDriver, WebElement]): ウェブドライバ
        by (str): HTMLの属性を指定する文字列
        value (str): 属性値を指定する文字列

    Returns:
        List[str]: 該当要素のテキストリスト(ない場合は空のリスト)
    """
    objs = safe_elements(driver, by, value)
    
    result: List[str] = []
    if objs:
        for obj in objs:
            result.append(obj.get_attribute('textContent'))
    
    return result

##### 必要に応じてラジオボタンによる切り替えを行う関数
def switch_radio_needed(driver: WebDriver, parkinfo: ParkInfo):
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        None
    """
    if parkinfo.radio_btn:
        switch_btn = safe_element(driver, By.XPATH, parkinfo.radio_btn)
        switch_btn.click()
        generate_sleep()

##### アトラクション名を取得する関数
def get_attraction_name(driver: WebDriver, parkinfo: ParkInfo) -> List[str]:
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        List[str]: アトラクション名リスト
    """
    attractions = []
    for xpath in parkinfo.attraction_name:
        attractions.extend(safe_attrubutes(driver, By.XPATH, xpath))
        
    # 元データの順序を保持しつつ重複削除
    return unique_data(attractions)

#### JSON出力を行う関数
def output_to_json(dirname: str, filename: str, data: Dict[str, Any], base_dirname: str=Config.MODE):
    """
    Args:
        dirname (str): ディレクトリ名
        filename (str): ファイル名
        data (Dict[str, Any]): JSON出力を行うデータ
        base_dirname (str): 遊園地ごとのディレクトリ(デフォルト:config.pyのMODEで指定した文字列)

    Returns:
        None
    """
    # アウトプット元のパスを作成
    base_path = os.path.join(os.getcwd(), base_dirname, dirname)
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f'{filename}.json')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
##### アトラクションの正式名称を取得しJSON出力する関数
def get_attraction_correct_name(driver: WebDriver, parkinfo: ParkInfo, forecast_date: str=Config.FORECAST_DATE) -> Dict[str, str]:
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        forecast_date (str): 予測日(デフォルト:config.pyのFORECAST_DATEで指定した文字列)

    Returns:
        Dict[str, str]: {アトラクション名:アトラクション正式名称}
    """
    attractions_correct = {}
    if parkinfo.attraction_correct_name:
        # 各アトラクションごとに繰り返し
        for xpath in parkinfo.attraction_name:
            for element in safe_elements(driver, By.XPATH, xpath):
                # アトラクション正式名称が出現するようにクリック
                driver.execute_script('arguments[0].click();', element)
                # 追加
                attractions_correct[element.get_attribute('textContent')] = safe_attrubutes(driver, By.XPATH, parkinfo.attraction_correct_name)[0]
    else:
        # 各アトラクションごとに繰り返し
        for xpath in parkinfo.attraction_name:
            for element in safe_elements(driver, By.XPATH, xpath):
                attractions_correct[element.get_attribute('textContent')] = element.get_attribute('textContent')
        
    output_to_json('アトラクション正式名称', forecast_date, attractions_correct)
    
    return attractions_correct

##### 指定日月初から指定日までの日付を生成する関数
def generate_dates(date: datetime) -> List[Tuple[int, int, int]]:
    """
    Args:
        date (datetime): 指定日

    Returns:
        List[Tuple[int, int, int]]: (年, 月, 日)
    """
    start_date = date.replace(day=1)
    
    date_list = [(d.year, d.month, d.day) for d in pd.date_range(start=start_date, end=date)]
    
    return date_list

##### 各年月のページ遷移処理を行う関数
def open_date_page(driver: WebDriver, parkinfo: ParkInfo, date: Union[WebElement, Tuple[int, int, int]]) -> bool:
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        date (Union[WebElement, Tuple[int, int, int]]): ウェブエレメントor年月日情報

    Returns:
        bool: ブール値(URLアクセス時にreturnするための値)
    """
    if parkinfo.calendar:
        date.click()
    else:
        url = parkinfo.url_template.format(year=date[0], month=date[1], day=date[2])
        # タイムアウト時にリトライする関数
        for _ in range(5):
            try:
                driver.get(url)
                generate_sleep()
                return True
            except (TimeoutException, urllib3.exceptions.ReadTimeoutError, WebDriverException):
                generate_sleep()
    
    return False

##### 日付リストを取得する関数
def get_date_list(driver: WebDriver, parkinfo: ParkInfo, date: datetime) -> List[str]:
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        date (datetime): 指定日

    Returns:
        List[str]: 逆順の日付リスト
    """
    if parkinfo.calendar:
        date_list = safe_elements(driver, By.XPATH, parkinfo.calendar)
    else:
        date_list = generate_dates(date)
    
    # 逆順
    return date_list[::-1]

##### スクレイピングを行う日付情報を取得する関数
def get_target_date(
    driver: WebDriver, 
    parkinfo: ParkInfo, 
    base_date: datetime, 
    date: Union[WebElement, Tuple[int, int, int]]
    ) -> str:
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス
        base_date (datetime): 年情報を保持した日付
        date (Union[WebElement, Tuple[int, int, int]]): ウェブエレメントor年月日情報

    Returns:
        str: 年月日情報
    """
    if parkinfo.date:
        date = safe_attrubutes(driver, By.XPATH, parkinfo.date)
        return base_date.strftime('%Y年') + date[0]
    else:
        return f'{date[0]}年{date[1]}月{date[2]}日'

##### 日付に関するデータフレームを作成する関数
def build_date_dateframe(date: str, parkinfo: ParkInfo) -> pd.DataFrame:
    """
    Args:
        date (str): 年月日情報
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        pd.DataFrame: 日付に関するデータフレーム
    """
    dt_list = [datetime.strptime(date+t, '%Y年%m月%d日%H:%M').strftime('%Y-%m-%d %H:%M') for t in parkinfo.times]
    
    return pd.DataFrame(dt_list, columns=['Date'])

##### アトラクション待ち時間を取得する関数
def get_wait_times(driver: WebDriver, parkinfo: ParkInfo) -> List[List[str]]:
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        List[List[str]]: [各時間帯ごとのアトラクション待ち時間]
    """
    wait_list = []
    
    for xpath in parkinfo.wait_times:
        values = safe_attrubutes(driver, By.XPATH, xpath)
        if values:
            wait_list.append(values)
    
    return wait_list

##### 必要に応じて前月ボタンによる切り替えを行う関数
def switch_previous_needed(driver: WebDriver, parkinfo: ParkInfo):
    """
    Args:
        driver (WebDriver): ウェブドライバ
        parkinfo (ParkInfo): 遊園地情報のインスタンス

    Returns:
        None
    """
    if parkinfo.previous_month:
        previous_button = safe_element(driver, By.XPATH, parkinfo.previous_month)
        previous_button.click()
        generate_sleep()

##### 過去天気をJSON形式で取得
def get_past_weather(
    url: str=Config.PAST_WEATHER, 
    start_date: str=Config.TRAIN_PRIOD[0], 
    end_date: str=Config.TRAIN_PRIOD[1],
    lat: float=Config.PARK_CONFIG[Config.MODE].lat_lng[0],
    lng: float=Config.PARK_CONFIG[Config.MODE].lat_lng[1]
    ) -> Optional[Dict[str, Any]]:
    """
    Args:
        url (str): 過去天気情報を取得するAPI(デフォルト:config.pyのPAST_WEATHERで指定した文字列)
        start_date (str): データ取得開始日(デフォルト:config.pyのTRAIN_PERIOD[0]で指定した文字列)
        end_date (str): データ取得終了日(デフォルト:config.pyのTRAIN_PERIOD[1]で指定した文字列)
        lat (float): 遊園地の緯度(デフォルト:config.pyのPARK_CONFIG内の緯度経度で指定した値)
        lng (float): 遊園地の緯度(デフォルト:config.pyのPARK_CONFIG内の緯度経度で指定した値)

    Returns:
        Optional[Dict[str, Any]]: JSON形式の過去天気情報(エラーの場合はNone)
    """
    # 取得パラメータ
    params: Dict[str, Any] = {
        'latitude': lat, # 必須
        'longitude': lng, # 必須
        'start_date': start_date, # 必須
        'end_date': end_date, # 必須
        'temperature_unit': 'celsius', # 温度単位を明示(°C表記)
        'wind_speed_unit': 'ms', # 風速単位を明示(m/s表記)
        'precipitation_unit': 'mm', # 降水量単位を明示(mm表記)
        'timeformat': 'iso8601', # 日時表記を明示
        'timezone': 'Asia/Tokyo', # タイムゾーン
        # 取得データ
        'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,pressure_msl,precipitation,rain,snowfall,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,shortwave_radiation,direct_radiation,diffuse_radiation,global_tilted_irradiance,wind_speed_10m,wind_direction_10m,wind_gusts_10m,et0_fao_evapotranspiration,weather_code,snow_depth'
    }
    
    # レスポンス
    res = requests.get(url, params=params)
    generate_sleep()
    
    # JSON形式にデコード
    data = res.json()
    
    # Errorが出た場合
    if data.get('error') is True:
        return None
    
    output_to_json('過去天気', f'{start_date}_{end_date}', data)
    
    return data

##### 天気予報をJSON形式で取得
def get_forecast_weather(
    url: str=Config.FORECAST_WEATHER, 
    forecast_date: str=Config.FORECAST_DATE,
    lat: float=Config.PARK_CONFIG[Config.MODE].lat_lng[0],
    lng: float=Config.PARK_CONFIG[Config.MODE].lat_lng[1]
    ) -> Optional[Dict[str, Any]]:
    """
    Args:
        url (str): 過去天気情報を取得するAPI(デフォルト:config.pyのPAST_WEATHERで指定した文字列)
        forecast_date (str): 予測日(デフォルト:config.pyのFORECAST_DATEで指定した文字列)
        lat (float): 遊園地の緯度(デフォルト:config.pyのPARK_CONFIG内の緯度経度で指定した値)
        lng (float): 遊園地の緯度(デフォルト:config.pyのPARK_CONFIG内の緯度経度で指定した値)

    Returns:
        Optional[Dict[str, Any]]: JSON形式の過去天気情報(エラーの場合はNone)
    """
    # 取得パラメータ
    params: Dict[str, Any] = {
        'latitude': lat, # 必須
        'longitude': lng, # 必須
        'start_date': forecast_date, # データ取得開始日
        'end_date': forecast_date, # データ取得終了日
        'temperature_unit': 'celsius', # 温度単位を明示(°C表記)
        'wind_speed_unit': 'ms', # 風速単位を明示(m/s表記)
        'precipitation_unit': 'mm', # 降水量単位を明示(mm表記)
        'timeformat': 'iso8601', # 日時表記を明示
        'timezone': 'Asia/Tokyo', # タイムゾーン
        # 取得データ
        'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,pressure_msl,precipitation,rain,snowfall,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,shortwave_radiation,direct_radiation,diffuse_radiation,global_tilted_irradiance,wind_speed_10m,wind_direction_10m,wind_gusts_10m,et0_fao_evapotranspiration,weather_code,snow_depth'
    }
    
    # レスポンス
    res = requests.get(url, params=params)
    generate_sleep()
    
    # JSON形式にデコード
    data = res.json()
    
    # Errorが出た場合
    if data.get('error') is True:
        return None
    
    output_to_json('天気予報', f'{forecast_date}', data)
    
    return data

##### 順序を保ったまま重複削除を行う関数
def unique_data(data: List[str]) -> List[str]:
    """
    Args:
        data (List): 重複が存在しているリスト

    Returns:
        List(str): 元の順序を保ったまま重複削除したリスト
    """
    # 重複削除
    unique = set(data)
    
    return sorted(unique, key=data.index)

##### CSV出力を行う関数
def output_to_csv(dirname: str, filename: str, df: pd.DataFrame, base_dirname: str=Config.MODE):
    """
    Args:
        dirname (str): ディレクトリ名
        filename (str): ファイル名
        df (pd.DataFrame): CSV出力を行うデータ
        base_dirname (str): 遊園地ごとのディレクトリ(デフォルト:config.pyのMODEで指定した文字列)

    Returns:
        None
    """
    # アウトプット元のパスを作成
    base_path = os.path.join(os.getcwd(), base_dirname, dirname)
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f'{filename}.csv')
    
    df.to_csv(file_path, index=False, encoding='utf-8')