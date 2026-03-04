import pandas as pd
from park_info import ParkInfo
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Tuple, Dict, Any, List

"""設定値"""
class Config:
    CHROMIUM_PATH: str = # Chromiumブラウザパス
    CHROME_PATH: str =  # Chromeドライバパス
    PAST_WEATHER: str =  # 過去天気のAPI
    FORECAST_WEATHER: str = # 天気予報のAPI
    PEOPLE_NUM: int =  # 参加人数
    TRAIN_PRIOD: Tuple[str, str] = (
        # 学習開始日
        # 学習終了日
    ) # 学習期間
    FORECAST_DATE: str = # 予測日
    MODE: str = # 予測対象の遊園地(PARK_CONFIGのkeyから指定)
    PARK_CONFIG: Dict[str, Any] = {
        '遊園地名': ParkInfo(
            time_window= # 遊園地の営業時間(Tuple[int, int])
            lat_lng= # 緯度経度(Tuple[float, float])
            url= # アクセス先URL(str)
            url_template= # URL内で日付切り替えを行う場合のテンプレート(str)
            radio_btn= # ラジオボタン切り替えを行うXPATH(str)
            previous_month= # 前月切り替えを行うXPATH(str)
            calendar= # カレンダー情報を取得するXPATH(str)
            date= # 日付を取得するXPATH(str)
            times= # 時間帯の範囲(List[str])
            attraction_name= # アトラクション名を取得するXPATH(List[str])
            attraction_correct_name= # アトラクションの正式名称を取得するXPATH(str)
            wait_times= # 待ち時間を取得するXPATH(List[str])
            predict_granulity= # 予測時間粒度(int)
        )
    }
    GOOGLE_FORM_CLIENT: str = # google formsのクライアントファイル
    TOKEN_PATH: str = # 認証トークン
    SCOPES: List[str] = # 使用API