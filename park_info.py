from dataclasses import dataclass
from typing import List, Tuple

"""クラス定義"""
##### データを入れるためだけのクラスを簡単に書けるデコレータ
@dataclass
class ParkInfo:
    time_window: Tuple[int, int] # 遊園地の営業時間
    lat_lng: Tuple[float, float] # 遊園地の緯度経度
    url: str # 遊園地のスクレイピング対象URL
    url_template: str # 遊園地のスクレイピング対象URLのテンプレート
    radio_btn: str # ラジオボタン切り替えを行うXPATH
    previous_month: str # 前月切り替えを行うXPATH
    calendar: str # カレンダー情報を取得するXPATH
    date: str # 日付を取得するXPATH
    times: List[str] # 過去待ち時間の時間帯
    attraction_name: List[str] # アトラクション名を取得するXPATH
    attraction_correct_name: str # アトラクション正式名称を取得するXPATH
    wait_times: List[str] # アトラクション待ち時間を取得するXPATH
    predict_granulity: int # 予測時間粒度