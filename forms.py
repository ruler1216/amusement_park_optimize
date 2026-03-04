import os
import json
import pandas as pd
import re
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from typing import Dict, List, Any, Optional
# 自作モジュール
from config import Config

"""関数定義"""

##### 最新のアトラクションファイルを読み込む関数
def _load_latest_attractions(dir: str) -> Dict[str, str]:
    """
    Args:
        dir (str): 対象となるディレクトリパス

    Returns:
        Dict[str, str]: {リネーム前のアトラクション名:アトラクション正式名称}
    """
    # {更新日時のタイムスタンプ:ファイルパス}
    files = {
        os.path.getmtime(os.path.join(dir, f)): os.path.join(dir, f)
        for f in os.listdir(dir)
    }
    
    latest_file = files[max(files.keys())]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        res = json.load(f)
    
    return res

##### google formsの認証を行う関数
def _get_credentials(
    token: str,
    client: str,
    scopes: List[str]
    ) -> Credentials:
    """
    Args:
        token (str): トークンパス
        client (str): クライアント情報
        scopes (List[str]): 使用API

    Returns:
        Credentials: google formsの認証情報
    """
    creds = None
    
    # 既存トークンがあれば使用
    if os.path.exists(token):
        creds = Credentials.from_authorized_user_file(token, scopes)
    
    # トークンが無効or存在しない場合は作成
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client, scopes
            )
            creds = flow.run_local_server(port=0)
        
        # token.jsonを保存
        with open(token, 'w') as t:
            t.write(creds.to_json())
    
    return creds

##### Text出力を行う関数
def output_to_text(dirname: str, filename: str, data: str, base_dirname: str=Config.MODE):
    """
    Args:
        dirname (str): ディレクトリ名
        filename (str): ファイル名
        data (str): Text出力を行うデータ
        base_dirname (str): 遊園地ごとのディレクトリ(デフォルト:config.pyのMODEで指定した文字列)

    Returns:
        None
    """
    # アウトプット元のパスを作成
    base_path = os.path.join(os.getcwd(), base_dirname, dirname)
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f'{filename}.txt')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)

##### フォームの質問文フォーマット
def _question_format(
    idx: int,
    attraction: str
    ) -> Dict[str, Any]:
    """
    Args:
        idx (int): インデックス番号
        attraction (str): アトラクション名

    Returns:
        Dict[str, Any]: 質問内容
    """
    # 1~10の10段階で質問
    return  {"createItem": {
                "item": {
                    "title": f"【{attraction}】の満足度を教えてください",
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [
                                    {"value": str(i)} for i in range(1, 11)
                                ]
                                }
                            }
                        }
                    },
                    "location": {
                        "index": idx
                    }
                }
            }

##### google formsのフォーム作成
def create_form(
    dir: str,
    token: str=Config.TOKEN_PATH,
    client: str=Config.GOOGLE_FORM_CLIENT,
    scopes: List[str]=Config.SCOPES,
    forecast_date: str=Config.FORECAST_DATE
    ) -> str:
    """
    Args:
        dir (str): 対象となるディレクトリパス
        token (str, optional): トークンパス(デフォルト:config.pyのTOKEN_PATH)
        client (str, optional): クライアント情報(デフォルト:config.pyのGOOGLE_FORM_CLIENT)
        scopes (List[str], optional): 使用API(デフォルト:config.pyのSCOPES)
        forecast_date (str): 予測日(デフォルト:config.pyのFORECAST_DATEで指定した文字列)

    Returns:
        str: フォームID
    """
    attractions = _load_latest_attractions(dir)
    creds = _get_credentials(token, client, scopes)
    service = build('forms', 'v1', credentials=creds)
    
    # フォーム作成
    form = {
        'info':{
            'title':'満足度アンケート',
            'documentTitle':f'{Config.MODE}_{datetime.today().strftime('%Y-%m-%d')}'
        }
    }
    
    result = service.forms().create(body=form).execute()
    form_id = result['formId']
    
    update_requests = [
        {
            'updateFormInfo':{
                'info':{
                    'description':'1~10で各アトラクションの満足度を教えてください。(1:すごく乗りたくない, 10:すごく乗りたい)'
                },
                'updateMask': 'description'
            }
        }
    ]
    
    service.forms().batchUpdate(
        formId=form_id,
        body={'requests':update_requests}
    ).execute()
    
    output_to_text('フォームID', forecast_date, form_id)
    
    # 質問追加
    requests = []
    
    for idx, attraction in enumerate(attractions.values()):
        requests.append(_question_format(idx, attraction))
    
    service.forms().batchUpdate(
        formId=form_id,
        body={'requests':requests}
    ).execute()
    
    return form_id

##### 各回答を取得する関数
def _get_answer(
    res: Dict[str, Any],
    question_map: Dict[str, Any]
    ) -> Dict[str, Any]:
    """
    Args:
        res (Dict[str, Any]): 全アトラクションのアンケート結果
        questin_map (Dict[str, Any]): questionIdと質問文の対応データ

    Returns:
        Dict[str, Any]: 整形済みアンケート結果
    """
    # 整形後アンケート結果
    result = {}
    # 回答取得
    answers = res.get('answers', {})
    # 整形処理
    for ans in answers.values():
        question_text = question_map.get(ans['questionId'], '不明な質問')
        pattern = re.compile(r'【(.+)】.+')
        qt = re.search(pattern, question_text).group(1)
        if 'textAnswers' in ans:
            val = int(ans['textAnswers']['answers'][0]['value'])
        result[qt] = val
    
    return result

##### google formsから回答を取得する関数
def get_from_responses(
    form_id: str,
    token: str=Config.TOKEN_PATH,
    client: str=Config.GOOGLE_FORM_CLIENT,
    scopes: List[str]=Config.SCOPES
    ) -> Optional[pd.DataFrame]:
    """
    Args:
        form_id (str): 対象となるフォームID
        token (str, optional): トークンパス(デフォルト:config.pyのTOKEN_PATH)
        client (str, optional): クライアント情報(デフォルト:config.pyのGOOGLE_FORM_CLIENT)
        scopes (List[str], optional): 使用API(デフォルト:config.pyのSCOPES)

    Returns:
        Optional[pd.DataFrame]: アンケート回答結果(回答がまだない場合はNone)
    """
    creds = _get_credentials(token, client, scopes)
    service = build('forms', 'v1', credentials=creds)
    
    # フォームのコンテンツを取得
    form_info = service.forms().get(formId=form_id).execute()
    form_items = form_info.get('items', [])
    
    # 質問文とquestionIdの紐づけ
    question_map = {}
    for item in form_items:
        if 'questionItem' not in item:
            continue
        question = item['questionItem']['question']
        question_id = question['questionId']
        title = item['title']
        
        question_map[question_id] = title
    
    response = service.forms().responses().list(
        formId=form_id
    ).execute()
    
    responses = response.get('responses', [])
    
    if len(responses) > 0:
        results = []
        for res in responses:
            results.append(_get_answer(res, question_map))
            
        print(f'{len(responses)}件の回答があったので、DataFrameで取得しました')
        return pd.DataFrame(results)
    else:
        print('まだ回答はありません')
        return None