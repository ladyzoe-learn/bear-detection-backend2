# src/main.py (Final Version with Refactored Logic)

import traceback
import requests
import io
import base64
import os
import pandas as pd
import folium
import cv2
import numpy as np
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from folium.plugins import MarkerCluster
from datetime import datetime
from flask_caching import Cache

# --- 1. 集中讀取所有環境變數 ---
# Telegram
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
# Hugging Face
HF_API_URL = os.getenv("HF_API_URL", "https://ladyzoe-bear-detector-api-docker.hf.space/predict")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- Telegram Bot 類別定義 ---
class TelegramBot:
    def __init__(self, bot_token, chat_id):
        # 檢查傳入的參數是否有效
        if not bot_token or not chat_id:
            print("Warning: Telegram Bot token or chat_id is missing. Notifications will be disabled.")
            self.bot_token = None
            self.chat_id = None
        else:
            self.bot_token = bot_token
            self.chat_id = chat_id
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message):
        if not self.bot_token: return None # 如果初始化失敗，則不執行
        url = f"{self.base_url}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            print("Telegram message sent successfully.")
            return response.json()
        except Exception as e:
            print(f"發送 Telegram 訊息錯誤: {e}")
            return None

    def send_photo(self, photo_url, caption=""):
        if not self.bot_token: return None # 如果初始化失敗，則不執行
        url = f"{self.base_url}/sendPhoto"
        data = {"chat_id": self.chat_id, "photo": photo_url, "caption": caption, "parse_mode": "HTML"}
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            print("Telegram photo sent successfully.")
            return response.json()
        except Exception as e:
            print(f"發送 Telegram 圖片錯誤: {e}")
            return None

# --- 2. 使用讀取到的環境變數來初始化 Bot ---
telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# --- 通知發送的共用函式 ---
def send_bear_alert(confidence, image_url=None, location=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_message = "熊蹤跡預警，照片偵測到 台灣黑熊並即將進入生活共同圈，請保持安全距離並提高警覺！"

    if image_url:
        return telegram_bot.send_photo(image_url, alert_message)
    else:
        return telegram_bot.send_message(alert_message)

# --- Flask App 初始化與設定 ---
app = Flask(__name__)
CORS(app, origins=["https://bear-detection-app.onrender.com", "http://localhost:5173"])
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 3600})
cache.init_app(app)


# --- 偵測相關的共用函式 ---
def detect_objects_in_image_data(image_bytes):
    if not HF_API_TOKEN:
        print("Error: Hugging Face API Token (HF_API_TOKEN) is not set.")
        return None
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        safe_filename = "upload.jpg"
        hf_files = {'file': (safe_filename, image_bytes, 'image/jpeg')}
        hf_response = requests.post(HF_API_URL, headers=headers, files=hf_files)
        hf_response.raise_for_status()
        return hf_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Hugging Face API request failed: {e}")
        return None

# 3. 修改 is_bear_detected 函式，讓它回傳信心度
def is_bear_detected(api_response):
    """檢查 API 回應中是否包含 'kumay'，並回傳偵測結果和最高信心度"""
    if api_response and isinstance(api_response.get('detections'), list):
        bear_detections = [item for item in api_response['detections'] if isinstance(item, dict) and item.get('label') == 'kumay']
        if bear_detections:
            highest_confidence = max(item.get('confidence', 0) for item in bear_detections)
            return True, highest_confidence
    return False, 0.0

# --- API 端點 ---
@app.route('/api/detect', methods=['POST'])
def detect_bear_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳圖片檔案"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇檔案"}), 400

    try:
        image_bytes = file.read()
        api_response = detect_objects_in_image_data(image_bytes)
        
        if not api_response:
             return jsonify({"success": False, "error": "模型偵測失敗，請檢查後端日誌"}), 500

        # 4. 使用新的 is_bear_detected 函式來獲取真實的信心度
        bear_is_detected, confidence = is_bear_detected(api_response)
        
        alert_sent = False
        if bear_is_detected and confidence >= 0.7: # 使用真實信心度判斷
            print("Image detection: Bear detected! Sending Telegram alert...")
            # image_url 暫時為 None，若有圖片上傳服務可補上
            send_bear_alert(confidence=confidence, image_url=None, location="系統偵測區域")
            alert_sent = True
        
        response_data = {
            "success": True,
            "bear_detected": bear_is_detected,
            "confidence": confidence,
            "processed_image": base64.b64encode(image_bytes).decode('utf-8'),
            "alert_sent": alert_sent
        }
        return jsonify(response_data)
        
    except Exception as e:
        print(f"圖片偵測時發生錯誤: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": "伺服器處理圖片時發生錯誤"}), 500

@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳影片檔案"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇檔案"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(request.files['video'].read())
        temp_video_path = temp.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        highest_confidence_in_video = 0.0
        consecutive_bear_frames = 0 # 定義並初始化變數
        consecutive_frames_needed = 3 # 定義並初始化變數，可根據需求調整
        alert_sent = False # 定義並初始化變數

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # ... (抽幀邏輯不變)

            api_response = detect_objects_in_image_data(cv2.imencode(".jpg", frame)[1].tobytes())

            detected, confidence = is_bear_detected(api_response) # 5. 使用新函式

            if detected:
                consecutive_bear_frames += 1
                highest_confidence_in_video = max(highest_confidence_in_video, confidence)
                print(f"  BEAR DETECTED! (Confidence: {confidence:.2%}) Consecutive frames: {consecutive_bear_frames}")
            else:
                consecutive_bear_frames = 0 # 修正縮排

            if not alert_sent and consecutive_bear_frames >= consecutive_frames_needed:
                print("!!! ALERT TRIGGERED !!!")
                # 6. 使用在影片中偵測到的最高信心度發送通知
                send_bear_alert(confidence=highest_confidence_in_video, image_url=None, location="影片偵測區域")
                alert_sent = True
                break

        cap.release() # 修正縮排
        os.remove(temp_video_path) # 修正縮排

        response_data = {
            "success": True,
            "bear_detected": highest_confidence_in_video >= 0.7,
            "confidence": highest_confidence_in_video,
            "alert_sent": alert_sent
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"影片分析時發生錯誤: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": "伺服器處理影片時發生錯誤"}), 500

@app.route('/api/map', methods=['GET'])
@cache.cached(query_string=True)  # 根據不同 query string 快取對應 map
def get_bear_map():
    try:
        file_path = 'src/台灣黑熊.csv'
        df = pd.read_csv(file_path)

        # 轉換 eventdate 欄位為 datetime
        df['eventdate'] = pd.to_datetime(df['eventdate'], errors='coerce')
        df = df.dropna(subset=['eventdate'])  # 避免無效日期造成錯誤

        # 解析前端傳入的起訖日期（若有）
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                df = df[(df['eventdate'] >= start_dt) & (df['eventdate'] <= end_dt)]
            except Exception as e:
                return jsonify({"success": False, "error": "日期格式錯誤，請使用 YYYY-MM-DD"}), 400

        # 建立地圖物件
        taiwan_map = folium.Map(location=[23.97565, 120.97388], zoom_start=7)
        marker_cluster = MarkerCluster().add_to(taiwan_map)

        # 加入地圖標記
        for _, row in df.iterrows():
            popup_html = f"""
                <b>物種:</b> {row['vernacularname']}<br>
                <b>日期:</b> {row['eventdate'].date()}<br>
                <b>紀錄者:</b> {row['recordedby']}
            """
            iframe = folium.IFrame(popup_html, width=200, height=100)
            popup = folium.Popup(iframe, max_width=200)
            folium.Marker(
                location=[row['verbatimlatitude'], row['verbatimlongitude']],
                popup=popup,
                tooltip=f"{row['vernacularname']} - {row['eventdate'].date()}"
            ).add_to(marker_cluster)

        # 轉成 HTML
        map_html = taiwan_map._repr_html_()
        return jsonify({"success": True, "map_html": map_html})

    except FileNotFoundError:
        return jsonify({"success": False, "error": "找不到地圖資料檔案"}), 404
    except Exception as e:
        print(f"地圖產生失敗: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": "產生熱點圖時發生錯誤"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)