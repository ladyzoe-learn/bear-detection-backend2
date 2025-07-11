# src/main.py (Final Version with Caching and Flexible Date Filtering)

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

# --- Telegram Bot 相關設定 ---
class TelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    def send_message(self, message):
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        try:
            response = requests.post(url, json=data)
            return response.json()
        except Exception as e:
            print(f"發送訊息錯誤: {e}")
            return None
    def send_photo(self, photo_url, caption=""):
        url = f"{self.base_url}/sendPhoto"
        data = {
            "chat_id": self.chat_id,
            "photo": photo_url,
            "caption": caption,
            "parse_mode": "HTML"
        }
        try:
            response = requests.post(url, json=data)
            return response.json()
        except Exception as e:
            print(f"發送圖片錯誤: {e}")
            return None

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

def send_bear_alert(confidence, image_url=None, location=None, timestamp=None, source_type="照片"):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 根據來源類型決定訊息內容
    if source_type == "影片":
        alert_message = "熊蹤跡預警，影片偵測到 台灣黑熊並即將進入生活共同圈，請保持安全距離並提高警覺！"
    else:
        alert_message = "熊蹤跡預警，照片偵測到 台灣黑熊並即將進入生活共同圈，請保持安全距離並提高警覺！"
    if image_url:
        result = telegram_bot.send_photo(image_url, alert_message)
    else:
        result = telegram_bot.send_message(alert_message)
    return result

# --- Flask App 初始化與設定 ---
app = Flask(__name__)
# 我們在許可名單中，同時加入「線上前端網址」和「本地前端網址」
CORS(app, origins=["https://bear-detection-app.onrender.com", "http://localhost:5173"])

# --- 快取設定 ---
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 3600})
cache.init_app(app)

# --- 環境變數讀取 ---
HF_API_URL = os.getenv("HF_API_URL", "https://ladyzoe-bear-detector-api-docker.hf.space/predict")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
print(f"--- DEBUG: HF_API_TOKEN loaded: {str(HF_API_TOKEN)[:5]}...") 

# --- 偵測相關的共用函式 ---
def detect_objects_in_image_data(image_bytes):
    if not HF_API_TOKEN:
        print("Error: Hugging Face API Token (HF_API_TOKEN) is not set.")
        return None
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        # ✅【核心修改】將檔名固定為安全的 ASCII 字元，避免因中文檔名等造成編碼錯誤
        safe_filename = "upload.jpg"
        hf_files = {'file': (safe_filename, image_bytes, 'image/jpeg')}
        hf_response = requests.post(HF_API_URL, headers=headers, files=hf_files)
        # print(f"--- DEBUG: Hugging Face response status code: {hf_response.status_code}")
        # print(f"--- DEBUG: Hugging Face response text: {hf_response.text}")
        hf_response.raise_for_status()
        return hf_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Hugging Face API request failed: {e}")
        return None

def is_bear_detected(api_response):
    if api_response and isinstance(api_response.get('detections'), list):
        for item in api_response['detections']:
            if isinstance(item, dict) and item.get('label') == 'kumay':
                return True
    return False

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
        bear_is_detected = is_bear_detected(api_response)
        confidence = 0.99 if bear_is_detected else 0
        alert_sent = False
        image_url = None # 若有圖片上傳服務可補上
        if bear_is_detected and confidence >= 0.7:
            print("Image detection: Bear detected! Sending Telegram alert...")
            send_bear_alert(confidence=confidence, image_url=image_url, location="系統偵測區域", source_type="照片")
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

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇檔案"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(video_file.read())
        temp_video_path = temp.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "無法打開影片檔案"}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30 # 如果讀不到FPS，給一個預設值

        alert_threshold_seconds = 2.0
        frames_to_process_per_second = 0.5
        frames_to_skip = max(1, int(fps / frames_to_process_per_second))
        consecutive_frames_needed = int(alert_threshold_seconds * frames_to_process_per_second)

        consecutive_bear_frames = 0
        max_consecutive_duration = 0.0
        alert_sent = False
        frame_count = 0
        
        print(f"Analyzing video: FPS={fps}, Processing {frames_to_process_per_second} frames/sec.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % frames_to_skip != 0: continue

            print(f"Processing frame #{frame_count}...")
            
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success: continue
            image_bytes = buffer.tobytes()

            api_response = detect_objects_in_image_data(image_bytes)
            
            if is_bear_detected(api_response):
                consecutive_bear_frames += 1
                print(f"  BEAR DETECTED! Consecutive frames: {consecutive_bear_frames}")
            else:
                current_duration = (consecutive_bear_frames / frames_to_process_per_second)
                max_consecutive_duration = max(max_consecutive_duration, current_duration)
                consecutive_bear_frames = 0

            if not alert_sent and consecutive_bear_frames >= consecutive_frames_needed:
                print("!!! ALERT TRIGGERED !!!")
                send_bear_alert(confidence=0.99, image_url=None, location="影片偵測區域", source_type="影片")
                alert_sent = True
                break
        
        final_duration = (consecutive_bear_frames / frames_to_process_per_second)
        max_consecutive_duration = max(max_consecutive_duration, final_duration)

        response_data = {
            "success": True,
            "alert_sent": alert_sent,
            "max_consecutive_duration_seconds": round(max_consecutive_duration, 2),
            "video_fps": fps,
        }
        return jsonify(response_data)
    finally:
        cap.release()
        os.unlink(temp_video_path)
        print("Video analysis complete. Temporary file deleted.")

@app.route('/api/map', methods=['GET'])
# @cache.cached(query_string=True) # 我們改變了回傳內容，先移除快取
def get_bear_map():
    print("Providing map data...")
    try:
        file_path = 'src/台灣黑熊.csv'
        df = pd.read_csv(file_path)
        df['eventdate'] = pd.to_datetime(df['eventdate'], errors='coerce')
        df.dropna(subset=['eventdate', 'verbatimlatitude', 'verbatimlongitude'], inplace=True)

        # 日期篩選邏輯
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        try:
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                df = df[df['eventdate'] >= start_dt]
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                df = df[df['eventdate'] <= end_dt]
        except ValueError:
            return jsonify({"success": False, "error": "日期格式錯誤，請使用 YYYY-MM-DD"}), 400

        # ✅【核心】不再產生 HTML，而是產生一個 JSON 陣列
        locations = []
        for _, row in df.iterrows():
            locations.append({
                "lat": row['verbatimlatitude'],
                "lng": row['verbatimlongitude'],
                "popup_html": f"""
                    <b>物種:</b> {row['vernacularname']}<br>
                    <b>日期:</b> {row['eventdate'].date()}<br>
                    <b>紀錄者:</b> {row['recordedby']}
                """
            })
        
        return jsonify({"success": True, "locations": locations})

    except FileNotFoundError:
        return jsonify({"success": False, "error": "找不到地圖資料檔案"}), 404
    except Exception as e:
        print(f"地圖資料產生失敗: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": "產生熱點圖時發生錯誤"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)