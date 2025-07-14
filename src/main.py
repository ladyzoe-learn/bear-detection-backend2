# src/main.py (Final Corrected Version)

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
TELEGRAM_BOT_TOKEN= os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
# Hugging Face
HF_API_URL = os.getenv("HF_API_URL", "https://ladyzoe-bear-detector-api-docker.hf.space/predict")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- Telegram Bot 類別定義 ---
class TelegramBot:
    def __init__(self, bot_token, chat_id):
        if not bot_token or not chat_id:
            print("Warning: Telegram Bot token or chat_id is missing. Notifications will be disabled.")
            self.bot_token = None
            self.chat_id = None
        else:
            self.bot_token = bot_token
            self.chat_id = chat_id
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message):
        if not self.bot_token: return
        url = f"{self.base_url}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            print("Telegram message sent successfully.")
        except Exception as e:
            print(f"發送 Telegram 訊息錯誤: {e}")

    def send_photo(self, photo_url, caption=""):
        if not self.bot_token: return
        url = f"{self.base_url}/sendPhoto"
        data = {"chat_id": self.chat_id, "photo": photo_url, "caption": caption, "parse_mode": "HTML"}
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            print("Telegram photo sent successfully.")
        except Exception as e:
            print(f"發送 Telegram 圖片錯誤: {e}")

telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# --- 通知發送的共用函式 ---
def send_bear_alert(confidence, image_url=None, location=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_message = (
        f"🐻 <b>黑熊預警系統</b> 🚨\n\n"
        f"⚠️ <b>偵測到疑似黑熊！</b>\n"
        f"🎯 <b>信心度：{confidence:.2%}</b>\n"
        f"🕒 <b>\n請立即採取適當的安全措施！</b>\n"
    )

    if image_url:
        telegram_bot.send_photo(image_url, alert_message)
    else:
        telegram_bot.send_message(alert_message)

# --- Flask App 初始化與設定 ---
app = Flask(__name__)
# 允許來自 Render 前端和本地測試伺服器的請求
CORS(app, origins=["https://bear-detection-app.onrender.com", "http://localhost:5173"])
# 初始化快取
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

def is_bear_detected(api_response):
    if api_response and isinstance(api_response.get('detections'), list):
        bear_detections = [item for item in api_response['detections'] if isinstance(item, dict) and item.get('label') == 'kumay']
        if bear_detections:
            highest_confidence = max(item.get('confidence', 0) for item in bear_detections)
            return True, highest_confidence
    return False, 0.0

# --- API 端點 ---

# 圖片偵測 API
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

        bear_is_detected, confidence = is_bear_detected(api_response)
        alert_sent = False
        if bear_is_detected and confidence >= 0.7:
            print("Image detection: Bear detected! Sending Telegram alert...")
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

# ✅【修正一】影片分析 API，改回即時觸發警報的邏輯
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

    cap = None
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "無法讀取影片檔案"}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("⚠️ FPS 無法讀取，使用預設 30 FPS")
            fps = 30

        alert_threshold_seconds = 2.0  # 恢復為 2 秒
        frames_to_process_per_second = 0.5 # 保持較低的抽幀率以優化性能
        frames_to_skip = max(1, int(fps / frames_to_process_per_second))
        consecutive_frames_needed = int(alert_threshold_seconds * frames_to_process_per_second)

        frame_count = 0
        consecutive_bear_frames = 0
        max_consecutive_duration = 0.0
        highest_confidence_in_video = 0.0
        alert_sent = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % frames_to_skip != 0: continue
            
            _, encoded = cv2.imencode(".jpg", frame)
            api_response = detect_objects_in_image_data(encoded.tobytes())
            detected, confidence = is_bear_detected(api_response)

            if detected:
                consecutive_bear_frames += 1
                highest_confidence_in_video = max(highest_confidence_in_video, confidence)
            else:
                current_duration = (consecutive_bear_frames / frames_to_process_per_second)
                max_consecutive_duration = max(max_consecutive_duration, current_duration)
                consecutive_bear_frames = 0

            if not alert_sent and consecutive_bear_frames >= consecutive_frames_needed:
                send_bear_alert(
                    confidence=highest_confidence_in_video,
                    image_url=None,
                    location="影片偵測區域"
                )
                alert_sent = True
                print("🚨 即時觸發警報並停止分析")
                break
        
        final_duration = (consecutive_bear_frames / frames_to_process_per_second)
        max_consecutive_duration = max(max_consecutive_duration, final_duration)

        return jsonify({
            "success": True,
            "alert_sent": alert_sent,
            "max_consecutive_duration_seconds": round(max_consecutive_duration, 2),
            "video_fps": fps
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"伺服器處理影片時發生錯誤: {e}"}), 500
    finally:
        if cap:
            cap.release()
        os.remove(temp_video_path)

# ✅【修正二】地圖 API，改為回傳純資料
@app.route('/api/map', methods=['GET'])
# @cache.cached(query_string=True) # 先註解掉快取，方便除錯
def get_bear_map():
    try:
        file_path = 'src/台灣黑熊.csv'
        df = pd.read_csv(file_path)
        df['eventdate'] = pd.to_datetime(df['eventdate'], errors='coerce')
        df.dropna(subset=['eventdate', 'verbatimlatitude', 'verbatimlongitude'], inplace=True)

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
        print(f"地圖產生失敗: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": "產生熱點圖時發生錯誤"}), 500

# 啟動伺服器
if __name__ == '__main__':
    # 修正 debug=True 造成的重複執行問題
    # 在 Render 上，debug 模式應為 False
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug_mode)