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
from linebot import LineBotApi
from linebot.models import TextSendMessage
from linebot.exceptions import LineBotApiError
from datetime import datetime
from flask_caching import Cache

# --- Flask App 初始化與設定 ---
app = Flask(__name__)
CORS(app, origins=["https://bear-detection-app.onrender.com"])

# --- 快取設定 ---
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 3600})
cache.init_app(app)

# --- 環境變數讀取 ---
HF_API_URL = os.getenv("HF_API_URL", "https://ladyzoe-bear-detector-api-docker.hf.space/predict")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# --- LINE 通知相關函式 ---
line_bot_api = None
if LINE_CHANNEL_ACCESS_TOKEN:
    try:
        line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        print("LINE Bot API initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize LINE Bot API: {e}")
else:
    print("LINE_CHANNEL_ACCESS_TOKEN not found in environment. LINE notification feature is disabled.")

def send_line_broadcast_message(message_text):
    if not line_bot_api:
        print("LINE Bot API not initialized. Cannot send message.")
        return
    try:
        line_bot_api.broadcast(TextSendMessage(text=message_text))
        print("LINE broadcast message sent successfully.")
    except LineBotApiError as e:
        print(f"Error sending LINE broadcast message: {e.status_code} {e.error.message}")
    except Exception as e:
        print(f"An unexpected error occurred when sending LINE message: {e}")

# --- 偵測相關的共用函式 ---
def detect_objects_in_image_data(image_bytes):
    if not HF_API_TOKEN:
        print("Error: Hugging Face API Token (HF_API_TOKEN) is not set.")
        return None
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        hf_files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        hf_response = requests.post(HF_API_URL, headers=headers, files=hf_files)
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
        
        if bear_is_detected:
            print("Image detection: Bear detected! Sending LINE notification...")
            alert_message = "熊蹤跡預警，照片偵測到 台灣黑熊並即將進入生活共同圈，請保持安全距離並提高警覺！"
            send_line_broadcast_message(alert_message)
        
        response_data = {
            "success": True,
            "bear_detected": bear_is_detected,
            "confidence": 0.99 if bear_is_detected else 0,
            "processed_image": base64.b64encode(image_bytes).decode('utf-8')
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
                alert_message = f"熊蹤跡預警，影片偵測到 台灣黑熊並即將進入生活共同圈，請保持安全距離並提高警覺！"
                send_line_broadcast_message(alert_message)
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
@cache.cached(query_string=True)
def get_bear_map():
    print("Generating new map (not from cache)...")
    try:

        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, '台灣黑熊.csv')
        df = pd.read_csv(file_path)
        df['eventdate'] = pd.to_datetime(df['eventdate'], errors='coerce')
        df = df.dropna(subset=['eventdate'])

        # ✅【建議修改處】使用更靈活的日期篩選邏輯
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
        # ✅【建議修改結束】

        taiwan_map = folium.Map(location=[23.97565, 120.97388], zoom_start=7)
        marker_cluster = MarkerCluster().add_to(taiwan_map)

        for _, row in df.iterrows():
            popup_html = f"""
                <b>事件ID:</b> {row['occurrenceid']}<br>
                <b>日期:</b> {row['eventdate'].date()}<br>
    
            """
            iframe = folium.IFrame(popup_html, width=200, height=100)
            popup = folium.Popup(iframe, max_width=200)
            folium.Marker(
                location=[row['verbatimlatitude'], row['verbatimlongitude']],
                popup=popup,
                tooltip=f"{row['occurrenceid']} - {row['eventdate'].date()}"
            ).add_to(marker_cluster)

        map_html = taiwan_map._repr_html_()
        return jsonify({"success": True, "map_html": map_html})

    except FileNotFoundError:
        return jsonify({"success": False, "error": "找不到地圖資料檔案"}), 404
    except Exception as e:
        print(f"地圖產生失敗: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": "產生熱點圖時發生錯誤"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)