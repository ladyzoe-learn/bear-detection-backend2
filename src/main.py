# src/main.py (已整合影片分析與 LINE 通知功能)

import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# 舊的 CORS(app) 設定有時在特定部署環境下不夠明確。
# 我們改用更精確的設定，明確指定允許哪個來源(前端網址)存取。

# ⚠️ 請注意：根據您錯誤日誌的截圖，您的前端來源(origin)是 'https://bear-detection-app.onrender.com'
# 這和您之前文字提供的前端網址 'https://bear-detection-frontend.onrender.com' 不一樣。
# 我們必須使用【日誌中顯示的正確網址】。
CORS(app, origins=["https://bear-detection-app.onrender.com"])
import requests
from PIL import Image
import io
import base64
import os
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# --- 新增匯入 (影片處理) ---
import cv2
import numpy as np
import tempfile
# -------------------------

# --- LINE 通知相關匯入 ---
from linebot import LineBotApi
from linebot.models import TextSendMessage
from linebot.exceptions import LineBotApiError
# -------------------------

app = Flask(__name__)
CORS(app)

# --- Hugging Face & LINE API 設定 ---
HUGGING_FACE_API_URL = "https://ladyzoe-bear-detector-api-docker.hf.space/predict"
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
# ------------------------------------

# --- LINE 通知相關函式 (與之前相同) ---
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
        print(f"Details: {e.error.details}")
    except Exception as e:
        print(f"An unexpected error occurred when sending LINE message: {e}")
# ------------------------------------


# --- 原有的圖片偵測端點 (稍作修改以方便共用) ---
def detect_objects_in_image_data(image_bytes):
    """一個共用函式，接收圖片的二進位數據並回傳偵測結果"""
    try:
        hf_files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        hf_response = requests.post(HUGGING_FACE_API_URL, files=hf_files)
        hf_response.raise_for_status()
        return hf_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Hugging Face API request failed: {e}")
        return None # 發生錯誤時回傳 None

def is_bear_detected(api_response):
    """檢查 API 回應中是否包含 'kumay'"""
    if api_response and isinstance(api_response.get('detections'), list):
        for item in api_response['detections']:
            if isinstance(item, dict) and item.get('label') == 'kumay':
                return True
    return False

@app.route('/api/detect', methods=['POST'])
def detect_bear_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳圖片檔案"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇檔案"}), 400

    try:
        image_bytes = file.read()
        
        # 呼叫共用函式進行偵測
        api_response = detect_objects_in_image_data(image_bytes)
        # 👇 【新增的修改】
        # 如果偵測到熊，就發送 LINE 通知
        if bear_is_detected:
            print("Image detection: Bear detected! Sending LINE notification...")
            alert_message = "警告：偵測到台灣黑熊出沒（圖片分析），請注意安全！"
            send_line_broadcast_message(alert_message)
        
        if not api_response:
             return jsonify({"success": False, "error": "模型偵測失敗"}), 500

        bear_is_detected = is_bear_detected(api_response)
        
        # 準備回傳給前端的資料
        response_data = {
            "success": True,
            "bear_detected": bear_is_detected,
            # 這裡我們先回傳一個固定的 confidence 和原始圖片
            # 因為詳細的框選邏輯目前在影片分析那邊
            # 您可以根據需求再將詳細的框選和信心度邏輯加回來
            "confidence": 0.99 if bear_is_detected else 0,
            "processed_image": base64.b64encode(image_bytes).decode('utf-8')
        }
        return jsonify(response_data)
        
    except Exception as e:
        print(f"圖片偵測時發生錯誤: {e}")
        return jsonify({"success": False, "error": "伺服器處理圖片時發生錯誤"}), 500

# --- ⭐️⭐️⭐️ 新增的影片分析端點 ⭐️⭐️⭐️ ---
@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳影片檔案"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇檔案"}), 400

    # 1. 將影片暫存到硬碟
    # 使用 tempfile 來安全地建立一個暫存檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(video_file.read())
        temp_video_path = temp.name

    try:
        # 2. 使用 OpenCV 打開影片
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "無法打開影片檔案"}), 500

        # 3. 獲取影片屬性與設定參數
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return jsonify({"success": False, "error": "無法讀取影片的FPS"}), 500

        # --- 偵測邏輯參數 ---
        alert_threshold_seconds = 2.0  # 連續 2 秒觸發警報
        frames_to_process_per_second = 1 # 每秒抽 1 幀進行分析 (可調整)
        
        # 計算需要跳過的幀數
        frames_to_skip = max(1, int(fps / frames_to_process_per_second))
        # 計算觸發警報需要連續偵測到的幀數
        consecutive_frames_needed = int(alert_threshold_seconds * frames_to_process_per_second)

        # --- 計數器與旗標 ---
        consecutive_bear_frames = 0
        max_consecutive_duration = 0.0
        alert_sent = False
        frame_count = 0
        
        print(f"Analyzing video: FPS={fps}, Processing {frames_to_process_per_second} frames/sec.")
        print(f"Alert threshold: {consecutive_frames_needed} consecutive frames.")

        # 4. 迴圈讀取影片幀
        while cap.isOpened():
            ret, frame = cap.read()
            # 如果影片結束，就跳出迴圈
            if not ret:
                break

            frame_count += 1
            # 5. 抽幀邏輯
            if frame_count % frames_to_skip != 0:
                continue

            print(f"Processing frame #{frame_count}...")
            
            # 6. 處理當前幀
            # 將 OpenCV 的 frame (Numpy array) 編碼成 JPG 格式的二進位數據
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                continue
            image_bytes = buffer.tobytes()

            # 7. 發送到模型進行偵測
            api_response = detect_objects_in_image_data(image_bytes)
            
            # 8. 更新連續計數器
            if is_bear_detected(api_response):
                consecutive_bear_frames += 1
                print(f"  BEAR DETECTED! Consecutive frames: {consecutive_bear_frames}")
            else:
                # 如果中斷了，計算這一次連續的總時長
                current_duration = (consecutive_bear_frames / frames_to_process_per_second)
                max_consecutive_duration = max(max_consecutive_duration, current_duration)
                # 歸零計數器
                consecutive_bear_frames = 0

            # 9. 檢查是否觸發警報
            if not alert_sent and consecutive_bear_frames >= consecutive_frames_needed:
                print("!!! ALERT TRIGGERED !!! Bear detected for over 2 seconds.")
                alert_message = f"警告：影片中偵測到台灣黑熊連續出現超過 {alert_threshold_seconds} 秒！"
                send_line_broadcast_message(alert_message)
                alert_sent = True
                # (可選) 觸發後直接跳出迴圈，節省運算資源
                break
        
        # 迴圈結束後，最後再更新一次最大連續時間
        final_duration = (consecutive_bear_frames / frames_to_process_per_second)
        max_consecutive_duration = max(max_consecutive_duration, final_duration)

        # 10. 準備回傳結果
        response_data = {
            "success": True,
            "alert_sent": alert_sent,
            "max_consecutive_duration_seconds": round(max_consecutive_duration, 2),
            "video_fps": fps,
        }
        return jsonify(response_data)

    finally:
        # 11. 清理工作
        cap.release()
        os.unlink(temp_video_path) # 刪除暫存檔案
        print("Video analysis complete. Temporary file deleted.")

# --- 原有的地圖端點 (維持不變) ---
@app.route('/api/map', methods=['GET'])
def get_bear_map():
    # ... (省略地圖程式碼，維持原樣)
    try:
        file_path = 'src/台灣黑熊.csv' 
        df = pd.read_csv(file_path)
        taiwan_map = folium.Map(location=[23.97565, 120.97388], zoom_start=7)
        marker_cluster = MarkerCluster().add_to(taiwan_map)
        for index, row in df.iterrows():
            popup_html = f"<b>物種:</b> {row['vernacularname']}<br><b>日期:</b> {row['eventdate']}<br><b>紀錄者:</b> {row['recordedby']}"
            iframe = folium.IFrame(popup_html, width=200, height=100)
            popup = folium.Popup(iframe, max_width=200)
            folium.Marker(location=[row['verbatimlatitude'], row['verbatimlongitude']], popup=popup, tooltip=f"{row['vernacularname']} - {row['eventdate']}").add_to(marker_cluster)
        map_html = taiwan_map._repr_html_()
        return jsonify({"success": True, "map_html": map_html})
    except FileNotFoundError:
        return jsonify({"success": False, "error": "找不到地圖資料檔案"}), 404
    except Exception as e:
        print(f"地圖產生失敗: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": "產生熱點圖時發生錯誤"}), 500

# 啟動伺服器
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
