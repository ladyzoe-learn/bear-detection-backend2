# main.py (最終修正版)
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import io
import base64
import os
import pandas as pd
import folium
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster

app = Flask(__name__)
CORS(app)

# Hugging Face 模型的 API URL
HUGGING_FACE_API_URL = "https://ladyzoe-bear-detector-api-docker.hf.space/predict"

@app.route('/api/detect', methods=['POST'])
def detect_bear():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳圖片檔案"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇檔案"}), 400

    try:
        image_bytes = file.read()

        # 1. 呼叫 Hugging Face 模型
        hf_files = {'file': (file.filename, image_bytes, file.mimetype)}
        hf_response = requests.post(HUGGING_FACE_API_URL, files=hf_files)
        hf_response.raise_for_status()
        api_response = hf_response.json() # 將變數改名以避免混淆

        bear_detected = False
        highest_confidence = 0.0

        # 2. ✅【關鍵修改處】直接遍歷 api_response 字典中 'detections' 鍵對應的列表
        detection_list = api_response.get('detections', []) # 使用 .get() 更安全
        if isinstance(detection_list, list):
            for item in detection_list:
                if isinstance(item, dict) and item.get('label') == 'kumay':
                    bear_detected = True
                    score = item.get('confidence', 0) # API 回傳的是 confidence
                    
                    if score > highest_confidence:
                        highest_confidence = score
        
        # 3. 將「原始」圖片轉換成 Base64
        image = Image.open(io.BytesIO(image_bytes))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 4. 建立 JSON 回應
        response_data = {
            "success": True,
            "bear_detected": bear_detected,
            "confidence": highest_confidence,
            "processed_image": processed_image_base64
        }
        
        return jsonify(response_data)

    except Exception as e:
        
        print("----------- UNEXPECTED ERROR -----------")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        
        import traceback
        traceback.print_exc()
        
        print("----------------------------------------")
        
        return jsonify({"success": False, "error": "伺服器發生未預期的錯誤，請查看後端日誌"}), 500
    
@app.route('/api/map', methods=['GET'])
def get_bear_map():
    try:
        # 建立地圖物件，中心點設在台灣
        # 注意：路徑是相對於你執行 main.py 的位置
        file_path = 'src/台灣黑熊.csv' 
        df = pd.read_csv(file_path)

        # 建立地圖，中心點設在台灣的中心
        taiwan_map = folium.Map(location=[23.97565, 120.97388], zoom_start=7)

        marker_cluster = MarkerCluster().add_to(taiwan_map) # 將集群添加到地圖

        # 遍歷 DataFrame 中的每一行數據，添加標記到集群中
        for index, row in df.iterrows():
            # 組合彈出視窗中要顯示的 HTML 內容
            # 確保你的 CSV 有 'vernacularname', 'eventdate', 'recordedby' 這些欄位
            # 如果沒有，請根據你的CSV實際欄位調整，或移除這些資訊
            popup_html = f"""
            <b>物種:</b> {row['vernacularname']}<br>
            <b>日期:</b> {row['eventdate']}<br>
            <b>紀錄者:</b> {row['recordedby']}
            """

            iframe = folium.IFrame(popup_html, width=200, height=100)
            popup = folium.Popup(iframe, max_width=200)

            # 在地圖上添加標記 (使用正確的英文欄位名稱)
            folium.Marker(
                location=[row['verbatimlatitude'], row['verbatimlongitude']],
                popup=popup,
                tooltip=f"{row['vernacularname']} - {row['eventdate']}"
            ).add_to(marker_cluster) # 将標記添加到 marker_cluster，而不是直接添加到地圖

        # 將地圖物件轉換成 HTML 字串
        map_html = taiwan_map._repr_html_()

        return jsonify({"success": True, "map_html": map_html})

    except FileNotFoundError:
        return jsonify({"success": False, "error": "找不到地圖資料檔案"}), 404
    except Exception as e:
        # 在日誌中印出詳細錯誤，方便除錯
        print(f"地圖產生失敗: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": "產生熱點圖時發生錯誤"}), 500

# 啟動伺服器
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

