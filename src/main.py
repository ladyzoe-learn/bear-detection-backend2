# main.py (最終修正版)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import io
import base64
import os

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

# 啟動伺服器
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

