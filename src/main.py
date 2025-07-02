# main.py (已移除畫圖功能的最終版本)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
# 移除了 ImageDraw 和 ImageFont
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
        detections = hf_response.json()

        bear_detected = False
        highest_confidence = 0.0

        # 2. 遍歷偵測結果，只判斷不畫圖
        if detections:
            for item in detections:
                if item.get('label') == 'kumay':
                    bear_detected = True
                    score = item.get('score', 0)
                    
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

    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "error": f"呼叫模型失敗: {e}"}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"伺服器內部錯誤: {str(e)}"}), 500

# 啟動伺服器
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

