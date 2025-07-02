# main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
 
app = Flask(__name__)
CORS(app)  # 允許所有來源的跨域請求

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
        # 讀取圖片檔案
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. 呼叫 Hugging Face 模型進行偵測
        hf_files = {'file': (file.filename, image_bytes, file.mimetype)}
        hf_response = requests.post(HUGGING_FACE_API_URL, files=hf_files)
        hf_response.raise_for_status()
        detections = hf_response.json()

        bear_detected = False
        highest_confidence = 0.0

        # 2. 在圖片上繪製邊界框
        draw = ImageDraw.Draw(image)
        try:
            # 嘗試載入稍大一點的字型
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            # 如果找不到字型，則使用預設字型
            font = ImageFont.load_default()

        if detections:
            for item in detections:
                if item.get('label') == 'kumay':
                    bear_detected = True
                    box = item.get('box')
                    score = item.get('score', 0)
                    
                    if score > highest_confidence:
                        highest_confidence = score

                    # 繪製矩形框
                    draw.rectangle(box, outline="lime", width=5)
                    
                    # 準備標籤文字
                    label = f"{item['label']}: {score:.2f}"
                    
                    # 計算文字位置
                    text_position = [box[0], box[1] - 25]
                    if text_position[1] < 0:
                        text_position[1] = box[1] + 5
                        
                    # 繪製文字背景和文字
                    text_bbox = draw.textbbox(text_position, label, font=font)
                    draw.rectangle(text_bbox, fill="lime")
                    draw.text(text_position, label, fill="black", font=font)

        # 3. 將處理後的圖片轉換成 Base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 4. 建立符合前端需求的 JSON 回應
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

# 這段是為了讓你在本機測試用，Render 會忽略它
# main.py 的最下方

if __name__ == '__main__':
    # Render 平台會透過環境變數 'PORT' 告訴我們應該使用哪個通訊埠
    # 我們從環境變數讀取它，如果沒有就預設為 10000
    port = int(os.environ.get('PORT', 10000))
    
    # 為了讓外部網路可以連線，host 必須設定為 '0.0.0.0'
    # debug=False 是生產環境的安全作法
    app.run(host='0.0.0.0', port=port, debug=False)

