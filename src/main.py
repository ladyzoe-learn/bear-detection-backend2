<<<<<<< HEAD
# main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import base64
=======
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
>>>>>>> 416cace7865a45d01cea8fa9957b7dc2dfc5c4e4

app = Flask(__name__)
# 允許所有來源的跨域請求，這對於前後端分離部署很重要
CORS(app) 

<<<<<<< HEAD
# Hugging Face 模型的 API URL
HUGGING_FACE_API_URL = "https://ladyzoe-bear-detector-api-docker.hf.space/detect/"

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
                if item.get('label') == 'bear':
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
=======
@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        "status": "healthy",
        "message": "台灣黑熊偵測 API 運行正常"
    })

@app.route('/api/detect', methods=['POST'])
def detect():
    """接收前端圖片，轉發至 Hugging Face API，並回傳最終判斷結果"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '沒有提供圖片檔案'}), 400

    image_file = request.files['image']
    
    # 從環境變數讀取 Hugging Face API 的 URL
    hf_url = os.environ.get('HUGGINGFACE_API_URL')
    if not hf_url:
        return jsonify({'success': False, 'error': '伺服器未配置 Hugging Face API URL'}), 500

    try:
        # 準備要轉發的檔案
        # 注意：Hugging Face API 的欄位名稱是 'file'
        files = {'file': (image_file.filename, image_file.stream, image_file.mimetype)}
        
        # 呼叫 Hugging Face API
        response_from_hf = requests.post(hf_url, files=files)
        response_from_hf.raise_for_status() # 如果狀態碼不是 2xx，會拋出錯誤

        # 解析從 Hugging Face 收到的 JSON 回應
        hf_data = response_from_hf.json()

        # --- 核心判斷邏輯 (這就是我們要修改的地方！) ---
        bear_detected = False
        confidence = 0.0
        # 安全地取得 detections 列表，如果不存在則預設為空列表
        detections = hf_data.get('detections', [])

        # 遍歷所有偵測到的物件
        for detection in detections:
            # 檢查物件的標籤是否為 'Taiwan-Black-Bear'
            # 這裡的標籤名稱必須與您模型訓練時的類別名稱完全一致！
            if detection.get('label') == 'Taiwan-Black-Bear':
                bear_detected = True
                # 如果找到黑熊，就記錄它的信心度並跳出迴圈
                confidence = detection.get('confidence', 0.0)
                break # 只要找到一隻熊就夠了

        # 從 Hugging Face 回應中取得處理後的 Base64 圖片
        processed_image_base64 = hf_data.get('processed_image')

        # 準備回傳給前端的最終結果
        return jsonify({
            'success': True,
            'bear_detected': bear_detected,
            'confidence': confidence,
            'processed_image': processed_image_base64,
            'message': '偵測完成'
        })

    except requests.exceptions.RequestException as e:
        # 處理網路請求錯誤
        return jsonify({'success': False, 'error': f'請求 Hugging Face API 失敗: {e}'}), 502
    except Exception as e:
        # 處理其他所有未預期的錯誤
        return jsonify({'success': False, 'error': f'伺服器內部錯誤: {e}'}), 500

if __name__ == '__main__':
    # 為了 Render 部署，從環境變數讀取 PORT，並將 host 設為 '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

>>>>>>> 416cace7865a45d01cea8fa9957b7dc2dfc5c4e4

# 這段是為了讓你在本機測試用，Render 會忽略它
if __name__ == '__main__':
    app.run(debug=True, port=5001)

