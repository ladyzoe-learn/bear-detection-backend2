from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app) 

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
    
    hf_url = os.environ.get('HUGGINGFACE_API_URL')
    if not hf_url:
        return jsonify({'success': False, 'error': '伺服器未配置 Hugging Face API URL'}), 500

    try:
        files = {'file': (image_file.filename, image_file.stream, image_file.mimetype)}
        
        response_from_hf = requests.post(hf_url, files=files)
        response_from_hf.raise_for_status()

        hf_data = response_from_hf.json()

        bear_detected = False
        confidence = 0.0
        # hf_data['detections'] 返回的是一個列表，例如 [{'label': 'kumay', ...}]
        detections = hf_data.get('detections', [])

        for detection in detections:
            # --- 核心修改：將比對的標籤從 'Taiwan-Black-Bear' 改為 'kumay' ---
            if detection.get('label') == 'kumay':
                bear_detected = True
                confidence = detection.get('confidence', 0.0)
                break 

        processed_image_base64 = hf_data.get('processed_image')

        return jsonify({
            'success': True,
            'bear_detected': bear_detected,
            'confidence': confidence,
            'processed_image': processed_image_base64,
            'message': '偵測完成'
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'success': False, 'error': f'請求 Hugging Face API 失敗: {e}'}), 502
    except Exception as e:
        return jsonify({'success': False, 'error': f'伺服器內部錯誤: {e}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

