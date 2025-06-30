from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
# 允許所有來源的跨域請求，這對於前後端分離部署很重要
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

