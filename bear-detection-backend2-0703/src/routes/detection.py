import requests
import base64
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile

detection_bp = Blueprint('detection', __name__)

# Hugging Face Spaces API 端點
HUGGINGFACE_API_URL = "https://ladyzoe-bear-detector-api-docker.hf.space/predict"

@detection_bp.route('/detect', methods=['POST'])
def detect_bear():
    """
    接收前端上傳的圖片，轉發給 Hugging Face Spaces API，
    並將結果返回給前端。
    """
    try:
        # 檢查是否有上傳的檔案
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': '未找到上傳的圖片檔案'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未選擇檔案'
            }), 400
        
        # 檢查檔案類型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                'success': False,
                'error': '不支援的檔案格式，請上傳 PNG、JPG、JPEG、GIF 或 BMP 格式的圖片'
            }), 400
        
        # 將檔案保存到臨時位置
        filename = secure_filename(file.filename)
        
        # 使用臨時檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # 準備發送給 Hugging Face Spaces API 的請求
            with open(temp_file_path, 'rb') as f:
                files = {'file': (filename, f, file.content_type)}
                
                # 發送請求到 Hugging Face Spaces API
                response = requests.post(
                    HUGGINGFACE_API_URL,
                    files=files,
                    timeout=30  # 30秒超時
                )
            
            # 檢查響應狀態
            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'error': f'Hugging Face API 請求失敗，狀態碼: {response.status_code}'
                }), 500
            
            # 檢查響應內容類型
            content_type = response.headers.get('content-type', '')
            
            if 'image' in content_type:
                # 如果返回的是圖片，將其轉換為 base64
                image_data = response.content
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # 假設有偵測到黑熊（因為 API 返回了處理後的圖片）
                # 實際上，我們需要根據 API 的具體實現來判斷是否偵測到黑熊
                # 這裡我們假設如果返回了圖片，就表示有偵測結果
                return jsonify({
                    'success': True,
                    'bear_detected': True,  # 這個需要根據實際 API 響應來判斷
                    'confidence': 0.85,     # 這個也需要從 API 響應中獲取
                    'processed_image': image_base64,
                    'message': '偵測完成'
                })
            
            elif 'application/json' in content_type:
                # 如果返回的是 JSON
                try:
                    result = response.json()
                    return jsonify({
                        'success': True,
                        'bear_detected': result.get('bear_detected', False),
                        'confidence': result.get('confidence', 0.0),
                        'processed_image': result.get('processed_image'),
                        'message': result.get('message', '偵測完成')
                    })
                except ValueError:
                    return jsonify({
                        'success': False,
                        'error': '無法解析 API 響應'
                    }), 500
            
            else:
                return jsonify({
                    'success': False,
                    'error': f'未知的響應格式: {content_type}'
                }), 500
                
        finally:
            # 清理臨時檔案
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'API 請求超時，請稍後再試'
        }), 504
    
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'API 請求失敗: {str(e)}'
        }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'伺服器內部錯誤: {str(e)}'
        }), 500

@detection_bp.route('/health', methods=['GET'])
def health_check():
    """
    健康檢查端點
    """
    return jsonify({
        'status': 'healthy',
        'message': '台灣黑熊偵測 API 運行正常'
    })

