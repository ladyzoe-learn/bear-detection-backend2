from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import io
import base64
import os
import tempfile
import ffmpeg
import traceback
import logging
import shutil
import collections # 【新增】引入 collections 模組，用於 deque

app = Flask(__name__)
CORS(app)

# 設定日誌記錄
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

HUGGING_FACE_API_URL = "https://ladyzoe-bear-detector-api-docker.hf.space/predict"

# 設定一個明確的臨時目錄路徑，避免 tempfile.TemporaryDirectory 可能遇到的權限問題
# 這裡假設 'temp_frames_storage' 子目錄與你的應用程式腳本在同一層
# 確保這個目錄在應用程式啟動前或創建臨時目錄前存在
CUSTOM_TEMP_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_frames_storage')
os.makedirs(CUSTOM_TEMP_STORAGE_DIR, exist_ok=True) # 確保目錄存在，如果不存在則創建

# detect_bear_from_image_bytes 函式保持不變
def detect_bear_from_image_bytes(image_bytes, filename, mimetype):
    """
    取得 image bytes → 呼叫 Hugging Face API → 回傳結果
    """
    hf_files = {'file': (filename, image_bytes, mimetype)}
    hf_response = requests.post(HUGGING_FACE_API_URL, files=hf_files)
    hf_response.raise_for_status()
    api_response = hf_response.json()

    bear_detected = False
    highest_confidence = 0.0

    detection_list = api_response.get('detections', [])
    if isinstance(detection_list, list):
        for item in detection_list:
            if isinstance(item, dict) and item.get('label') == 'kumay':
                bear_detected = True
                score = item.get('confidence', 0)
                if score > highest_confidence:
                    highest_confidence = score

    return bear_detected, highest_confidence

# /api/detect 路由保持不變
@app.route('/api/detect', methods=['POST'])
def detect_bear():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳圖片檔案"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇檔案"}), 400

    try:
        image_bytes = file.read()
        bear_detected, highest_confidence = detect_bear_from_image_bytes(
            image_bytes, file.filename, file.mimetype
        )

        image = Image.open(io.BytesIO(image_bytes))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG") # 假設 processed_image 依然是 JPEG
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            "success": True,
            "bear_detected": bear_detected,
            "confidence": highest_confidence,
            "processed_image": processed_image_base64
        })

    except Exception as e:
        app.logger.error(f"[DETECT IMAGE] Unexpected error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "伺服器發生未預期的錯誤"}), 500


---
## 調整後的影片處理路由 (`/api/detect-video`)
---
@app.route('/api/detect-video', methods=['POST'])
def detect_bear_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳影片檔案"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇影片"}), 400

    video_path = None # 初始化為 None，確保 finally 區塊可以安全地檢查
    frames_dir = None # 初始化為 None，用於手動創建的臨時目錄

    # 【新增】用於追蹤最近5個影格的熊檢測狀態 (True/False)
    # collections.deque 是一個高效的雙端佇列，maxlen=5 會自動保持最多5個元素
    bear_detection_history = collections.deque(maxlen=5)
    
    # 【修改】將 results 變數更名為 final_results，以明確其最終輸出目的
    final_results = [] 

    # 【新增】設定黑熊預警的信心度閾值
    BEAR_WARNING_CONFIDENCE_THRESHOLD = 0.50 # 50%

    try:
        # 將上傳的影片保存到臨時檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(file.read())
            video_path = tmp.name # 獲取臨時檔案的路徑
        
        # 從 batch_extract_frames 函式繼承的參數設定
        # 這些值是固定的，你可以根據需求調整
        fixed_fps = 1       # 每秒擷取影格數
        fixed_width = 640   # 縮放後的寬度
        fixed_height = 360  # 縮放後的高度
        output_image_format = "png" # 輸出圖片格式，按照 batch_extract_frames 預設為 PNG
        
        # 使用 tempfile.mkdtemp() 手動創建臨時目錄，並指定路徑
        frames_dir = tempfile.mkdtemp(dir=CUSTOM_TEMP_STORAGE_DIR)
        
        # output_%05d.png 是 batch_extract_frames 中的命名模式
        output_pattern = os.path.join(frames_dir, f'output_%05d.{output_image_format}')

        app.logger.info(f"從影片 '{file.filename}' 提取影格到 '{frames_dir}'...")
        try:
            # FFmpeg 命令：從原始影片提取影格，縮放，並指定輸出幀率
            (
                ffmpeg
                .input(video_path) # 直接輸入原始上傳的影片
                .filter('fps', fps=fixed_fps) # 使用固定的幀率
                .filter('scale', fixed_width, fixed_height) # 使用固定的寬高
                .output(output_pattern, vsync='vfr', **{'start_number': 1}) # start_number 可以設定為1，類似 batch_extract_frames 的起始編號概念
                .run(overwrite_output=True, capture_stderr=True)
            )
            app.logger.info("影格提取完成。")

        except ffmpeg.Error as e:
            error_details = e.stderr.decode('utf-8')
            app.logger.error(f"[FFMPEG FRAME EXTRACTION FAILED] stderr: {error_details}")
            return jsonify({
                "success": False,
                "error": "影片處理失敗，無法從影片中提取影格。"
            }), 500
        
        # 遍歷臨時目錄中的所有提取出的影格，並進行熊的檢測
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(f'.{output_image_format}')]) # 只選擇正確格式的圖片
        if not frame_files:
            app.logger.warning(f"未能從影片 '{file.filename}' 提取任何影格。")
            return jsonify({
                "success": False,
                "error": "未能從影片中提取任何影格，請檢查影片內容或參數設定。"
            }), 500

        app.logger.info(f"開始對 {len(frame_files)} 個影格進行熊檢測...")
        for frame_index_zero_based, frame_file in enumerate(frame_files): # 0-based index
            logical_frame_index = frame_index_zero_based + 1 
            frame_path = os.path.join(frames_dir, frame_file)
            
            # 【新增】為當前影格初始化結果字典，包含預警標誌
            current_frame_result = {
                "frame_index": logical_frame_index,
                "bear_detected": False, 
                "confidence": 0.0,      
                "bear_warning_triggered": False # 預設為 False
            }

            try:
                with open(frame_path, 'rb') as f:
                    image_bytes = f.read()
                
                bear_detected, confidence = detect_bear_from_image_bytes(
                    image_bytes, 
                    f"original_video_frame_{logical_frame_index}.{output_image_format}", # 檔案名稱
                    f"image/{output_image_format}" # MIME 類型是 image/png
                )
                
                # 更新當前影格的檢測結果
                current_frame_result["bear_detected"] = bear_detected
                current_frame_result["confidence"] = confidence

                # 【修改】判斷是否為「有效檢測」：偵測到熊且信心度達到閾值
                is_valid_bear_detection = bear_detected and (confidence >= BEAR_WARNING_CONFIDENCE_THRESHOLD)
                
                # 【修改】更新熊檢測歷史記錄，只將有效檢測納入計算
                bear_detection_history.append(is_valid_bear_detection)

                # 【新增】檢查是否觸發黑熊預警 (5張圖片中有2張被判定為黑熊)
                # sum(1 for detected in bear_detection_history if detected) 計算 deque 中 True 的數量
                if sum(1 for detected in bear_detection_history if detected) >= 2:
                    current_frame_result["bear_warning_triggered"] = True
                    app.logger.warning(
                        f"在影格 {logical_frame_index} 處觸發黑熊預警！"
                        f"最近 {len(bear_detection_history)} 影格中有 {sum(1 for detected in bear_detection_history if detected)} 個有效偵測到黑熊。"
                    )
                
                # 將當前影格的完整結果添加到最終結果列表
                final_results.append(current_frame_result)

            except Exception as e:
                app.logger.warning(f"處理影格 '{frame_file}' 時發生錯誤: {e}")
                # 即使處理失敗，也將包含預設值或部分結果的字典添加到 final_results
                final_results.append(current_frame_result) 
            
        app.logger.info("所有影格檢測完成。")
            
        return jsonify({
            "success": True,
            "total_frames_processed": len(final_results), # 【修改】回傳 final_results 的長度
            "results": final_results # 【修改】回傳 final_results
        })

    except Exception as e:
        app.logger.error(f"[DETECT VIDEO] 處理影片時發生未預期的錯誤: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "影片分析時發生未預期的伺服器錯誤。"}), 500
    
    finally:
        # 手動清理創建的臨時目錄及其內容
        if frames_dir and os.path.exists(frames_dir):
            try:
                shutil.rmtree(frames_dir)
                app.logger.info(f"已刪除臨時影格目錄: {frames_dir}")
            except OSError as e:
                app.logger.error(f"無法刪除臨時影格目錄 {frames_dir}: {e}")

        # 確保臨時影片檔案在處理完畢後被刪除
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)
            app.logger.info(f"已刪除臨時影片檔案: {video_path}")


# 啟動
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
