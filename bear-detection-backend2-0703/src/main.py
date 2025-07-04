from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import io
import base64
import os
import cv2
import tempfile
import ffmpeg
import traceback
import logging # 【修改點】引入 logging 模組

app = Flask(__name__)
CORS(app)

# 【修改點】設定日誌記錄
# 在正式環境，建議設定日誌檔案路徑、等級和格式
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


HUGGING_FACE_API_URL = "https://ladyzoe-bear-detector-api-docker.hf.space/predict"

# ... detect_bear_from_image_bytes 和 /api/detect 路由保持不變 ...
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
        image.save(buffered, format="JPEG")
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


# ==============================================================================
# ===================== 以下為適合正式環境的影片處理路由 =====================
# ==============================================================================
@app.route('/api/detect-video', methods=['POST'])
def detect_bear_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳影片檔案"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇影片"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(file.read())
        video_path = tmp.name

    try:
        max_segment_length = 10
        min_segment_length = 5
        min_fps = 1
        max_fps = 3
        output_image_format = "jpeg"
        results = []
        segment_index = 1

        try:
            probe = ffmpeg.probe(video_path)
            video_duration = float(probe['format']['duration'])
        except ffmpeg.Error as e:
            error_details = e.stderr.decode('utf-8')
            # 【修改點】將詳細錯誤記錄到日誌，而不是回傳給使用者
            app.logger.error(f"[FFMPEG PROBE FAILED] stderr: {error_details}")
            return jsonify({
                "success": False,
                "error": "無法讀取影片資訊，檔案可能已損毀或格式不支援。"
            }), 500

        current_time = 0
        while current_time < video_duration:
            end_time = min(current_time + max_segment_length, video_duration)
            
            with tempfile.NamedTemporaryFile(delete=True, suffix='_segment.mp4') as segment_tmp:
                segment_path = segment_tmp.name
                
                try:
                    (
                        ffmpeg
                        .input(video_path, ss=current_time, to=end_time)
                        .output(segment_path, c='copy')
                        .run(overwrite_output=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    error_details = e.stderr.decode('utf-8')
                    # 【修改點】記錄詳細錯誤
                    app.logger.error(f"[FFMPEG SEGMENT FAILED] Segment {segment_index}, stderr: {error_details}")
                    return jsonify({
                        "success": False,
                        "error": "影片處理失敗，無法對影片進行分段。"
                    }), 500

                probe_segment = ffmpeg.probe(segment_path)
                segment_duration = float(probe_segment['format']['duration'])
                target_fps = min_fps + (max_fps - min_fps) * (segment_duration - min_segment_length) / (max_segment_length - min_segment_length) if (max_segment_length - min_segment_length) > 0 else min_fps
                target_fps = max(min_fps, min(max_fps, target_fps))

                with tempfile.TemporaryDirectory() as frames_dir:
                    output_pattern = os.path.join(frames_dir, f'frame_%04d.{output_image_format}')
                    try:
                        (
                            ffmpeg
                            .input(segment_path)
                            .output(output_pattern, r=target_fps, vsync='vfr')
                            .run(overwrite_output=True, capture_stderr=True)
                        )
                    except ffmpeg.Error as e:
                        error_details = e.stderr.decode('utf-8')
                        # 【修改點】記錄詳細錯誤
                        app.logger.error(f"[FFMPEG FRAME EXTRACTION FAILED] Segment {segment_index}, stderr: {error_details}")
                        return jsonify({
                            "success": False,
                            "error": "影片處理失敗，無法從影片中提取影格。"
                        }), 500
                    
                    frame_files = sorted(os.listdir(frames_dir))
                    for frame_index, frame_file in enumerate(frame_files, 1):
                        frame_path = os.path.join(frames_dir, frame_file)
                        try:
                            with open(frame_path, 'rb') as f:
                                image_bytes = f.read()
                            
                            bear_detected, confidence = detect_bear_from_image_bytes(
                                image_bytes, f"segment{segment_index}_frame{frame_index}.{output_image_format}", f"image/{output_image_format}"
                            )
                            results.append({
                                "segment": segment_index,
                                "frame": frame_index,
                                "bear_detected": bear_detected,
                                "confidence": confidence
                            })
                        except Exception as e:
                            app.logger.warning(f"Could not process frame {frame_file} in segment {segment_index}: {e}")

            current_time = end_time
            segment_index += 1

        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as e:
        # 【修改點】記錄未預期的錯誤
        app.logger.error(f"[DETECT VIDEO] Unexpected error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "影片分析時發生未預期的伺服器錯誤。"}), 500
    
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


# 啟動
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    # 在正式環境，建議使用 Gunicorn 或 uWSGI 等 WSGI 伺服器，而不是 app.run()
    # 例如: gunicorn --bind 0.0.0.0:10000 your_app_file_name:app
    app.run(host='0.0.0.0', port=port, debug=False)
