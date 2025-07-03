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

app = Flask(__name__)
CORS(app) # 啟用 CORS 擴展，允許跨域請求

HUGGING_FACE_API_URL = "https://ladyzoe-bear-detector-api-docker.hf.space/predict"

# 共用函式，幫助影片 route 與原本圖片 route 都能用
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

        # 回傳圖片
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
        print("----------- UNEXPECTED ERROR -----------")
        print(e)
        return jsonify({"success": False, "error": "伺服器發生未預期的錯誤"}), 500

@app.route('/api/detect-video', methods=['POST'])
def detect_bear_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "沒有上傳影片檔案"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "沒有選擇影片"}), 400

    try:
        # 先將影片寫入暫存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp: # 這裡的 suffix 只是初始設定，FFmpeg 可以處理多種格式
            tmp.write(file.read())
            video_path = tmp.name

        segment_length = 5  # 秒
        min_segment_length = 5
        max_segment_length = 10
        min_fps = 1
        max_fps = 3
        output_image_format = "jpg"  # 可以在這裡更改輸出的圖片格式 (jpg, jpeg, png)

        results = []
        segment_index = 1

        probe = ffmpeg.probe(video_path)
        video_duration = float(probe['format']['duration'])

        current_time = 0
        while current_time < video_duration:
            end_time = min(current_time + max_segment_length, video_duration)

            with tempfile.NamedTemporaryFile(delete=False, suffix=f'_segment_{segment_index}.mp4') as segment_tmp: # 這裡的 suffix 只是暫時的
                segment_path = segment_tmp.name
                try:
                    (
                        ffmpeg
                        .input(video_path, ss=current_time, to=end_time)
                        .output(segment_path, c='copy') # 複製編碼，加快速度
                        .run(overwrite_output=True, quiet=True)
                    )
                except ffmpeg.Error as e:
                    print(f"Error creating segment {segment_index}: {e.stderr.decode('utf8')}")
                    os.unlink(video_path)
                    os.unlink(segment_path)
                    return jsonify({"success": False, "error": f"影片分段失敗: {e}"}), 500

            probe_segment = ffmpeg.probe(segment_path)
            segment_duration = float(probe_segment['format']['duration'])
            target_fps = min_fps + (max_fps - min_fps) * (segment_duration - min_segment_length) / (max_segment_length - min_segment_length) if (max_segment_length - min_segment_length) > 0 else min_fps
            target_fps = max(min_fps, min(max_fps, target_fps))

            frame_index = 1
            with tempfile.TemporaryDirectory() as frames_dir:
                output_pattern = os.path.join(frames_dir, f'frame_%04d.{output_image_format}') # 輸出圖片的格式在這裡控制
                try:
                    (
                        ffmpeg
                        .input(segment_path)
                        .output(output_pattern, r=target_fps, vsync='vfr') # r 設定幀率, vsync vfr 依照實際幀生成
                        .run(overwrite_output=True, quiet=True)
                    )
                except ffmpeg.Error as e:
                    print(f"Error extracting frames from segment {segment_index}: {e.stderr.decode('utf8')}")
                    os.unlink(video_path)
                    os.unlink(segment_path)
                    return jsonify({"success": False, "error": f"幀提取失敗: {e}"}), 500
                    
                frame_files = sorted(os.listdir(frames_dir))
                for frame_file in frame_files:
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
                        frame_index += 1
                    except Exception as e:
                        print(f"Error processing frame {frame_file}: {e}")

            os.unlink(segment_path) # 清除片段檔案
            current_time = end_time
            segment_index += 1

        os.unlink(video_path)  # 清掉原始暫存影片
        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as e:
        print("----------- VIDEO DETECT ERROR -----------")
        print(e)
        return jsonify({"success": False, "error": f"影片分析失敗: {e}"}), 500

# 啟動
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)