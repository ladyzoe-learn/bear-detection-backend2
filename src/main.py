from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd # 引入 pandas
import os # 引入 os

# 載入 YOLOv5 模型 (這部分維持不變)
# 注意：這裡的路徑是相對於您在 Render 上的部署環境，請確認是否正確
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 
# model.eval()

app = Flask(__name__)
CORS(app)  # 允許所有來源的跨域請求

@app.route('/')
def index():
    return render_template('index.html')

# 既有的 /predict 路由維持不變
@app.route('/predict', methods=['POST'])
def predict():
    # 這裡的 AI 預測邏輯完全不變
    # ... (您原本的預測程式碼)
    return jsonify({'result': 'dummy_prediction'}) # 假設的回傳值

# --- 我們新增的程式碼開始 ---

@app.route('/api/sightings', methods=['GET'])
def get_sightings():
    """
    提供台灣黑熊出沒紀錄的 API 端點。
    """
    csv_path = '台灣黑熊.csv' # CSV 檔案的路徑

    # 檢查 CSV 檔案是否存在
    if not os.path.exists(csv_path):
        return jsonify({"error": "Sighting data not found"}), 404

    try:
        # 使用 pandas 讀取 CSV 檔案 (預設使用 UTF-8)
        df = pd.read_csv(csv_path)

        # 確認必要的欄位是否存在
        required_columns = ['verbatimlatitude', 'verbatimlongitude', 'vernacularname', 'eventdate']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "CSV file is missing required columns"}), 500
        
        # 將 DataFrame 轉換為 JSON 格式，並設定 orient='records'
        # 這會產生一個 list of dictionaries，非常適合前端使用
        # 例如: [ { "col1": "valA", "col2": "valB" }, { ... } ]
        sighting_data = df.to_dict(orient='records')
        
        return jsonify(sighting_data)

    except Exception as e:
        # 如果在處理過程中發生任何錯誤，回傳 500 錯誤
        return jsonify({"error": str(e)}), 500

# --- 我們新增的程式碼結束 ---


if __name__ == '__main__':
    app.run(debug=True)

