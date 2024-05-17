from flask import Flask, request, jsonify, render_template
import zipfile
import os
import torch
import numpy as np
import pandas as pd
from model_predict import predict_new_data  
from ResNetAndrewNg import ResNetAndrewNg
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}
CSV = pd.read_csv('data/percentile_dummy.csv')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_files_and_predict(record_name):
    
    dat_path = os.path.join(UPLOAD_FOLDER, f"{record_name}.dat")
    hea_path = os.path.join(UPLOAD_FOLDER, f"{record_name}.hea")
    data = os.path.join(UPLOAD_FOLDER, record_name)
    # 파일 존재 여부 확인
    if os.path.exists(dat_path) and os.path.exists(hea_path):
        prob, percentile = predict_new_data(data, CSV)
        print(prob, percentile)
        return jsonify({'record_id': record_name, 'prob': prob, 'percentile': percentile}), 200
    else:
        return jsonify({'error': '필요한 파일이 없습니다.'}), 400

@app.route('/ecg_post', methods=['POST'])
def upload_ecg_file():
    file = request.files['file']

    if 'file' not in request.files:
        return jsonify({'error': '파일이 요청에 포함되지 않았습니다.'}), 400
    
    if file.filename == '':
        return jsonify({'error': '파일명이 없습니다.'}), 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 압축 해제
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])
        os.remove(filepath)  # 원본 ZIP 파일 삭제

        return check_files_and_predict(filename.rsplit('.', 1)[0]) 
    else:
        return jsonify({'error': '허용되지 않은 파일 형식입니다.'}), 400
    
    
@app.route('/sound_post', methods=['POST'])
def upload_sound_file():

    file = request.files['file']
    filename = file.filename

    if 'file' not in request.files:
        return jsonify({'error': '파일이 요청에 포함되지 않았습니다.'}), 400
    
    if file.filename == '':
        return jsonify({'error': '파일명이 없습니다.'}), 400
    
    time.sleep(11)  

    return jsonify({'record_id': filename.rsplit('.', 1)[0], 'prob': 0.7526391419384874, 'percentile': 18.54}), 200

@app.route('/ecg')
def upload_ecg_form():
    return render_template('ecg_upload.html')

@app.route('/sound')
def upload_sound_form():
    return render_template('sound_upload.html')

@app.route('/')
def upload_index_form():
    return render_template('index.html')

# 로컬 테스트 용
# if __name__ == '__main__':
#     app.run(debug=True)

# 배포용
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

