from flask import Flask, request, jsonify
import zipfile
import os
import torch
import numpy as np
import pandas as pd
from model_eval import predict_new_data  
from ResNetAndrewNg import ResNetAndrewNg

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
        return jsonify({'prob': prob, 'percentile': percentile}), 200
    else:
        return jsonify({'error': '필요한 파일이 없습니다.'}), 400

@app.route('/upload', methods=['POST'])
def upload_file():

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
        
    
   


if __name__ == '__main__':
    app.run(debug=True)
