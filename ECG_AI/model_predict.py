import torch
import numpy as np
from ResNetAndrewNg import ResNetAndrewNg
import pandas as pd
import wfdb
from biosppy.signals import ecg
from datetime import datetime
# 필요한 함수 정의

def find_percentile(score, df):
    per_commnet = '백분위수 계산 중 . . .'
    print(per_commnet)
    rounded_score = round(score, 2)
    row = df[df['Score'].round(2) == rounded_score]
    if not row.empty:
        return row['Percentile'].iloc[0]
    else:
        return "해당 Score 값을 가진 데이터가 없습니다."

def model_inference(model, input_tensor):
    with torch.inference_mode():
            prob = model(input_tensor.float())[0][1].item()
    return prob

def adjust_len(arr, desire_len=256):
    current_len = len(arr)
    if current_len < desire_len:
        zero_n = desire_len - current_len
        arr = np.pad(arr, (0, zero_n), 'constant')
    return arr[:desire_len]

def get_rpeaks_from_dat(record_name):
    
    record = wfdb.rdrecord(record_name)
    print(":: Record 불러오기 완료 ")
    raw_data = record.p_signal[:, 0]
    Hz = record.fs
    print(":: raw data, Hz 추출 완료")
    out = ecg.ecg(signal=raw_data, sampling_rate=Hz, show=False)
    print(":: 신호처리 완료")
    return out['rpeaks']

def get_hr_from_rpeaks(rpeak_arr, hz, window, step):
    hr_sec_list = list()
    for idx in range(int(np.ceil(rpeak_arr.max()/(hz*step)))):
        target_rpeaks = rpeak_arr[(rpeak_arr >= hz*step*idx) & (rpeak_arr < (hz*step*idx) + (hz*window))]
        rr_interval_median = np.median(np.diff(target_rpeaks))
        hr = 60*hz/rr_interval_median
        hr_sec_list.append(hr)
    return hr_sec_list

# 새로운 데이터에 대한 예측 확률을 반환하는 함수
def predict_new_data(record_name, df):
    # 모델 로드
    load_comment = '모델 로드 중 . . .'
    print(load_comment)

    checkpoint = torch.load('model/checkpoint_hr_sleep.pth', map_location=torch.device('cpu'))
    config = checkpoint['config']
    weight = checkpoint['weight']
    model = ResNetAndrewNg(config)
    
    model.load_state_dict(weight)
    model.eval()

    # .dat 파일 로드 및 전처리

    print("파일로부터 peaks 추출 중. . .")
    rpeaks = get_rpeaks_from_dat(record_name)
    print("hr 추출 중. . .")
    hr_arr = get_hr_from_rpeaks(np.array(rpeaks), hz=100, window=300, step=300) 
    print("데이터 조정 중. . .")
    hr_arr = np.nan_to_num(hr_arr)
    hr_arr = adjust_len(hr_arr, desire_len=256)
    # 모델 추론
    infer_comment = '추론 중 . . .'
    print(infer_comment)
    hr_tensor = torch.from_numpy(np.array(hr_arr))
    hr_tensor = hr_tensor.unsqueeze(0).unsqueeze(0)
    prob = model_inference(model, hr_tensor)
    percentile = find_percentile(prob, df)

    return prob, percentile


if __name__ == "__main__":
    # test_data 
    df = pd.read_csv('data/percentile_dummy.csv')
    record_name = 'data/apnea-ecg-database-1.0.0/a01' 
    start_comment = '수면 무호흡증 진단을 시작합니다 . . .'
    print(start_comment)
    start = datetime.now()
    prob, percentile = predict_new_data(record_name, df)
    print(f"Predicted probability: {prob}")
    print(f"Percentile(high): {percentile}")
    print('Elapsed time:', datetime.now() - start)
