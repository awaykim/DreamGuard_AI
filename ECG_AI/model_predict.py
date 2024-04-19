import torch
import numpy as np
from ResNetAndrewNg import ResNetAndrewNg
import pandas as pd
import wfdb
from biosppy.signals import ecg
from datetime import datetime
import time
import threading


# 시간 추적을 위한 함수
def track_time(start_time, stop_event):
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"\rElapsed time: {elapsed_time:.2f} seconds", end='')
        time.sleep(0.5)  # 0.5초마다 업데이트


def find_percentile(score, df):
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
    raw_data = record.p_signal[:, 0]
    Hz = record.fs
    print("- 심전도로부터 RPEAKS 추출")
    out = ecg.christov_segmenter(signal=raw_data, sampling_rate=Hz)

    return out[0]

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

    start_comment = '== 수면 무호흡증 진단 시작 =='
    print(start_comment)
    start = datetime.now()

    # 모델 로드
    checkpoint = torch.load('model/checkpoint_hr_sleep.pth', map_location=torch.device('cpu'))
    print("- 모델 로드 완료")
    config = checkpoint['config']
    weight = checkpoint['weight']
    model = ResNetAndrewNg(config)
    
    model.load_state_dict(weight)
    model.eval()

    # .dat 파일 로드 및 전처리
    rpeaks = get_rpeaks_from_dat(record_name)
    hr_arr = get_hr_from_rpeaks(np.array(rpeaks), hz=100, window=300, step=300) 
    hr_arr = np.nan_to_num(hr_arr)
    hr_arr = adjust_len(hr_arr, desire_len=256)
    print("- 데이터 전처리 완료")
    # 모델 추론
    print("== OSA 추론 완료 ==")
    hr_tensor = torch.from_numpy(np.array(hr_arr))
    hr_tensor = hr_tensor.unsqueeze(0).unsqueeze(0)
    prob = model_inference(model, hr_tensor)
    percentile = find_percentile(prob, df)


    print(f"Predicted probability: {prob}")
    print(f"Percentile(high): {percentile}")
    print('Total Elapsed time:', datetime.now() - start)

    return prob, percentile


if __name__ == "__main__":
    # test_data 
    df = pd.read_csv('data/percentile_dummy.csv')
    record_name = 'data/apnea-ecg-database-1.0.0/a01' 
    
    
    prob, percentile = predict_new_data(record_name, df)
    