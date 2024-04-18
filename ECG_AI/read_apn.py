import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# record_name ='data/a01_data/a01'  # 여기에 데이터셋이 위치한 실제 경로를 입력해야 합니다.
record_name ='data/apnea-ecg-database-1.0.0/a01'
# ECG 데이터 로드 (.dat 파일)
ecg_record = wfdb.rdrecord(record_name)

# 호흡 정지 이벤트 정보 로드 (.apn 파일)
apnea_ann = wfdb.rdann(record_name, 'apn')

# 호흡 정지 이벤트 정보 출력
for i, annotation in enumerate(apnea_ann.symbol):
    start_time = apnea_ann.sample[i] / ecg_record.fs  # 시작 시간(초)
    print(f"Event: {annotation}, Start Time: {start_time} seconds")

# ECG 신호 플롯
plt.figure(figsize=(10, 4))
plt.plot(ecg_record.p_signal[0:3000,0])  # 첫 3000개 샘플을 플롯합니다.
plt.title('ECG Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()





# 주의: 실제 데이터셋 경로를 'data_path'에 정확히 입력해야 합니다.
