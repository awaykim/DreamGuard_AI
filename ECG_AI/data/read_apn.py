import wfdb
import numpy as np
import matplotlib.pyplot as plt

record_name = 'a01'
data_path = './a01_data'  # 여기에 데이터셋이 위치한 실제 경로를 입력해야 합니다.

# ECG 데이터 로드 (.dat 파일)
ecg_record = wfdb.rdrecord(f'{data_path}/{record_name}')

# 호흡 정지 이벤트 정보 로드 (.apn 파일)
apnea_ann = wfdb.rdann(f'{data_path}/{record_name}', 'apn')

# ECG 신호 플롯
plt.figure(figsize=(10, 4))
plt.plot(ecg_record.p_signal[0:3000,0])  # 첫 3000개 샘플을 플롯합니다.
plt.title('ECG Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# 호흡 정지 이벤트 정보 출력
for i, annotation in enumerate(apnea_ann.symbol):
    start_time = apnea_ann.sample[i] / ecg_record.fs  # 시작 시간(초)
    print(f"Event: {annotation}, Start Time: {start_time} seconds")

# 주의: 실제 데이터셋 경로를 'data_path'에 정확히 입력해야 합니다.
