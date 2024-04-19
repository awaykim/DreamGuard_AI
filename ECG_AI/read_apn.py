import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# record_name ='data/a01_data/a01'  
record_name ='data/apnea-ecg-database-1.0.0/a01'
# ECG 데이터 로드 (.dat 파일)
ecg_record = wfdb.rdrecord(record_name)

# 호흡 정지 이벤트 정보 로드 (.apn 파일)
apnea_ann = wfdb.rdann(record_name, 'apn')

time_array = np.arange(ecg_record.sig_len) / ecg_record.fs

# ECG 신호 플롯
plt.figure(figsize=(10, 4))
# plt.plot(ecg_record.p_signal[0:3000,0])  
plt.plot(time_array[0:1000], ecg_record.p_signal[0:1000,0])
plt.title('ECG Signal')
# plt.xlabel('Sample')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()

# 호흡 정지 이벤트 정보 출력
for i, annotation in enumerate(apnea_ann.symbol):
    start_time = apnea_ann.sample[i] / ecg_record.fs  # 시작 시간(초)
    print(f"Event: {annotation}, Start Time: {start_time} seconds")
