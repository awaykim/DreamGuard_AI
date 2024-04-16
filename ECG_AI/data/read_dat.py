import wfdb
import matplotlib.pyplot as plt

# 데이터셋 경로와 레코드 이름 설정
record_name = './apnea-ecg-database-1.0.0/a01'  # 예시로, a01이라는 레코드 이름을 사용

# 레코드 읽기
record = wfdb.rdrecord(record_name)

# 신호 데이터와 필드 정보 추출
signal = record.p_signal
fields = record.__dict__
sampling_rate = record.fs
n_samples = record.sig_len
total_seconds = n_samples / sampling_rate
total_minutes = total_seconds / 60
print(f"Total time: {total_seconds} seconds ({total_minutes} minutes)")

# 첫 번째 신호 채널을 시각화
plt.plot(signal[:, 0])
plt.title('ECG Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

