import numpy as np
from biosppy.signals import ecg
import time
from multiprocessing import Pool
import os
import json
import datetime
import mne
import pandas as pd
import glob

# def find_all_file_path(folder_dir: str, filetype: str = None) -> list:
#     """[summary]
#     해당 디렉토리 안에 있는 모든 파일을 읽어오는 함수.


#     Args:
#         folder_dir (str): xml 파일을 담고 있는 디렉토리
#         file_type (str) : 확장자 만약 None 일 경우 타입 상관없이 모든 파일 경로   ex) xml

#     Return:
#         [list]: 조건에 부합하는 파일들의 절대 경로
#     """
#     filepath_list = list()

#     for dirpath, _, filenames in os.walk(folder_dir):
#         for f in filenames:
#             path = os.path.abspath(os.path.join(dirpath, f))

#             if filetype == None:
#                 filepath_list.append(path)
#                 continue

#             if path.split(".")[-1] == filetype:
#                 filepath_list.append(path)

#     return filepath_list

def rpeaks_numpy(arr_filepath):
    ecg_arr = np.load(arr_filepath)
    out = ecg.ecg(signal=ecg_arr, sampling_rate=250., show=False)
    
    save_path = arr_filepath.replace('2hour','rpeak')
    np.save(save_path,out[2])
    print(save_path)

def get_rpeaks_from_edf(edf_file):
    signal_list = edf_file.__dict__['_raw_extras'][0]['ch_names']
    ECG_idx = signal_list.index('ECG')
    
    raw_data = edf_file.get_data()[ECG_idx]
    
    Hz = dict(edf_file.info)['sfreq']
    
    out = ecg.ecg(signal=raw_data, sampling_rate=Hz, show=False)
    return out[2]


def rpeaks_mne(row):
    filename = row[0]
    json_filepath = row[4]
    #filename = json_filepath.split('/')[-1].split('.')[0]
    
    with open(json_filepath) as f:
        label_json = json.load(f)
    
    label = label_json['Test_Result']['OSA_Risk'] 
    pid = label_json['Case_Info']['Patient_Number']
    
    data = mne.io.read_raw_edf(row[2])
    
    rpeaks = get_rpeaks_from_edf(data)
    
    save_path = os.path.join(*['/']+json_filepath.split('/')[:-2]+["rpeak"]+[f"{filename}_{pid}_{label}.npy"])
    np.save(save_path,rpeaks)
    print(save_path)





if __name__=='__main__':
    # src_dir = '/home/jangood1122/workspace/data/DATADAM_SLEEP/ai_data/2hour/'
    # arr_filepath = find_all_file_path(src_dir,'npy')

    # pool = Pool(processes=6)
    # pool.map(rpeaks_save,arr_filepath)
    # pool.close()
    # pool.join()


    src_dir = '/home/jangood1122/workspace/data/DATADAM_SLEEP/ai_data/final_data'
    edf_filelist = glob.glob(src_dir + "/**/*.edf", recursive=True)
    json_filelist = glob.glob(src_dir + "/**/*.json", recursive=True)
    print(len(os.listdir(src_dir)))

    edf_file_df = pd.DataFrame( [[filepath.split('/')[-1].split('_')[0][:-8],filepath.split('/')[-2],filepath] for filepath in edf_filelist if 'temp' not in filepath],columns=['filename','mid','edf_filepath'])
    json_file_df = pd.DataFrame( [[filepath.split('/')[-1].split('_')[0][:-5],filepath.split('/')[-2],filepath] for filepath in json_filelist if 'temp' not in filepath],columns=['filename','mid','json_filepath'])
    file_df = pd.merge(edf_file_df,json_file_df,on='filename')

    pool = Pool(processes=12)
    pool.map(rpeaks_mne,file_df.values)
    pool.close()
    pool.join()

