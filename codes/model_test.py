import torch 
import pandas as pd
import numpy as np
import os
from ECG_AI.ResNetAndrewNg import ResNetAndrewNg
from datetime import datetime
from DataDamSolver import get_performance
from run_hr import get_hr_from_rpeaks
from utils import get_rpeaks_from_edf
import mne

import warnings
warnings.filterwarnings('ignore')


###### 반드시 datadam2022-sleep 폴더 위치에서 실행시켜야합니다. 

def model_inference(model,input_tensor):
    model.eval()
    prob = model(input_tensor.float())[0][1].item()
    return prob

def adjust_len(arr,desire_len=256):
    if len(arr)>desire_len:
        arr = arr[:desire_len]
    elif len(arr)<desire_len:
        zero_n = desire_len-len(arr)
        arr = np.pad(arr,(0,zero_n))
    return arr

def main():

    ###
    print(f"[Start model test] {datetime.now()}")
    ### model load 

    print('[1] Load pretrained Model')
    checkpoint = torch.load('./data/1215_1_300_300_3_real_final/checkpoint_hr_sleep.pth', map_location=torch.device('cpu') )
    config = checkpoint['config']
    weight = checkpoint['weight']


    model = ResNetAndrewNg(config)

    model.load_state_dict(weight)

    model.eval()

    folder_dir = '/source_data/'
    test_csv = pd.read_csv('./data/sleep_test.csv')
    test_csv['edf_file_path'] = test_csv.filename.apply(lambda x : os.path.join(folder_dir,x,f'{x}.edf'))
    
    cutoff = 0.446

    print('[2] Start model inference')
    result_list = list()
    for idx,row in test_csv.iterrows():
        try:
            label = row.label
            filename = row.file_path.split('/')[-1]
            edf_file =  mne.io.read_raw_edf(row.edf_file_path)
            rpeaks = get_rpeaks_from_edf(edf_file)
            hr_arr = get_hr_from_rpeaks(np.array(rpeaks),hz=250,window=300,step=300)

            if np.nanmax(hr_arr)>160 and np.nanmin(hr_arr)<50:
                result_list.append([row.filename,row.label,None,None])
                continue

            hr_arr = np.nan_to_num(hr_arr)
            hr_arr = adjust_len(hr_arr,desire_len=256)


            #MODEL INFERENCE 
            hr_tensor = torch.from_numpy(np.array(hr_arr))
            hr_tensor = hr_tensor.unsqueeze(0).unsqueeze(0)
            prob = model_inference(model,hr_tensor)

            if cutoff is not None:
                pred = int(prob>cutoff)
            else:
                pred = None

            result_list.append([row.filename,row.label,prob,pred])
        except Exception as ex:
            print(ex)
            result_list.append([row.filename,None,None])

    print('[3] Collect all inference data')
    prob_df = pd.DataFrame(result_list,columns=['filename','label','prob','pred'])

    labels = prob_df[~prob_df.prob.isna()].label.values.ravel()
    probs = prob_df[~prob_df.prob.isna()].prob.values.ravel()

    auroc, acc, f_beta,sens,spec,cutoff,(TN, FP, FN, TP) = get_performance(labels, probs, 1,cutoff)

    result_df = pd.DataFrame([[auroc, acc, f_beta,sens,spec,cutoff,TN, FP, FN, TP]],columns=['auroc', 'acc', 'f_beta','sens','spec','cutoff','TN', 'FP', 'FN', 'TP'])


    print('[4] Save all results')
    prob_df.to_csv(f'./data/final_test_prob_{cutoff}.csv')
    result_df.to_csv(f'./data/final_test_result_{cutoff}.csv')

    print(result_df.drop(['auroc','cutoff'],axis=1))

    print(f"[Finish model test] {datetime.now()}")

    return True

if __name__=="__main__":
    main()