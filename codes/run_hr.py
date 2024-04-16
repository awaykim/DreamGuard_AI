import os
import sys
print(os.path.abspath('.'),"########")

try:
    from ECG_AI.DataDamloader2_hr import *
    from codes.DataDamSolver import *
    from ECG_AI.ResNetAndrewNg import *
except ModuleNotFoundError:
    sys.path.append('./model/code')
    from ECG_AI.DataDamloader2_hr import *
    from codes.DataDamSolver import *
    from ECG_AI.ResNetAndrewNg import *

import pandas as pd
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import pathlib
import os
from shutil import copyfile

# copyfile(src, dst)


def main():
    
    param_list = [   #step window kernel_size 
        
        [60*1,60*10,7],
        [60*3,60*10,7],
        [60*5,60*10,7],
        [60*1,60*5,7],
        [60*3,60*5,7],
        [60*5,60*5,7],
        [60*1,60*10,5],
        [60*3,60*10,5],
        [60*5,60*10,5],
        [60*1,60*5,5],
        [60*3,60*5,5],
        [60*5,60*5,5],
        [60*1,60*10,9],
        [60*3,60*10,9],
        [60*5,60*10,9],
        [60*1,60*5,9],
        [60*3,60*5,9],
        [60*5,60*5,9],
        [60*1,60*10,3],
        [60*3,60*10,3],
        [60*5,60*10,3],
        [60*1,60*5,3],
        [60*3,60*5,3],
        [60*5,60*5,3],
    ]
    
    
    for idx,(param) in enumerate(param_list):
        config = get_config(param[0],param[1],param[2])
        solver = DataDamSolver(config)
        solver.fit()
        del(solver)

def get_hr_from_rpeaks(rpeak_arr,hz,window,step):
    hr_sec_list = list()
    for idx in range(int(np.ceil(rpeak_arr.max()/(hz*step)))):
        target_rpeaks = rpeak_arr[(rpeak_arr>=hz*step*idx)&(rpeak_arr<(hz*step*idx)+(hz*window))]
        rr_interval_median = np.median(np.diff(target_rpeaks))
        hr = 60*hz/rr_interval_median
        hr_sec_list.append(hr)
    return hr_sec_list


def get_hr_arr(file_df_path,hz,window,step):
    data_list = list()
    data_len_list = list()

    file_df=pd.read_csv(file_df_path)

    minimum = 60*60*3/step

    for idx,row in file_df.iterrows():
        rpeak_arr = np.load(row.file_path)
        label = row.label

        hr_sec_list = get_hr_from_rpeaks(rpeak_arr,hz,window,step)
        
        if np.array(hr_sec_list).max()<160 and np.array(hr_sec_list).min()>50: # and len(hr_sec_list)>=minimum:
            data_list.append([hr_sec_list,label])
            data_len_list.append(len(hr_sec_list))
    
    return data_list,data_len_list
        
    
def get_config(step,window,kernel_size):
    # 학습할 configuration을 세팅합니다.
    config = dict()

    hz = 250
    # window = 60*10
    # step = 60*2
    
    folder_name = '1215_1'
    train_list,train_len_list = get_hr_arr(f'../label/{folder_name}/train_rpeak_pid.csv',hz,window,step)
    valid_list,valid_len_list = get_hr_arr(f'../label/{folder_name}/valid_rpeak_pid.csv',hz,window,step)
    test_list,test_len_list = get_hr_arr(f'../label/{folder_name}/test_rpeak_pid.csv',hz,window,step)


    # 실행시킬 라벨을 입력하세요
    config["data"] = dict()
    config["data"]["train"] = [train_list,max(train_len_list)]  #[arr,label]
    config["data"]["valid"] = [valid_list,max(valid_len_list)]
    config["data"]["test"] = [test_list,max(test_len_list)]

    # 모델을 세팅하세요
    config["kernel_size"] = kernel_size
    config["dropout"] = 0.15
    config["output_size"] = 2
    config["model"] = ResNetAndrewNg(config).cuda()

    config["dataloader"] = dict()
    
    for k, v in config["data"].items():
        if k == "train":
            shuffle = False
            sampler = ImbalancedDatasetSampler(DataDamloader2_hr(v[0],256))
        else:
            shuffle = False
            sampler = None
        config["dataloader"][k] = DataLoader(
            dataset=DataDamloader2_hr(v[0],256),
            batch_size=8,
            drop_last=False,
            sampler=sampler
        )

    # loss function을 세팅하세요
    config["loss_function"] = torch.nn.CrossEntropyLoss()

    # optimizer를 세팅하세요
    config["optimizer"] = torch.optim.Adam(
        config["model"].parameters(), lr=0.0001, weight_decay=0.00001
    )

    # epoch을 세팅하세요
    config["epochs"] = 50

    # save_path를 세팅하세요
    config["save_path"] = f"/home/jangood1122/workspace/collaboration/datadam2022-sleep/result/{folder_name}_{window}_{step}_{kernel_size}/"
    
    pathlib.Path(config['save_path']).mkdir(exist_ok=True,parents=True)
    copyfile('./ResNetAndrewNg.py',os.path.join(config['save_path'],'ResNetAndrewNh.py'))


    return config


if __name__ == "__main__":
    main()
