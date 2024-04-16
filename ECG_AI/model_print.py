from torch import torch
from ECG_AI.ResNetAndrewNg import ResNetAndrewNg

config = {
    "kernel_size": 7,  # 예시 값
    "dropout": 0.15,     
    "output_size": 2   
}

model = ResNetAndrewNg(config)
model = torch.load('checkpoint_hr_sleep.pth', map_location=torch.device('cpu'))

model_str = str(model)
with open('model/model_output.txt', 'w') as f:
    f.write(model_str)
# print(model)
print("done")