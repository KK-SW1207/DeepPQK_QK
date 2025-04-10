
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from torch.cuda.amp import autocast as autocast, GradScaler  # uncomment lines related with `amp` to use apex
from torch.nn import MultiheadAttention
from dataset2 import MyDataset
from model import DeepPQK_QK, test
import torch.nn as nn  
import torch.nn.functional as F  

print(sys.argv)

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
SHOW_PROCESS_BAR = False
data_path = './pre_data/'
seed = np.random.randint(33927, 33928) ##random 
print(seed)
path = Path(f'../runs/DeepPQK_{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')
device = torch.device("cuda:0")  # or torch.device('cpu')
            
max_seq_len = 1000  
max_pkt_len = 63
max_smi_len = 150

batch_size = 165
n_epoch = 40
interrupt = None
save_best_epoch = 13 #  when `save_best_epoch` is reached and the loss starts to decrease, save best model parameters
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# GPU uses cudnn as backend to ensure repeatable by setting the following (in turn, use advances function to speed up training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

writer = SummaryWriter(path)
f_param = open(path / 'parameters.txt', 'w')

print(f'device={device}')
print(f'seed={seed}')
print(f'write to {path}')
f_param.write(f'device={device}\n'
          f'seed={seed}\n'
          f'write to {path}\n')
               

print(f'max_seq_len={max_seq_len}\n'
      f'max_pkt_len={max_pkt_len}\n'
      f'max_smi_len={max_smi_len}')

f_param.write(f'max_seq_len={max_seq_len}\n'
      f'max_pkt_len={max_pkt_len}\n'
      f'max_smi_len={max_smi_len}\n')


assert 0<save_best_epoch<n_epoch

model = DeepPQK_QK(512,512)
model = model.to(device)
print(model)
f_param.write('model: \n')
f_param.write(str(model)+'\n')
f_param.close()

data_loaders = {phase_name:
                    DataLoader(MyDataset(data_path, phase_name,
                                         max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None),
                               batch_size=batch_size,
                               pin_memory=True,
                               num_workers=8,
                               shuffle=True)
                for phase_name in ['training', 'validation', 'test',"test105"]}
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, epochs=n_epoch,
                                          steps_per_epoch=len(data_loaders['training']))
"""optimizer = optim.SGD(model.parameters(), lr=5e-5)  # 使用SGD作为优化器  
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=n_epoch,  
                                          steps_per_epoch=len(data_loaders['training']))"""


#print(len(data_loaders['training'][0]))
loss_function = nn.MSELoss(reduction='sum')

# fp16
#model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
        
start = datetime.now()
print('start at ', start)

best_epoch = -1
best_val_loss = 100000000
best_val_CORR=0.75
for epoch in range(1, n_epoch + 1):  
    tbar = tqdm(enumerate(data_loaders['training']), disable=True, total=len(data_loaders['training']))  
    #print(tbar)  
    for idx, (*x, y) in tbar:  
        model.train()  
  
        for i in range(len(x)):  
            x[i] = x[i].to(device)  
        y = y.to(device)  
  
        optimizer.zero_grad()  
        # output = model(*x)  
        # loss = loss_function(output.view(-1), y.view(-1))  
  
        # fp16  
        with autocast():  
            output = model(*x)  
            loss = loss_function(output.view(-1), y.view(-1))  
        loss.backward()   
       
        optimizer.step()  
        scheduler.step()  
  
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() / len(y):.3f}')  
   
    
    loss_files = {}  
    R_files={}
    for _p in ['training', 'validation', 'test',"test105"]:
        filename2 = f"NEW{_p}_R.txt"  # 创建文件名 
        filename = f"NEW{_p}_loss.txt"
        loss_files[_p] = open(filename, 'a')
        R_files[_p] = open(filename2, 'a')
        performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
        print(f'Epoch {epoch}, {_p} Loss: {performance["loss"]}, {_p} c_index: {performance["c_index"]}')

        loss_files[_p].write(f"{_p} Loss: {performance['loss']}\n") 
        R_files[_p].write(f"{_p} R: {performance['c_index']}\n")


        for i in performance:  
            writer.add_scalar(f'{_p} {i}', performance[i], global_step=epoch)  
        if _p=='validation' and epoch>=save_best_epoch and performance["CORR"]>best_val_CORR:  
            best_val_CORR = performance['CORR']  
            best_epoch = epoch  
            torch.save(model.state_dict(), path / 'best_model.pt') 
model.load_state_dict(torch.load(path / 'best_model.pt'))
with open(path / 'result.txt', 'w') as f:
    f.write(f'best model found at epoch NO.{best_epoch}\n')
    for _p in ['training', 'validation', 'test',]:
        performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')
        f.write('\n')
        print()

print('training finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
