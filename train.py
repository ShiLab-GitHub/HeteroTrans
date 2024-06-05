import os
import math
import numpy as np
import pandas as pd
import json
import operator
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
import wandb
import dill
from data import create_dataloader
from model import PharmHGT as Model
from schedular import NoamLR
from utils import get_func,remove_nan_label

from torch.utils.tensorboard import SummaryWriter

def ci(y,f):
    y = y.squeeze().numpy()
    f = f.squeeze().numpy()
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def evaluate(dataloader,model,device,metric_fn,metric_dtype,task):
    metric = 0
    ci_value_sum = 0
    for bg,labels,gf in dataloader:
        bg,labels,gf = bg.to(device),labels.type(metric_dtype),gf.to(device)
        pred = model(bg,gf).cpu().detach()
        if task == 'classification':
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':
            pred = torch.softmax(pred,dim=1)
        num_task =  pred.size(1)
        if num_task >1:
            m = 0
            ci_value = 0
            for i in range(num_task):
                try:
                    m += metric_fn(*remove_nan_label(pred[:,i],labels[:,i]))
                    ci_value += ci(*remove_nan_label(labels[:,i],pred[:,i]))
                except:
                    print(f'only one class for task {i}')
            m = m/num_task
            ci_value = ci_value/num_task
        else:
            m = metric_fn(pred,labels.reshape(pred.shape))
            ci_value = ci(labels.reshape(pred.shape),pred)
        metric += m.item()*len(labels)
        ci_value_sum += ci_value*len(labels)
    metric = metric/len(dataloader.dataset)
    ci_value_sum = ci_value_sum/len(dataloader.dataset)
    
    return metric, ci_value_sum

def train(data_args,train_args,model_args,writer,seeds=[0,100,200,300,400]):
    # 设置参数
    epochs = train_args['epochs']                               
    device = train_args['device'] if torch.cuda.is_available() else 'cpu'
    save_path = train_args['save_path']

    wandb.config = train_args

    os.makedirs(save_path,exist_ok=True)
    preprocess = input("加载预处理数据y/n:")
    # 不同种子下训练
    results = []
    for seed in seeds:
        # 设置种子
        torch.manual_seed(seed)
        for fold in range(train_args['num_fold']):
            wandb.init(project='PharmHGT', entity='entity_name',group=train_args["data_name"],name=f'seed{seed}_fold{fold}',reinit=True)
            # 加载数据，具体加载看create_dataloader函数
            if preprocess=='y':
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_train.pkl','rb') as f:
                    trainloader = dill.load(f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_valid.pkl','rb') as f:
                    valloader = dill.load(f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_test.pkl','rb') as f:
                    testloader = dill.load(f)
            else:
                trainloader = create_dataloader(data_args,f'{seed}_fold_{fold}_train.csv',shuffle=True)
                valloader = create_dataloader(data_args,f'{seed}_fold_{fold}_valid.csv',shuffle=False,train=False)
                testloader = create_dataloader(data_args,f'{seed}_fold_{fold}_test.csv',shuffle=False,train=False)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_train.pkl','wb') as f:
                    dill.dump(trainloader, f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_test.pkl','wb') as f:
                    dill.dump(testloader, f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_valid.pkl','wb') as f:
                    dill.dump(valloader, f)
            print(f'dataset size, train: {len(trainloader.dataset)}, \
                    val: {len(valloader.dataset)}, \
                    test: {len(testloader.dataset)}')
            # 加载模型
            model = Model(model_args).to(device)
            # 指定优化器等
            optimizer = Adam(model.parameters())
            scheduler = NoamLR(
                optimizer=optimizer,
                warmup_epochs=[train_args['warmup']],
                total_epochs=[epochs],
                steps_per_epoch=len(trainloader.dataset) // data_args['batch_size'],
                init_lr=[train_args['init_lr']],
                max_lr=[train_args['max_lr']],
                final_lr=[train_args['final_lr']]
            )

            loss_fn = get_func(train_args['loss_fn'])
            metric_fn = get_func(train_args['metric_fn'])
            if train_args['loss_fn'] in []:
                loss_dtype = torch.long
            else:
                loss_dtype = torch.float32

            if train_args['metric_fn'] in []:
                metric_dtype = torch.long
            else:
                metric_dtype = torch.float32

            if train_args['metric_fn'] in ['auc','acc']:
                best = 0
                op = operator.ge
            else:
                best = np.inf
                op = operator.le
            best_epoch = 0
            # 训练模型
            for epoch in tqdm(range(epochs)):
                # 设置模型为训练模式
                model.train()
                total_loss = 0
                for bg,labels,gf in trainloader:
                    bg,labels,gf = bg.to(device),labels.type(loss_dtype).to(device),gf.to(device)
                    # 将bg传递给模型的forward函数，运行模型
                    pred = model(bg,gf)
                    num_task =  pred.size(1)
                    if num_task > 1:
                        loss = 0
                        for i in range(num_task):
                            loss += loss_fn(*remove_nan_label(pred[:,i],labels[:,i]))
                    else:
                        loss = loss_fn(*remove_nan_label(pred,labels.reshape(pred.shape)))
                    total_loss += loss.item()*len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                total_loss = total_loss / len(trainloader.dataset)
                
                # val
                model.eval()
                val_metric, val_ci = evaluate(valloader,model,device,metric_fn,metric_dtype,data_args['task'])
                if op(val_metric,best):
                    best = val_metric
                    best_epoch = epoch
                    torch.save(model.state_dict(),os.path.join(save_path,f'./best_fold{fold}.pt'))


                wandb.log({f'train {train_args["loss_fn"]} loss':round(total_loss,4),
                           f'valid {train_args["metric_fn"]}': round(val_metric,4),
                           'lr': round(math.log10(scheduler.lr[0]),4),
                           'CI': round(val_ci,4),
                           })
                writer.add_scalar(tag="loss", 
                      scalar_value=total_loss,
                      global_step=epoch
                      )
                writer.add_scalar(tag="lr", 
                      scalar_value=math.log10(scheduler.lr[0]),
                      global_step=epoch
                      )
                writer.add_scalar(tag="Validate/RMSE", 
                      scalar_value=val_metric,
                      global_step=epoch
                      )
                writer.add_scalar(tag="Validate/CI", 
                      scalar_value=val_ci,
                      global_step=epoch
                      )
            # evaluate on testset
            model = Model(model_args).to(device)
            state_dict = torch.load(os.path.join(save_path,f'./best_fold{fold}.pt'))
            model.load_state_dict(state_dict)
            model.eval()
            test_metric, test_ci = evaluate(testloader,model,device,metric_fn,metric_dtype,data_args['task'])
            results.append(test_metric)
            writer.add_scalar(tag="Test/RMSE", 
                      scalar_value=test_metric,
                      global_step=fold
                      )
            writer.add_scalar(tag="Test/CI", 
                      scalar_value=test_ci,
                      global_step=fold
                      )
            print(f'best epoch {best_epoch} for fold {fold}, val {train_args["metric_fn"]}:{best}, test: {test_metric}, test CI:{test_ci}')
            wandb.finish()
    return results


if __name__=='__main__':

    import sys
    config_path = sys.argv[1]
    # 加载数据集对应的配置文件（config文件夹下的数据集配置）
    config = json.load(open(config_path,'r'))
    # 拆分参数
    data_args = config['data']
    train_args = config['train']
    train_args['data_name'] = config_path.split('/')[-1].strip('.json')
    model_args = config['model']
    seed = config['seed']
    if not isinstance(seed,list):
        seed = [seed]
    
    print(config)
    # 调用train函数进行模型训练
    comment = "data_{}&seed_{}".format(train_args['data_name'], seed)
    writer = SummaryWriter(comment=comment)
    results = train(data_args,train_args,model_args,writer,seed)
    print(f'average performance: {np.mean(results)}+/-{np.std(results)}')