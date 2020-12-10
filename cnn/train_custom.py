import os
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
import torch
import torch.nn as nn
import torchvision
#import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
#from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# Sortof custom

import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axesv
import numpy as np
import math
import time

import argparse
import h5py

# Ray
import json
from functools import partial
import itertools
#import ray
#from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
# Optimizers
from adabelief_pytorch import AdaBelief
from ranger import Ranger
import adabound
# Own modules
import model.ArtificialFO as AFO
import model.CNNmodule as CNNmodule
import model.ResidualCNN as RCNNmodule
import model.CustomResNet as CustomResNet
import model.load_various as ld
import model.AdamRNW as AdamRNW

import utils

def report(f_name,e_name,ksplit,epoch, train_loss, valid_loss, valid_acc):
    current = './experiments/'+f_name+e_name+'/k'+str(ksplit)+'e'+str(epoch)
    np.savetxt(current+'train_loss.txt',train_loss,fmt='%f')
    np.savetxt(current+'valid_loss.txt',np.array([valid_loss]),fmt='%f')
    np.savetxt(current+'valid_acc.txt',np.array([valid_acc]),fmt='%f')
    return None

def run_sweep(default,sweep_conf):
    f_name = 'sw'    
    keys, values = zip(*sweep_conf.items())
    for key in keys: f_name += '_'+key 
    f_name +='/'
    #print(f_name)
    #print(keys)
    #print(values)
    c=0
    for bundle in itertools.product(*values):
        #if (c!=5) and (c != 2):
        #    c+=1
        #    continue
        #if c < 12:
        #    c+=1
        #    continue
        #print(c)
        d = dict(zip(keys, bundle))
        #print(d)
        config_updated = dict(default)
        config_updated.update(d)
        e_name = str(c)
        train(f_name,e_name, config_updated)
        
        c+=1
    
    #for key, value in sweep_conf.items():
    #    f_name += '_'+key
    #    name_list.append(key)
    #    sw_list.append(value)
    #print(f_name)
    #for combination in list(itertools.product(*sw_list)):
        #print(combination)
        #config_updated = dict(default)
        
        
    #    train(f_name,e_name, config_updated)
    return None

def train(f_name,e_name, config):
    # Possible training sets:
    # TRAIN_xray_pTD_unnorm_3ch.hdf5
    # TRAIN_xray_MMII_unnorm_3ch.hdf5
    current_location='/home/jovyan/work/mlfoss/cnn2'
    #'/home/paperspace/jesper/mlfoss/cnn2''/home/jovyan/work/mlfoss/cnn2'
    #print(torch.cuda.device_count())
    config['log_interval'] = 50
    config['save']=1
    if config['save']:
        os.makedirs('experiments/'+f_name,exist_ok=True)
        os.makedirs('experiments/'+f_name+e_name,exist_ok=True)
        with open(current_location+'/experiments/'+f_name+e_name+'/config.json', "w") as fp:
            json.dump(config, fp)
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #json.dump(config,open(current_location+'/experiments/'+f_name+e_name+'/dict.json', "w"))
    # Print hyperparameters
    print('\nRun started. \nPrinting the hyperparameters:')
    #for key,value in hyperparameter_defaults.items(): print(key, ' : ', value)
    for key in config.keys(): print(key, ' : ', config[key])
    # Setup
    ind_sets = ld.create_indices(current_location+'/data/'+config['dataset']+str(config['n_input_ch'])+'ch.hdf5', config['kfold'])
    ######################################
    ### Check for mix parameter here below
    ######################################
    #Gather FO's
    if config['da_addafo']:
        fo= AFO.gather_fos(current_location+'/data/'+config['dataset'][:-8]+'_globalref_unnorm_'+str(config['n_input_ch'])+'ch.hdf5',mix=config['mix'])
    else:
        fo=[]
    #fo = AFO.filter_fo(fo,-0.5,0.5)
    # K-fold cross validation
    train_loaders = []
    valid_loaders = []
    norms = ld.load_norms(current_location+'/data/'+config['dataset']+str(config['n_input_ch'])+'ch.hdf5')
    for train_indices, valid_indices in ind_sets:
        tforms = [transforms.Normalize(mean = norms[:,0], std = norms[:,1])]
        tforms=[]
        if config['da_addafo']:
            top_images = AFO.new_topim(fo)
            tforms += [AFO.AddAFO(32,top_images)]
        if config['da_rotations'] : 
            tforms += [ld.RandFlipRot(32, p=0.5)]
        #tforms += [transforms.Normalize(mean = norms[:,0], std = norms[:,1])]
        if config['da_zero_ch'] :
            tforms += [ld.ZeroChannel(32, ch=config['da_zero_ch'])]
        # Train
        train_loaders.append(ld.k_dataload(current_location+'/data/'+config['dataset']+str(config['n_input_ch'])+'ch.hdf5',
                                            train_indices, tforms, fo, batch_size = config['BATCH_SIZE'], num_workers=config['num_workers']))
        # Valid
        valid_loaders.append(ld.k_dataload(current_location+'/data/'+config['dataset']+str(config['n_input_ch'])+'ch.hdf5',
                                            valid_indices, [tforms[0]], fo, batch_size = config['BATCH_SIZE'], num_workers=config['num_workers']))
    # k Models
    models = []
    for i in range(config['kfold']):
        #RCNNmodule.resnext50_32x4d_foss(num_classes=2))#
        #CNNmodule.ConvNet( input_pictures_dim[0]*input_pictures_dim[1], 2,n_input_ch, n_layer_outputs, kernels_shape,dropout=0.2)
        #CustomResNet.resnet_foss(num_classes=2) )
        models.append(CNNmodule.ConvNet( config['input_pictures_dim'][0]*config['input_pictures_dim'][1], config['n_outputs'],config['n_input_ch'], config['n_layer_outputs'], config['kernels_shape'], config['activation'], dropout=0.2).to(device=config['device']))
    #wandb.watch(models[0])
    # k Optimizers
    weights_for_loss = torch.tensor([1,config['weights_for_loss_positive']], dtype=torch.float).to(device=config['device'])
    def cross_entropy(pred, targets, label_smoothing=config['lsmooth'], weights=weights_for_loss):
        """https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/17"""
        logsoftmax = nn.LogSoftmax(dim=1)
        soft_targets = F.one_hot(targets, num_classes=2)
        soft_targets = label_smoothing + soft_targets * (1-2*label_smoothing)
        return torch.mean(torch.sum(weights*(- soft_targets * logsoftmax(pred)), 1))
    if config['lsmooth']:
        criterion = cross_entropy
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_for_loss)#cross_entropy
    optimizers = []
    for i in range(config['kfold']):
        if config['optimizer_ind'] == 0:
            optimizers.append( optim.SGD(models[i].parameters(), lr=config['lr'], momentum=0.9, nesterov=True))
        elif config['optimizer_ind'] == 1:
            optimizers.append( adabound.AdaBound(models[i].parameters(), lr=config['lr'], final_lr=0.1))
        elif config['optimizer_ind'] == 2:
            optimizers.append(AdamRNW.AdamRNW(models[i].parameters(), weight_decay=4e-3, lr=config['lr']))
        elif config['optimizer_ind'] == 3:
            optimizers.append(Ranger(models[i].parameters(), lr=config['lr']))
        elif config['optimizer_ind'] == 4:
            optimizers.append(AdaBelief(models[i].parameters(), lr=config['lr'], eps=1e-10, betas=(0.9,0.999), weight_decouple = True, rectify = True))
        else:
            optimizers.append(optim.Adam(models[i].parameters(), weight_decay=4e-3, lr=config['lr']))
    # Train
    start = time.time()
    for ksplit in range(config['kfold']):
        if ksplit > config['max_kfold_run']-1: continue
        print('\nK-fold = ' + str(ksplit))        
        model = models[ksplit]
        optimizer = optimizers[ksplit]
        trainloader = train_loaders[ksplit]
        validloader = valid_loaders[ksplit]
        # Give base line
        bl=0
        for batch_idx,(data, target) in enumerate(validloader): 
            bl += target.sum().item()
        print("Baseline = {:.3f}".format( 1- bl/float(len(ind_sets[ksplit][1]))))
            
        #lr_run = config['lr']+0.0
        correct = 0
        total=0
        for epoch in range(1, config['N_EPHOCS']+1):
            print('')
            print('Epoch = '+str(epoch))
            # Train
            steps = 0
            train_loss_tot =[]
            train_loss = 0
            model.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device=config['device']), target.to(device=config['device'])
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)#F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                pred = output.data.max(1, keepdim=True)[1]
                total += target.size(0)
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                
                steps += config['BATCH_SIZE']
                if batch_idx > 0 and batch_idx % config['log_interval'] == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                        epoch, batch_idx * config['BATCH_SIZE'], len(ind_sets[ksplit][0]),
                        100. *config['BATCH_SIZE']* batch_idx / len(ind_sets[ksplit][0]), train_loss/(config['log_interval']*config['BATCH_SIZE']), steps))
                    train_loss_tot.append(train_loss/(config['log_interval']*config['BATCH_SIZE']))
                    # 
                    # Create custom logging function
                    #
                    #wandb.log({'Loss/train/k'+str(ksplit)+'': train_loss/(config.log_interval*config.BATCH_SIZE),
       #                        'Epoch':(epoch-1)+steps/float(len(ind_sets[ksplit][0]))})
                    train_loss = 0
            train_acc = correct.item()/total
            # Validate
            model.eval()
            val_loss = 0
            correct = 0
            total=0
            with torch.no_grad():
                for batch_idx,(data, target) in enumerate(validloader):
                    data, target = data.to(device=config['device']), target.to(device=config['device'])
                    
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    total += target.size(0)
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            #wandb.log({'Loss/valid/k'+str(ksplit)+'': val_loss})
            #wandb.log({'Loss/valid/k'+str(ksplit)+'': val_loss/ float(len(ind_sets[ksplit][1])),
            #           'Accuracy/valid/k'+str(ksplit)+'': correct.item() / float(len(ind_sets[ksplit][1])),
             #          'Epoch':epoch})

            if epoch % config['lr_div_rate']     == 0:
                #lr_run = utils.cyclic_learningrate(config['lr'], lr_run)
                for param_group in optimizers[ksplit].param_groups:
                    param_group['lr']= param_group['lr']/5#lr_run
                    
            val_acc = correct.item() / float(len(ind_sets[ksplit][1]))
            if config['save']:
                valid_loss_pr = val_loss/ float(len(ind_sets[ksplit][1]))
                report(f_name,e_name, ksplit,epoch,train_loss_tot, valid_loss_pr, val_acc)
                torch.save({
                    'state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    'experiments/'+f_name+e_name+'/ConvNet_k'+str(ksplit)+'_e'+str(epoch)+'.tar')
            #tune.report(loss=(val_loss / float(len(ind_sets[ksplit][1]))), accuracy=correct.item() / total)

            print('Accuracy: ',val_acc)#/float(len(valid_loaders_list[ksplit].dataset)))
            print('Epoch time: ', utils.timeSince(start, time.time()))
        #writer.close()

#temp_dir="/home/jovyan/work/mlfoss/cnn2/"
if __name__ == '__main__':
    config = dict(
        dataset = 'MIX_TRAIN_unnorm_',#'TRAIN_xray_CD_full_unnorm_',#'MIX_TRAIN_unnorm_', #'TRAIN_xray_MMII_unnorm_',#'TRAIN_xray_pTD_unnorm_',#3ch.hdf5',
        mix = 1,
        da_addafo = 0,
        da_rotations = 1,
        da_zero_ch = 0,
        kfold = 5,
        max_kfold_run = 5,
        BATCH_SIZE = 64,
        num_workers = 4,
        input_pictures_dim = (32,32),
        n_input_ch=2, #3
        n_outputs = 2,
        kernels_shape = [3,3,3],
        n_layer_outputs = [16,32,64,64],
        activation = 'relu',
        lr = 0.001,
        lsmooth = 0.0,
        lr_div_rate = 6,
        weights_for_loss_positive = 2,
        N_EPHOCS = 15,
        optimizer_ind = 2
    )
                              
    #sweep_conf = {'BATCH_SIZE':[64]#,128,256], 
                  #'lr':[0.001]}#,0.002,0.008,0.016]} #[0.00025]}
    
    
    #sweep_conf = {'BATCH_SIZE':[64], 'lr':[0.001]}
    #sweep_conf = {'dataset' : ['TRAIN_xray_MMII_unnorm_',
    #                           'TRAIN_xray_pTD_unnorm_',
    #                           'TRAIN_xray_CD_unnorm_'],
    #              'da_addafo':[0,1],
    #              'weights_for_loss_positive':[1,2,3,4]}
    #sweep_conf = {'dataset' : ['TRAIN_xray_CD_unnorm_',
    #                           'TRAIN_xray_CD_full_unnorm_']}
    #sweep_conf = {'n_input_ch':[2,3]}
    #sweep_conf = {'optimizer_ind' : [1,2,3,4,5],
    #              'lr_div_rate' : [6, 25],
    #              'lr' : [0.008, 0.001]}
    sweep_conf = {'weights_for_loss_positive':[4],
        'da_zero_ch':[1,2]}
    #sweep_conf = {'kernels_shape' : [[5,5,5], [7,7,7]]}
    run_sweep(config,sweep_conf)
