# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:16:02 2019

@author: Mirac
"""
from imports import *
from lossFunctions import *
from evalCnn import *
import os
import sys
import csv
from shutil import rmtree
from torch.autograd import Variable
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import time
import copy

def train_cnn(method, architecture, alias, dataset, pretrain, validation, num_epochs, learning_rate, weight_decay, use_model):
    
    torch.manual_seed(42)
    
    batch_size = 128
    reward = 2.2

    if dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 100

        
    if not use_model:
        try:
            rmtree(os.path.join('results',alias))
        except BaseException:
            pass  # directory doesn't yet exist, no need to clear it
        os.makedirs(os.path.join('results',alias))
        
        with open(os.path.join('results', alias, 'log_train'), 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(["METHOD : ", method])
            logwriter.writerow(["DATASET : ", dataset])
            logwriter.writerow(["ARCHITECTURE : ", architecture])
            logwriter.writerow(["NUM_EPOCHS : ", num_epochs])
            logwriter.writerow(["LEARNING_RATE : ", learning_rate])
            logwriter.writerow(["WEIGHT_DECAY : ", weight_decay])
            logwriter.writerow(["PRETRAIN : ", pretrain])
            logwriter.writerow(["VALIDATION : ", validation])

    
    if use_model:
        model, starting_epoch, best_acc = load_model(use_model)
        
        if starting_epoch >= num_epochs:
            print("ALREADY TRAINED UP TO NUM_EPOCHS")
            return

    else:
        starting_epoch = 0
        model = construct_model(architecture, method, num_classes)
        best_acc = 0.0
        

    # put model on GPU
    #if not use_model:
    #    model = nn.DataParallel(model)
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)

    dataloaders = construct_dataloaders(dataset, batch_size, validation)

    # define criterion, optimizer for training
    try:
        criterion
    except NameError:
        if method == "deep_gamblers":
            criterion = DeepGamblerLoss(reward)
        elif method == "single_shot":
            criterion = SingleShotLoss()
        elif method == "selectivenet":
            criterion = SelectiveNetLoss()
        else:
            criterion = nn.CrossEntropyLoss() #TODO: Implement mixup as a separate loss
        
    param_iter = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.SGD(
        param_iter,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay)
    
    #optimizer = optim.Adam(
    #    param_iter,
    #    lr=LR,
    #    betas=(0.9, 0.999),
    #    weight_decay=WEIGHT_DECAY)
    
    #if "mixup" in method:
    #    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.5)
    #else:
    #    lambda1 = lambda epoch: 0.5 ** (epoch // 25)
    #    #lambda1 = lambda epoch: 1
    #    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1, last_epoch=-1)
    
    lambda1 = lambda epoch: 0.5 ** (epoch // 25)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1, last_epoch=-1) #This line is necessary to initialize the scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1, last_epoch=starting_epoch)


    # train model
    since = time.time()

    if pretrain:
        model = pretrain_model(architecture, model, method, dataloaders, optimizer, scheduler, alias=alias)
    
    model, best_epoch = train_model(model, method,
                                    dataloaders, criterion, optimizer, scheduler,
                                    starting_epoch, num_epochs, alias=alias,
                                    return_best_model=False,
                                    best_acc=best_acc)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    


def train_model(model, method, dataloaders, criterion, optimizer, scheduler, starting_epoch, num_epochs, alias,
                return_best_model=False, best_acc=0.0):
    #num_epochs = num_epochs + starting_epoch
    best_epoch = -1
    last_train_loss = -1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset_sizes = {x: len(dataloaders[x]) * dataloaders[x].batch_size for x in ['train', 'val']}
    
    # iterate over epochs
    for epoch in range(starting_epoch + 1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            accMeter = AverageMeter()
            lossMeter = AverageMeter()

            # iterate over all data in train/val dataloader:
            i = 0
            for data in dataloaders[phase]:
                i += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = inputs.shape[0]
                
                if "mixup" in method:
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.1)
                    inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()
                    if "mixup" in method:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    
                    elif method == "single_shot":
                        model_new = copy.deepcopy(model)
                        model_new.train(False)
                        alphas = get_alphas_for_single_shot(model_new, inputs)
                        #alphas = 0
                        loss = criterion(outputs, labels, torch.cuda.FloatTensor(alphas)) #TODO: change cuda with device
                        
                    else:
                        loss = criterion(outputs, labels)
                        
                        if loss!=loss:
                            import pdb
                            pdb.set_trace()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                predictions = torch.argmax(outputs, dim=1)
                if "mixup" in method:  
                    accuracy = torch.mean(((lam * (predictions == labels_a) + (1-lam) * (predictions == labels_b))).float()).item() * 100
                else:
                    accuracy = torch.mean((predictions == labels).float()).item() * 100
                    
                lossMeter.update(loss.item(), batch_size)
                accMeter.update(accuracy, batch_size)
                    

                sys.stdout.write("\r Progress in the epoch:     %.3f" % (i * batch_size / dataset_sizes[phase] * 100)) #keep track of the progress
                sys.stdout.flush()

            epoch_loss = lossMeter.avg
            epoch_acc = accMeter.avg
            
            if phase == 'train':
                last_train_loss = epoch_loss
                last_train_acc = epoch_acc
                

            # checkpoint model if has best val loss yet
            else:   
                #Create checkpoint in each epoch 
                if True:
                    state = {
                        'architecture': model.architecture,
                        'train_method': method,
                        'num_classes': model.num_classes,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'epoch' : epoch
                    }
                    path = os.path.join('results',alias)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(state, os.path.join(path,'checkpoint'))
                
                #Create checkpoint for the best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    
                    state = {
                        'architecture': model.architecture,
                        'train_method': method,
                        'num_classes': model.num_classes,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'epoch' : epoch
                    }
                    path = os.path.join('results',alias)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(state, os.path.join(path,'best_checkpoint'))
                
                    if return_best_model:
                        best_model = copy.deepcopy(model)
                  
            # log training and validation loss over each epoch
  
                with open(os.path.join('results',alias,'log_train'), 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
                    logwriter.writerow([epoch, last_train_loss, last_train_acc, epoch_loss, epoch_acc])
                    
            print(phase + ' epoch {}:loss {:.4f} acc {:.4f} with data size {}'.format(epoch, epoch_loss, epoch_acc, dataset_sizes[phase]))
            
        scheduler.step()
    
    if return_best_model:
        model = best_model

    return model, best_epoch
    
