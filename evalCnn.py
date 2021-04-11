# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:16:02 2019

@author: Mirac
"""
from imports import *
from lossFunctions import *
import os
import sys
import csv
from shutil import rmtree
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import time
import copy

def eval_cnn(model, dataloader, eval_method, alias):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # set model to eval mode; required for proper predictions given use of batchnorm
    #model.train(method == 'mc_dropout')
    model.train(False)
    
    with torch.no_grad():
            
        if eval_method == "deep_gamblers":
            reservation_list = []
            correct_predictions_list = []
            for data in dataloader:
                inputs, labels = data
                batch_size = inputs.shape[0]
                inputs, labels = inputs.to(device), labels.to(device)
                #Bu alttaki ikisini kullansam ne olur?
                #inputs = Variable(inputs.cuda())
                #labels = Variable(labels.cuda())
                
                outputs = F.softmax(model(inputs), dim=1)
                class_scores, reservation = outputs[:, :-1], outputs[:, -1]
                reservation = reservation.cpu().detach().numpy()
                predictions = torch.argmax(class_scores, dim=1)
                correct_predictions = (predictions == labels).cpu().detach().numpy()
                
                reservation_list.extend(list(reservation))
                correct_predictions_list.extend(list(correct_predictions))
                
            coverages = [1, 0.95, 0.9, 0.85, 0.8, 0.75]
            
            reservation_array = np.array(reservation_list)
            correct_predictions_array = np.array(correct_predictions_list)
            
            with open(os.path.join('results',alias,'log_test'), 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(["coverage", "test_acc", "risk"])
                for coverage in coverages:
                    threshold = np.percentile(reservation_list, 100 * coverage)
                    covered_idx = reservation_array <= threshold
                    test_acc = np.mean(correct_predictions_array[covered_idx]) * 100
                    risk = (1 - np.mean(correct_predictions_array[covered_idx]))
                    logwriter.writerow([coverage, test_acc, risk])
                    
                
        
        elif eval_method == "softmax_response" or "mc_dropout" in eval_method:
            confidence_list = []
            correct_predictions_list = []
            for data in dataloader:
                inputs, labels = data
                batch_size = inputs.shape[0]
                inputs, labels = inputs.to(device), labels.to(device)
                #Bu alttaki ikisini kullansam ne olur?
                #inputs = Variable(inputs.cuda())
                #labels = Variable(labels.cuda())

                
                if "mc_dropout" in eval_method:
                    class_scores = F.softmax(model(inputs), dim=1)
                    model_new = copy.deepcopy(model)
                    #model_new.train(True)
                    model_new.train(False)
                    _, confidence = utils.mc_dropout_confidence(model_new, inputs, p=0.5, T=100)
                    model_new.train(False)
                    predictions = torch.argmax(class_scores, dim=1)
                    correct_predictions = (predictions == labels).cpu().detach().numpy()
            
                elif "softmax_response" in eval_method:
                    class_scores = F.softmax(model(inputs), dim=1)
                    predictions = torch.argmax(class_scores, dim=1)
                    correct_predictions = (predictions == labels).cpu().detach().numpy()
                    confidence = np.max(class_scores.cpu().detach().numpy(), axis=1)
                
                correct_predictions_list.extend(list(correct_predictions))
                confidence_list.extend(list(confidence))
                
                
            coverages = [1, 0.95, 0.9, 0.85, 0.8, 0.75]
            
            confidence_array = np.array(confidence_list)
            correct_predictions_array = np.array(correct_predictions_list)
            
            with open(os.path.join('results',alias,'log_test'), 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(["method : ", eval_method])
                logwriter.writerow(["coverage", "test_acc", "risk"])
                for coverage in coverages:
                    threshold = np.percentile(confidence_list, 100 * (1 - coverage))
                    covered_idx = confidence_array >= threshold
                    test_acc = np.mean(correct_predictions_array[covered_idx]) * 100
                    risk = (1 - np.mean(correct_predictions_array[covered_idx]))
                    logwriter.writerow([coverage, test_acc, risk])
    
    

    