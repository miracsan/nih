# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:15:28 2019

@author: Mirac
"""

import torch
import torch.nn.functional as F
import numpy as np

class SelectiveNetLoss(torch.nn.Module):
    def __init__(self, coverage=0.95, lamda=32, alpha=0.5):
        super(SelectiveNetLoss, self).__init__()
        self.coverage = coverage
        self.lamda = lamda
        self.alpha = alpha
        

    def forward(self, outputs, targets):
        
        classification, auxiliary, selection = outputs[:,:10], outputs[:,10:20], outputs[:,-1];
        selection = torch.sigmoid(selection);
        
        selected_samples = selection > 0.5;
        current_coverage = torch.mean(selected_samples.float());
        
        #selective_risk = torch.mean(selected_samples * F.cross_entropy(classification, targets, reduction='none')) / torch.pow(current_coverage, 2);
        
        #coverage_cost = torch.pow(torch.max(self.coverage - current_coverage, torch.zeros(1, device=targets.device)), 2);
        #risk = F.cross_entropy(auxiliary, targets);
        
        
        #total_loss = self.alpha * (selective_risk + self.lamda * coverage_cost) + (1 - self.alpha) * risk;
        total_loss = F.cross_entropy(classification, targets)
        
        if total_loss != total_loss:
            import pdb
            pdb.set_trace()
        
        return total_loss

class DeepGamblerLoss(torch.nn.Module):
    def __init__(self, reward):
        super(DeepGamblerLoss, self).__init__()
        self.reward = reward
        
    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        class_probs, reservation = probs[:,:-1], probs[:,-1]
        gain = torch.gather(probs, dim=1, index=targets.unsqueeze(1)).squeeze()
        doubling_rate = (gain.add(reservation.div(self.reward))).log()

        return -torch.mean(doubling_rate)
        
        
class SingleShotLoss(torch.nn.Module):
    def __init__(self):
        super(SingleShotLoss, self).__init__()
        
    def forward(self, outputs, targets, alphas):
        targets_onehot = torch.zeros(outputs.shape, device=targets.device);
        targets_onehot.scatter_(1, targets.reshape(-1, 1), 1);
        log_scores = F.log_softmax(outputs, dim=1);
        first_cost_term = torch.sum((1 - alphas) * log_scores * targets_onehot, dim=1);
        second_cost_term = alphas * torch.mean(log_scores, dim=1).reshape(-1, 1);
        loss = -torch.mean(first_cost_term + second_cost_term)
        
        #loss = -torch.mean(torch.sum(log_scores * targets_onehot, dim=1))
        
        #first_cost_term = torch.sum((1 - alphas) * F.cross_entropy(outputs, targets, reduction='none'), dim=1)
        #second_cost_term = 0
        
        
        
        if loss != loss:
            import pdb
            pdb.set_trace()
        
        return loss
        
        
        
