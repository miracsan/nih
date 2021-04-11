#imports
from imports import *
from lossFunctions import *
from trainCnn import train_cnn

import csv

def main():
    METHOD = "baseline"
    ARCHITECTURE = "vgg16_bn_conf_aware_paper"
    #ARCHITECTURE = "resnet50"
    #ALIAS = "mixup_alpha_0.2_resnet"
    ALIAS = "baseline_vgg16_bn_conf_aware_paper"
    DATASET = "cifar10"
    VALIDATION = False
    USE_MODEL = 0
    #USE_MODEL = os.path.join('results', ALIAS,'checkpoint')
    
    print("method : ", METHOD)
    print("alias : ", ALIAS)
    print("validation : ", VALIDATION)
    
    
    NUM_EPOCHS = 300
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 5e-4 
    PRETRAIN = False
                        
    
    train_cnn(METHOD, ARCHITECTURE, ALIAS, DATASET, PRETRAIN, VALIDATION, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, use_model=USE_MODEL)
    
if __name__ == "__main__":
    main()
