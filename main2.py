#imports
from imports import *
from lossFunctions import *
from trainCnn import train_cnn
from camUtils import *
from tensorboardUtils import *

import csv

METHOD = "selectivenet"
ARCHITECTURE = "vgg16_bn_dropout"
#ARCHITECTURE = "resnet50"
#ALIAS = "mixup_alpha_0.2_resnet"
ALIAS = "selective_vgg16_bn"
DATASET = "cifar10"
VALIDATION = False
EVALUATE = False
#USE_MODEL = torch.load(os.path.join('results', ALIAS,'checkpoint'))
USE_MODEL = 0

print("method : ", METHOD)
print("alias : ", ALIAS)
print("validation : ", VALIDATION)


NUM_EPOCHS = 300
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4 
PRETRAIN = False
                    

train_cnn(METHOD, ARCHITECTURE, ALIAS, DATASET, PRETRAIN, VALIDATION, EVALUATE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, use_model=USE_MODEL)
