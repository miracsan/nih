from evalCnn import *
import utils

import torch

def main():
    EVAL_METHOD = "softmax_response"
    ALIAS = "baseline_vgg16_bn_conf_aware_paper"
    DATASET = "cifar10"
    BATCH_SIZE = 128
    VALIDATION = False

    USE_MODEL = os.path.join('results', ALIAS,'checkpoint')
    
    print("method : ", EVAL_METHOD)
    print("alias : ", ALIAS)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, _ = utils.load_model(USE_MODEL)
    #model.to(device)
    dataloaders = utils.construct_dataloaders(DATASET, BATCH_SIZE, VALIDATION)
    
    
    eval_cnn(model, dataloaders['test'], EVAL_METHOD, ALIAS)
    

if __name__ == "__main__":
    main()
