import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import numpy as np
import os
from customModels import *

def create_checkpoint(model, method, epoch, epoch_loss, epoch_acc, criterion, alias):

    print('saving')
    state = {
        'model': model,
        'method': method,
        'epoch': epoch,
        'epoch_loss': epoch_loss,
        'epoch_acc': epoch_acc,
        'criterion' : criterion
    }
    
    path = os.path.join('results',alias)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(state, os.path.join(path,'checkpoint'))


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
### MIXUP

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    

### MC

def mc_dropout_forward(model, x, p=0.5):
    
    if model.architecture == 'vgg16_bn_dropout':
        x = model.features(x);
        x = x.view(x.size(0), -1);
        x = F.dropout(x, p=p);
        x = model.classifier[:4](x);
        x = model.classifier[4](x);
    
    elif model.architecture == 'resnet50':
        feature_extraction = nn.Sequential(*list(model.children())[:-1]);
        x = feature_extraction(x);
        x = x.view(x.size(0), -1);
        x = F.dropout(x, p=p);
        x = model.fc(x);
        
    return x
    

def mc_dropout_confidence(model, x, p=0.5, T=100):
    repetitions = []
    
    for i in range(T):
        inference = F.softmax(mc_dropout_forward(model, x, p=p), dim=1) 
        repetitions.append(inference.cpu().detach().numpy())

    repetitions = np.array(repetitions)
    mc = np.var(repetitions, 0)
    mc = np.mean(mc, -1)
    average_class_scores = np.mean(repetitions, 0)
    #predictions = np.argmax(average_class_scores, 1)
    
    return average_class_scores, -mc



### SINGLE SHOT

def get_alphas_for_single_shot(model, inputs, T=20):
    with torch.set_grad_enabled(False):
        #initial_state = model.train
        #model.train(False)
        repetitions = []
        for i in range(T):
            inference = F.softmax(mc_dropout_forward(model, inputs), dim=1) 
            repetitions.append(inference.cpu().detach().numpy())
        
        repetitions = np.array(repetitions)
        mean_inference = np.mean(repetitions, 0)
        alphas = np.array([get_normalised_var(repetitions[:,i,:], mean_inference[i]) for i in range(inputs.shape[0])]).reshape(-1,1)
        #model.train(initial_state)
        
        return alphas
    
    
    
    
def get_normalised_var(p, q):
    bhattacharya_coefs = get_bhattacharya_coef(p, q)
    return 1 - np.mean(bhattacharya_coefs)
    
def get_bhattacharya_coef(p, q):
    q = q.reshape(1, -1)
    if p.shape[1] != q.shape[1]:
        raise Exception("Distributions of different length given!")
    
    return np.sum(np.sqrt(p * q), 1)
    
    

### SETTING UP THE TRAINING

def pretrain_model(architecture, model, method, dataloaders, optimizer, scheduler, alias):
    if architecture == "vgg16_bn":
        original_classifier = copy.deepcopy(model.classifier[6])
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_labels)
        model.cuda()
        model, _ = train_model(model, method,
                               dataloaders, nn.CrossEntropyLoss(), optimizer, scheduler,
                               starting_epoch=0, num_epochs=100, alias=alias)
        model.classifier[6] = original_classifier
        model.cuda()
    
    elif architecture == "vgg16_bn_dropout":
        original_classifier = copy.deepcopy(model.classifier[4])
        num_ftrs = model.classifier[4].in_features
        model.classifier[4] = nn.Linear(num_ftrs, n_labels)
        model.cuda()
        model, _ = train_model(model, method,
                               dataloaders, nn.CrossEntropyLoss(), optimizer, scheduler,
                               starting_epoch=0, num_epochs=100, alias=alias)
        model.classifier[4] = original_classifier
        model.cuda()
        
    elif architecture == "densenet121":
        original_classifier = copy.deepcopy(model.classifier);
        num_ftrs = model.classifier.in_features;
        model.classifier = nn.Linear(num_ftrs, n_labels);
        model.cuda();
        model, _ = train_model(model, method,
                               dataloaders, nn.CrossEntropyLoss(), optimizer, scheduler,
                               starting_epoch=0, num_epochs=100, alias=alias)
        model.classifier = original_classifier
        model.cuda()
        
    elif architecture == "resnet50":
        original_classifier = copy.deepcopy(model.fc);
        num_ftrs = model.fc.in_features;
        n_labels = model.fc.out_features
        model.fc = nn.Linear(num_ftrs, n_labels-1);
        model.cuda();
        model, _ = train_model(model, method,
                               dataloaders, nn.CrossEntropyLoss(), optimizer, scheduler,
                               starting_epoch=0, num_epochs=100, alias=alias)
        model.fc = original_classifier
        model.cuda()
    return model
    

def construct_model(architecture, method, num_classes):
    if architecture == "vgg16_bn":
        from vgg import vgg16_bn
        model = models.vgg16_bn(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        if method in ["deep_gamblers"]:
            model.classifier[6] = nn.Linear(num_ftrs, num_classes + 1)
        else:
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "vgg16_bn_conf_aware_paper":
        from model.vgg import vgg16 as  vgg16_bn_conf_aware_paper
        model = vgg16_bn_conf_aware_paper(num_classes=10)
            
    elif architecture == 'vgg16_bn_dropout':
        from vgg import vgg16_bn 
        if method in ["deep_gamblers"]:
            model = vgg16_bn(dropout=True, num_classes=num_classes+1, input_size=32)
        elif method in ["selectivenet"]:
            model = SelectiveVgg16_bn(n_labels)
        else:
            model = vgg16_bn(dropout=True, num_classes=num_classes, input_size=32)
            
    elif architecture == "densenet121":
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        if method in ["deep_gamblers"]:
            model.classifier = nn.Linear(num_ftrs, num_classes + 1)
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)
            
    elif architecture == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        if method in ["deep_gamblers"]:
            model.fc = nn.Linear(num_ftrs, num_classes + 1)
        elif method in ["selectivenet"]:
            model = SelectiveResNet50(num_classes)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Unknown architecture. Aborting...")
        return
    
    model.architecture = architecture
    model.num_classes = num_classes
    
    return model
    

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    architecture = checkpoint['architecture']
    num_classes = checkpoint['num_classes']
    train_method = checkpoint['train_method']
    state_dict = checkpoint['state_dict']
    starting_epoch = checkpoint['epoch'] if 'epoch'in checkpoint else 0
    best_acc = checkpoint['best_acc'] if 'best_acc' in checkpoint else 0.0
    
    model = construct_model(architecture, train_method, num_classes)
    model.load_state_dict(state_dict)
    
    print("Existing model was trained using {0}".format(train_method))
    
    return model, starting_epoch, best_acc
        
    
def construct_dataloaders(dataset, batch_size, validation):
    if dataset == "cifar10":
        transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
        
    else:
        print("Unknown dataset. Aborting...")
        return
        
    dataloaders = {}
    if validation:
        if method == "deep_gamblers":
            val_size = 2000
            train_size = len(train_set) - val_size
            train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])
        else:
            train_size = int(0.8 * len(train_set))
            val_size = len(train_set) - train_size
            train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])
            
        dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
        dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True, num_workers=8)
        dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, num_workers=8)
        
    else:
        dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
        dataloaders['val'] = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=8)
        dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=8)
        
    return dataloaders
