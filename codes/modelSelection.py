from __future__ import print_function
from __future__ import division
import sklearn
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageOps
import os
import copy
#print(PyTorch Version ,torch.__version__)
#print(Torchvision Version ,torchvision.__version__)
from torch.utils.data import Subset
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch.nn.functional as F
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import GuidedBackprop
from captum.attr import GuidedGradCam
from captum.attr import DeepLift
from captum.attr import Lime
from captum.attr import Saliency
from captum.attr import visualization as viz

model_names = ['resnet18', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
root_path = datadrivedatasetKahikatea
data_dir = root_path + data
#data_dir = DCUB_200_2011CUB_200_2011data
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = model_names[1]

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
num_epochs = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

save_to_path =  root_path + models
#save_to_path = DCUB_200_2011CUB_200_2011models

 # Detect if we have a GPU available
device = torch.device(cuda0 if torch.cuda.is_available() else cpu)

print(device)

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == resnet18
         Resnet18
        
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == resnet50
         Resnet50
        
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == alexnet
         Alexnet
        
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == vgg
         VGG11_bn
        
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == squeezenet
         Squeezenet
        
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == densenet
         Densenet
        
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == inception
         Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else
        print(Invalid model name, exiting...)
        exit()

    return model_ft, input_size
    
    
def getExplainer(model, layer=None, method='guidedGradCAM', model_name=None)
    explainer = None
    
    if method == 'integratedGradient'
        explainer = IntegratedGradients(model)
    
    elif method == 'gradientShap'
        explainer = GradientShap(model)
    elif method == 'saliency'
        explainer = Saliency(model)
    elif method == 'occlusion'
        explainer = Occlusion(model)
        
    elif method == 'guidedBackProp'
        explainer = GuidedBackprop(model)
        
    elif method == 'deepLift'
        explainer = DeepLift(model)
        
    elif method =='lime'
        explainer = Lime(model)
    
    elif method == 'guidedGradCAM'
        if layer == None
            if model_name == 'resnet18'
                layer = model.layer4
            elif model_name == 'resnet50'
                layer = model.layer4
            elif model_name == 'alexnet'
                layer = model.features[10]
            elif model_name == 'vgg'
                layer = model.features[25]
            elif model_name == 'squeezenet'
                layer = model.features[12]
            elif model_name == 'densenet'
                layer = model.features.denseblock4.denselayer16
            elif model_name == 'inception'
                layer = model.Mixed_7c 
            else
                layer = [layer for name, layer in model.named_modules()][-3]
        explainer = GuidedGradCam(model, layer)  
    else
        pass
        
    return explainer


def getAttributions(explainer, input = None, target = None, method=None)
    attributions = None
    if method == 'integratedGradient'
        attributions = explainer.attribute(input, target=target, n_steps=200)
    elif method == 'lime'
        attributions = explainer.attribute(input, target=target, n_perturb_samples=200)
    elif method == 'gradientShap'
        torch.manual_seed(0)
        np.random.seed(0)
        # Defining baseline distribution of images
        rand_img_dist = torch.cat([input  0, input  1])

        attributions = explainer.attribute(input,
                                          n_samples=50,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=target)
    elif method == 'occlusion'
        attributions = explainer.attribute(input,
                                       strides = (3, 8, 8),
                                       target=target,
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)
    else
        attributions = explainer.attribute(input, target)
    
    return attributions
def visualizeAttr(attributions, img, method=['heat_map'], sign=['positive'], show=False)
    return viz.visualize_image_attr(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method=method,
                             show_colorbar=True,
                             sign=sign,
                             outlier_perc=1,
                             use_pyplot=show)
                             
transform = transforms.Compose([
                 transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor()
                ])
transform_normalize = transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )

def load_img(file)
    
    img = Image.open(file).convert('RGB')

    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)
    return input, transformed_img
    
def explain(model_name, model, image_path, explainer_method, expl_form)
    
    #print(loading image...)
    input, transformed_img = load_img(image_path)
    input= input.to(device)
    
    model = model.eval()
    
    #idx_to_labels = {'0'  'negative', '1'  'postive'}
    #print(idx_to_labels)
    
    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    #predicted_label = idx_to_labels[str(pred_label_idx.item())]
    #print('Predicted', predicted_label, '(', prediction_score.squeeze().item(), ')')


    explainer = getExplainer(model, method=explainer_method, model_name=model_name)
    attributions = getAttributions(explainer, input, target=pred_label_idx.item(), method=explainer_method)
       
    return attributions
    
#expl and true_expl are of the same size
from sklearn import metrics
def eval_expl_auc(expl, true_expl)
    e = expl.flatten()
    t = true_expl.flatten()
    return metrics.roc_auc_score(t, e)

def eval_expl_precision(expl, true_expl, threshold=0.01)
    e = expl.flatten()
    t = true_expl.flatten()
    e[ethreshold] = 1
    tp = np.nansum(et)
    
    return tpnp.nansum(e)

def eval_expl_recall(expl, true_expl, threshold=0.01)
    e = expl.flatten()
    t = true_expl.flatten()
    e[ethreshold] = 1
    tp = np.nansum(et)
    
    return tpnp.nansum(t)
    
from os import path
import random
random.seed(0)
def evaluate_expl(model_name, model, dataloaders, explainer='guidedGradCAM', saved_model_name = None, expl_ratio=1)
    aucs = {'train'[], 'val'[], 'test'[]}
    precisions = {'train'[], 'val'[], 'test'[]}
    recalls = {'train'[], 'val'[], 'test'[]}
    f1s = {'train'[], 'val'[], 'test'[]}
    #model_path = data_dir + 'models' + 'resnet18_1.pth'
    
    
    for phase in ['val','test']
        idx = dataloaders[phase].dataset.indices
        samples = dataloaders[phase].dataset.dataset.samples
        for ind in idx
            #if random.random()=expl_ratio
            #    continue
            img_path = samples[ind][0]
            mask_path = img_path.replace('datave','explve')
            if not path.exists(mask_path)
                continue
            
            
            input, image = load_img(img_path)
        
        
            expl = explain(model_name, model, img_path, explainer, 'heat_map')
            tmp = np.transpose(expl.squeeze().cpu().detach().numpy(), (1,2,0))
            
            ex = viz._normalize_image_attr(tmp, 'positive', 2)
            
            ex = np.abs(ex)
        
            
            mask = Image.open(mask_path)
            mask = ImageOps.grayscale(mask)
            #mask.save(data_dir+'tempmask'+y.split('')[1]+'.jpg')
            
            mask = transform(mask)
            
            
            mask = mask[0].numpy()
                 
            
            mask = 1 - mask
            mask[mask  0] = 1
            #plt.imsave(data_dir+'tempmask'+y.split('')[1]+'.jpg', mask)
            #plt.imsave(data_dir+'tempex'+y.split('')[1]+'.jpg', ex)
            
            
            
            #handling extreme cases
            if np.sum(mask) == 0
                mask[0][0] = 1
            if np.sum(mask) = 224224
                mask[0][0] = 0
                
            if np.sum(ex) == 0
                ex[0][0] = 1
            if np.sum(ex) = 224224
                ex[0][0] = 0
                
            auc = eval_expl_auc(ex, mask)
            #pre = eval_expl_precision(ex, mask)
            #recall = eval_expl_recall(ex, mask)
            #f1 = 2  pre  recall  (recall + pre + 1e-10)
            
            
            aucs[phase].append(auc)
            #precisions[sub].append(pre)
            #recalls[sub].append(recall)
            #f1s[sub].append(f1)
            
            ##compute recalls precisions
    avg_auc_train = np.nansum(aucs['train'])len(aucs['train'])
    avg_auc_val = np.nansum(aucs['val'])len(aucs['val'])
    avg_auc_test = np.nansum(aucs['test'])len(aucs['test'])
    
    return avg_auc_train, avg_auc_val, avg_auc_test, aucs['val']
    
    
    
def train_model(model_name, model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False, 
                save_models=False, save_to_path=None, save_to_prefix=None, explainer='guidedGradCAM', expl_ratio=1.0)
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    test_acc_history = []
    
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []
    
    train_auc_history = []
    val_auc_history = []
    test_auc_history = []
    
    ratio_val_auc_history=[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs)
        print('Epoch {}{}'.format(epoch, num_epochs - 1))
        print('-'  10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']
            if phase == 'train'
                model.train()  # Set model to training mode
            else
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train')
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train'
                        # From httpsdiscuss.pytorch.orgthow-to-optimize-inception-model-with-auxiliary-classifiers7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4loss2
                    else
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train'
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()  inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss  len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()  len(dataloaders[phase].dataset)

            print(model_name + ' {} Loss {.4f} Acc {.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc  best_acc
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            #record statistics
            if phase == 'val'
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            elif phase == 'test'
                test_acc_history.append(epoch_acc)
                test_loss_history.append(epoch_loss)
            else
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        
        model.eval()
        avg_auc_train, avg_auc_val, avg_auc_test, aucs_val = evaluate_expl(model_name, model, dataloaders, explainer=explainer)
        train_auc_history.append(avg_auc_train)
        val_auc_history.append(avg_auc_val)
        test_auc_history.append(avg_auc_test)
        
        ratio_val_auc = {}
        for ratio in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
            ratio_val_auc[ratio] = np.sum(random.sample(aucs_val, (int)(ratio  len(aucs_val))))(int)(ratio  len(aucs_val))
        
        ratio_val_auc_history.append(ratio_val_auc)
        print()
        if (save_models)
            torch.save(model, save_to_path + '' + save_to_prefix + '_' + str(epoch) + '.pth')    
         

    time_elapsed = time.time() - since
    print('Training complete in {.0f}m {.0f}s'.format(time_elapsed  60, time_elapsed % 60))
    print('Best val Acc {4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if save_models
        torch.save(model, save_to_path + '' + save_to_prefix + '_' + str(num_epochs) + '.pth')  
    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history, test_acc_history, test_loss_history, train_auc_history, val_auc_history, test_auc_history, ratio_val_auc_history
    
    
def set_parameter_requires_grad(model, feature_extracting)
    if feature_extracting
        for param in model.parameters()
            param.requires_grad = False

from sklearn.model_selection import train_test_split
def train_val_test_dataset(dataset, train_idx, val_idx, test_idx)
    image_datasets = {}
    image_datasets['train'] = Subset(dataset, train_idx)
    image_datasets['val'] = Subset(dataset, val_idx)
    image_datasets['test'] = Subset(dataset, test_idx)
    return image_datasets


def run(model_name, random_state, explainer, train_ratio, val_ratio)
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # Data augmentation and normalization for training
    # Just normalization for validation
    transformer = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    

    print(Initializing Datasets and Dataloaders...)
    
    # Create training and validation dataloaders
    image_dataset = datasets.ImageFolder(data_dir, transformer)
    train_idx, test_idx = sklearn.model_selection.train_test_split(list(range(len(image_dataset))), train_size=train_ratio, random_state = random_state)
    train_idx, val_idx = sklearn.model_selection.train_test_split(train_idx, test_size=val_ratio, random_state = random_state)
    split_datasets = train_val_test_dataset(image_dataset, train_idx, val_idx, test_idx)
    dataloaders_dict = {x torch.utils.data.DataLoader(split_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val', 'test']}

   
    
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.train()

    params_to_update = model_ft.parameters()

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, train_acc_history, train_loss_history, val_acc_history, val_loss_history, test_acc_history, test_loss_history, train_auc_history, val_auc_history, test_auc_history, ratio_val_auc_history = train_model(model_name, model_ft, dataloaders_dict, criterion, 
                    optimizer_ft, num_epochs=num_epochs, is_inception=(model_name==inception), save_models=False,
                            save_to_path = save_to_path, save_to_prefix = model_name, explainer = explainer)
    
    df = pd.DataFrame()
    df['train_acc'] = [h.cpu().numpy().item() for h in train_acc_history]
    df['train_loss'] = train_loss_history
    df['val_acc'] = [h.cpu().numpy().item() for h in val_acc_history]
    df['val_loss'] = val_loss_history
    df['test_acc'] = [h.cpu().numpy().item() for h in test_acc_history]
    df['test_loss'] = test_loss_history
    df['train_auc'] = train_auc_history
    df['val_auc'] = val_auc_history
    df['test_auc'] = test_auc_history
    for ratio in [0,0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1]
        df['val_auc_'+str(ratio)] = [item[ratio] for item in ratio_val_auc_history]
    
    df.to_csv("stats/" + model_name + "_"+explainer+"_"+str(random_state)+'_'+str(train_ratio)+'_'+str(val_ratio)+".csv", index=False)

if __name__ == '__main__'
    explainer = 'guidedBackProp'
    for model_name in  model_names
        for random_state in range(10)
            print(model_name + '-' + str(random_state))
            run(model_name, random_state, explainer, 0.3, 0.33)
