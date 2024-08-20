import os
from PIL import Image
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import librosa
import scipy
import pywt
from random import shuffle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import glob, os, time, copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import torch.optim as optim
import gc

technique="Non_Seperable_Kernel"

norm_dict = {
'normalize_torch' : transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
),
'normalize_05' : transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
}


import os
import random
import glob
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ReadDatasetImages(Dataset):
    def __init__(self, folder_name, normalize_fun, img_size, val=False):
        folder_name = os.path.join(folder_name, "k1")

        labels = sorted(['abvag', 'novag'])
        num_label = np.arange(len(labels))

        d_val = []
        l_val = []
        d_train = []
        l_train = []

        for index, name in zip(num_label, labels):
            img_list = glob.glob(os.path.join(folder_name, name, name + '*'))
            random.shuffle(img_list)
            for i, img_path in enumerate(img_list):
                if (i % 3 == 0):
                    d_val.append(img_path)
                    l_val.append(index)
                else:
                    d_train.append(img_path)
                    l_train.append(index)

        if val:
            self.data = d_val
            self.labels = l_val
        else:
            self.data = d_train
            self.labels = l_train

        self.transform = transforms.Compose([

            transforms.Resize((img_size,img_size), interpolation=2),
            transforms.ToTensor(),
            norm_dict[normalize_fun]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        IMG=Image.open(img_path)
        arr = np.array(IMG)
        arr = np.uint8(arr*255)
        image = self.transform(Image.fromarray(arr))


        return image, label



from functools import partial

from torch import nn
import torchvision.models as M
import pretrainedmodels

resnet18 = M.resnet18
resnet34 = M.resnet34
resnet50 = M.resnet50
resnet101 = M.resnet101
resnet152 = M.resnet152
vgg16 = M.vgg16
vgg16_bn = M.vgg16_bn
densenet121 = M.densenet121
densenet161 = M.densenet161
densenet201 = M.densenet201


class ResNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super(ResNetFinetune, self).__init__()
        self.net = net_cls(pretrained=True)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.densenet121):
        super(DenseNetFinetune, self).__init__()
        self.net = net_cls(pretrained=True)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


class InceptionV3Finetune(nn.Module):
    finetune = True

    def __init__(self, num_classes):
        super(InceptionV3Finetune, self).__init__()
        self.net = M.inception_v3(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        if self.net.training:
            x, _aux_logits = self.net(x)
            return x
        else:
            return self.net(x)


class FinetunePretrainedmodels(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls, net_kwards):
        super(FinetunePretrainedmodels, self).__init__()
        self.net = net_cls(**net_kwards)
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


resnet18_finetune = partial(ResNetFinetune, net_cls=M.resnet18)
resnet34_finetune = partial(ResNetFinetune, net_cls=M.resnet34)
resnet50_finetune = partial(ResNetFinetune, net_cls=M.resnet50)
resnet101_finetune = partial(ResNetFinetune, net_cls=M.resnet101)
resnet152_finetune = partial(ResNetFinetune, net_cls=M.resnet152)

densenet121_finetune = partial(DenseNetFinetune, net_cls=M.densenet121)

densenet161_finetune = partial(DenseNetFinetune, net_cls=M.densenet161)
densenet201_finetune = partial(DenseNetFinetune, net_cls=M.densenet201)

xception_finetune = partial(FinetunePretrainedmodels,
                            net_cls=pretrainedmodels.xception,
                            net_kwards={'pretrained': 'imagenet'})

inceptionv4_finetune = partial(FinetunePretrainedmodels,
                               net_cls=pretrainedmodels.inceptionv4,
                               net_kwards={'pretrained': 'imagenet+background', 'num_classes': 1001})

inceptionresnetv2_finetune = partial(FinetunePretrainedmodels,
                                     net_cls=pretrainedmodels.inceptionresnetv2,
                                     net_kwards={'pretrained': 'imagenet+background', 'num_classes': 1001})

nasnetmobile_finetune = partial(FinetunePretrainedmodels,
                                net_cls=pretrainedmodels.nasnetamobile,
                                net_kwards={'pretrained': 'imagenet', 'num_classes': 1000})

nasnet_finetune = partial(FinetunePretrainedmodels,
                          net_cls=pretrainedmodels.nasnetalarge,
                          net_kwards={'pretrained': 'imagenet', 'num_classes': 1000})




BATCH_SIZE = 16
IMAGE_SIZE = 224
NB_CLASSES = 2
print('Classes:{}'.format(NB_CLASSES))
folder = './weights/'

MODELS=[resnet18_finetune,
resnet34_finetune,
resnet50_finetune,
resnet101_finetune,
resnet152_finetune,
densenet121_finetune,
densenet161_finetune,
densenet201_finetune,
xception_finetune,
inceptionv4_finetune,
inceptionresnetv2_finetune,
nasnetmobile_finetune,
nasnet_finetune]

def get_model():
    print('[+] loading model... ')
    model=MODELS[2](NB_CLASSES)
    # if use_gpu:
    #     model.cuda()
    print('done')
    return model




# val_dataset = Read_dataset(folder_name='./dataset/',normalize_fun='normalize_torch', img_size=IMAGE_SIZE,val=True)
# validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
#                                     batch_size=BATCH_SIZE,
#                                     shuffle=False)

# train_dataset = Read_dataset(folder_name='./dataset/',normalize_fun='normalize_torch', img_size=IMAGE_SIZE)
# training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
#                                   batch_size=BATCH_SIZE,
#                                   shuffle=True)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False):
    since = time.time()

    train_acc_history = []
    test_acc_history = []

    train_losses = []
    test_losses = [] 

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            temp = 0
            sum_iter = 0
            # Iterate over data.
            out_pred = []
            out_scores = []
            out_labels = []

            x_true = []  # List to store true input data
            x_preds = []  # List to store predicted input data
            y_true = []  # List to store true labels
            y_preds = []  # List to store predicted labels
            
            for inputs, labels in dataloaders[phase]:
                sum_iter += 4

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(type(labels[0][0]))
                    # print(type(outputs[0][0]))
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'test':
                        x_true.extend(inputs.cpu().numpy())
                        x_preds.extend(outputs.cpu().detach().numpy())
                        y_true.extend(labels.cpu().numpy())
                        y_preds.extend(preds.cpu().numpy())

                    out_pred.extend(list(preds.data.cpu().numpy()))
                    out_scores.extend(list(outputs.data.cpu().numpy()))
                    out_labels.extend(list(labels.data.cpu().numpy()))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                test_losses.append(epoch_loss)
                test_acc_history.append(epoch_acc)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_pred = [out_pred, out_scores, out_labels]
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            if phase == 'test':
                test_acc_history.append(epoch_acc)

        print()

    dic = {'x_true': x_true, 'x_preds': x_preds, 'y_true': y_true, 'y_preds': y_preds}


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, test_acc_history, best_pred,train_losses,test_losses, dic




def plot_loss(train_losses, test_losses, num_epochs):
    plt.figure()
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.show()

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def calculate_metrics(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return roc_auc, f1, recall, precision, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.__str__())

dic={"Model":[],"ROC_Score":[],"F1 Score":[],"Recall":[],"Precision":[],"Accuracy":[]}

val_dataset = ReadDatasetImages(folder_name='./transformed_data/',normalize_fun='normalize_torch', img_size=IMAGE_SIZE,val=True)
validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    
train_dataset = ReadDatasetImages(folder_name='./transformed_data/',normalize_fun='normalize_torch', img_size=IMAGE_SIZE)
training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
dataloader={"train" : training_data_loader,"test" : validation_data_loader}

for i in range(3,len(MODELS)):
    try:
        model_ft = MODELS[i](NB_CLASSES).to(device)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay=0.0001)
        # optimizer_ft = optim.Adam(params_to_update, lr=0.0001)
        
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()    
        
        
        new_model, hist_1, hist_2, outputs,train_losses,test_losses,d = train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs=100)
    
    
        plot_loss(train_losses, test_losses, 100)
        path=f"{technique}/{MODELS[i].keywords['net_cls'].__name__}"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, 'loss_func.png'))
        plt.close()
    
        plot_roc_curve(d["y_true"], d["y_preds"])
        path=f"{technique}/{MODELS[i].keywords['net_cls'].__name__}"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, 'roc_auc_plot.png'))
        plt.close()
        dic={"Model":[],"ROC_Score":[],"F1 Score":[],"Recall":[],"Precision":[],"Accuracy":[]}

        roc_auc, f1, recall, precision, accuracy = calculate_metrics(d["y_true"], d["y_preds"])
        dic["Model"].append(MODELS[i].keywords['net_cls'].__name__)
        dic["ROC_Score"].append(roc_auc)
        dic["F1 Score"].append(f1)
        dic["Recall"].append(recall)
        dic["Precision"].append(precision)
        dic["Accuracy"].append(accuracy)
    
        # After training is done, delete the model
        del new_model
        del hist_1
        del hist_2
        del outputs
        del train_losses
        del test_losses
        del model_ft
        del optimizer_ft
        
        # Optionally trigger garbage collection to reclaim memory
        gc.collect()
        
        csv_path = f"{technique}/metrics.csv"
        if os.path.exists(csv_path):
            # Load existing data from the CSV file
            df = pd.read_csv(csv_path)
        else:
            # Create a new DataFrame if the CSV file doesn't exist
            df = pd.DataFrame()
        
        # Append new data to the DataFrame
        new_data = pd.DataFrame(dic)
        df=pd.concat([df,new_data],axis=0,ignore_index=True)
    
        # Write the DataFrame to the CSV file
        df.to_csv(csv_path, index=False)
        dic={"Model":[],"ROC_Score":[],"F1 Score":[],"Recall":[],"Precision":[],"Accuracy":[]}

    except Exception:
        with open(f"{technique}/{MODELS[i].keywords['net_cls'].__name__}.txt","a") as file:
            file.write(f"{MODELS[i].keywords['net_cls'].__name__} error {e}")
            dic={"Model":[],"ROC_Score":[],"F1 Score":[],"Recall":[],"Precision":[],"Accuracy":[]}

        pass

# df=pd.DataFrame(dic)
# df.to_csv(f"{technique}/metrics.csv")
