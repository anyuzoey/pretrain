
# coding: utf-8

# In[1]:


from torchvision import models
from functions import *
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from scipy.io import loadmat
from PIL import Image
from os.path import splitext
from os import listdir
import torch.utils.data
from glob import glob

# in order to get reproducable results
seed = 1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

torch.cuda.manual_seed(seed)
def _init_fn(worker_id):
    np.random.seed(int(seed))

import cmapy
from pytorchtools import EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Only GPU 3 is visible to this code
time1 = time.time()

lr = 1e-3
batch_size = 32 #64
isportion = "1pic" #["none", "10per", "1per", "1per12", "1pic"]
modelName = 'deeplabv3_resnet101' #["deeplabv3_resnet101","resnet101", "unet", "deeplab", "hed", "rcf"]
targetData = 'thebe' #["thebe", "bsds", "faultSeg"]
isinlinenorm = False
isClip = False
isExtremOutlier = False
isPretrain = 'bsdsft' # ["none","cocoft", "faultSegft", "bsdsft", "imagenetft"]
isfeature_extract = False
if isinlinenorm:
    isDatasetNorm = False
    isSourceDatasetNorm = False
elif isPretrain == "none":  
    isDatasetNorm = True
    isSourceDatasetNorm = False
else:
    isDatasetNorm = False
    isSourceDatasetNorm = True
if targetData == 'thebe':
    data_folder = "/data/anyu/thebeData"
    data_path = "{}/processedThebe".format(data_folder)
elif targetData == 'bsds':
    data_folder = "/data/anyu/BSDS500/data"
    data_path = "{}".format(data_folder)
elif targetData == 'faultSeg':
    data_folder = "/data/anyu/faultSegData"
    data_path = "{}".format(data_folder)
best_iou_threshold=0.5
epoches = 100
train_iterations = 100 #100, 500, 20
val_iterations = 20 # 20, 50, 20
lr_patience = 3
patience = 10

best_model_fpath = modelName + "_{}_{}_{}_{}".format(targetData,lr_patience,patience,lr)
if targetData != 'bsds':
    if isClip:
        if isExtremOutlier:
            best_model_fpath = best_model_fpath + "_clipEO"
        else:
            best_model_fpath = best_model_fpath + "_clip"
if isDatasetNorm:
    best_model_fpath = best_model_fpath + "_DN"
if isPretrain == "cocoft":
    best_model_fpath = best_model_fpath + "_cocoft"
    if isfeature_extract:
        best_model_fpath = best_model_fpath + "_fe"
elif isPretrain == "imagenetft":
    best_model_fpath = best_model_fpath + "_imagenetft"
elif isPretrain == "faultSegft":
    best_model_fpath = best_model_fpath + "_faultSegft"
elif isPretrain == "bsdsft":
    best_model_fpath = best_model_fpath + "_bsdsft"
if isPretrain != "none" and isSourceDatasetNorm:
    best_model_fpath = best_model_fpath + "_SDN"
    
if isinlinenorm:
    best_model_fpath = best_model_fpath + "_ilnorm"
if isportion != "none":
    best_model_fpath = best_model_fpath + "_" + isportion
best_model_fpath = best_model_fpath + "_{}_{}_crop96_b{}_new".format(train_iterations, val_iterations, batch_size) #_trainaugvalnohvflip
print(best_model_fpath)

if modelName == "deeplabv3_resnet101":
    if isPretrain == "cocoft":
        print("use pretrain cocoft")
        model = models.segmentation.deeplabv3_resnet101(pretrained=True,aux_loss=False)
        pretrain_model_fpath = "coco pretrain"     
    elif isPretrain == "bsdsft":
        print("use pretrain bsdsft")
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=False)
        pretrain_model_fpath = "deeplabv3_resnet101_bsds_3_10_0.001_DN_100_20_crop96_b32_trainaugvalnohvflip"
    elif isPretrain == "faultSegft":
        print("use pretrain faultSegft")
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=False)
        pretrain_model_fpath = "deeplabv3_resnet101_faultSeg_3_10_0.001_DN_100_20_crop96_b32_trainaugvalnohflip"
    elif isPretrain == "none":
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=False)
    elif isPretrain == "imagenetft":
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=True)
        pretrain_model_fpath = "imagenet pretrain"   
elif modelName == "unet":
    from model_zoo.UNET import Unet
    model = Unet()
    print("use model Unet")
    if isPretrain == "bsdsft":
        pretrain_model_fpath = "unet_bsds_3_10_0.01_DN"
    elif isPretrain == "faultSegft":
        pretrain_model_fpath = "unet_faultSeg_3_10_0.01_DN_v2"
elif modelName == "deeplab":
    from model_zoo.DEEPLAB.deeplab import DeepLab
    model = DeepLab(backbone='mobilenet', num_classes=1, output_stride=16)
    print("use model DeepLab")
    if isPretrain == "bsdsft":
        pretrain_model_fpath = "deeplab_bsds_3_10_0.01_DN"
    elif isPretrain == "faultSegft":
        pretrain_model_fpath = "deeplab_faultSeg_3_10_0.01_DN_v2"
elif modelName == "hed":
    from model_zoo.HED import HED
    model = HED()
    print("use model HED")
    if isPretrain == "bsdsft":
        pretrain_model_fpath = "hed_bsds_3_10_0.001_DN"
    elif isPretrain == "faultSegft":
        pretrain_model_fpath = "hed_faultSeg_3_10_0.001_DN_v2"
elif modelName == "rcf":
    from model_zoo.RCF import RCF
    model = RCF()
    print("use model RCF")
    if isPretrain == "bsdsft":
        pretrain_model_fpath = "rcf_bsds_3_10_0.001_DN"
    elif isPretrain == "faultSegft":
        pretrain_model_fpath = "rcf_faultSeg_3_10_0.001_DN_v2"
else:
    print("please select a valid model")
if isPretrain != "none":
    print("pretrain weights are: " + pretrain_model_fpath)


# In[2]:


print(model)


# In[3]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
    set_parameter_requires_grad(model, isfeature_extract)

    if isPretrain == "cocoft":
#         num_ftrs = model.aux_classifier[4].in_channels
#         model.aux_classifier[4] = nn.Conv2d(num_ftrs, 1, kernel_size=(1, 1), stride=(1, 1))
        # Handle the primary net
        num_ftrs = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(num_ftrs, 1, kernel_size=(1, 1), stride=(1, 1))
        print(model)


# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if isPretrain != "none" and isPretrain != "cocoft" and isPretrain != "imagenetft":
    print("load pretrained weights")
    model.load_state_dict(torch.load(pretrain_model_fpath, map_location="cuda:0"))
# Send the model to GPU
model = model.double()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(optimizer)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=lr_patience, verbose=True)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
# if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
#     params_to_update = model.parameters()
#     print("Params to learn:")
#     if isfeature_extract:
#         params_to_update = []
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 params_to_update.append(param)
#                 print("\t",name)
#     else:
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 print("\t",name)



# In[5]:


class thebeDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)
    
    def transform(self, img, mask):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        if isinlinenorm == False:          
            # scale to (0,1) dataset wise, using dataset min, max
            if isClip:
                if isExtremOutlier:  
    #                 print("in clip eo")
                    img = np.clip(img, -0.1924, 0.1939) # to change
                    img = (img+0.1924)/(0.1939+0.1924)
                else:
    #                 print("in clip")
                    img = np.clip(img, -0.1096, 0.1111) # to change
                    img = (img+0.1096)/(0.1096+0.1111)
            else:
    #             print("in noclip")
                img = (img+8.3939)/(9.4332+8.3939)
            # normalize based on dataset mean std
            if isDatasetNorm:
                if isClip:
                    if isExtremOutlier:
    #                     print("in clip eo dn")
                        img = TF.normalize(img, [0.4986,], [0.2230,])
                    else:
    #                     print("in clip dn")
                        img = TF.normalize(img, [0.4985,], [0.2855,]) # to change
                else:
    #                 print("in noclip dn")
                    img = TF.normalize(img, [0.4708,], [0.0079,])
        
#         print(img.shape)
        if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
            imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
            imgrgb[0] = img
            imgrgb[1] = img
            imgrgb[2] = img
        else:
            imgrgb = img
            
        if isSourceDatasetNorm:
            if isPretrain == "cocoft" or isPretrain == "imagenetft":
                imgrgb = TF.normalize(imgrgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            elif isPretrain == "faultSegft":
                if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
                    imgrgb = TF.normalize(imgrgb, [0.4915, 0.4915, 0.4915], [0.0655, 0.0655, 0.0655])
#                     imgrgb = TF.normalize(imgrgb, [0.5005, 0.5005, 0.5005], [0.1205, 0.1205, 0.1205]) # clipEO
                else:
                    imgrgb = TF.normalize(imgrgb, [0.4915,], [0.0655,])
#                     imgrgb = TF.normalize(imgrgb, [0.5005,], [0.1205,])
            elif isPretrain == "bsdsft":
                imgrgb = TF.normalize(imgrgb, [0.4316, 0.4357, 0.3637], [0.2521, 0.2360, 0.2446])
#                 if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
#                     imgrgb = TF.normalize(imgrgb, [0.4256, 0.4256, 0.4256], [0.2327, 0.2327, 0.2327])
#                 else:
#                     imgrgb = TF.normalize(imgrgb, [0.4256,], [0.2327,])  
            else:
                print("please enter a valid source dataset set")
        
        return imgrgb, mask
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask = np.load("{}/{}.npy".format(self.masks_dir,idx))
        img = np.load("{}/{}.npy".format(self.images_dir,idx))
        
#         print(img.shape)

        img, mask = self.transform(img, mask)
    

        return (img, mask)
    

class bsdsDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir, aug):
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.aug = aug

    def __len__(self):
        return len(self.ids)
    
    def transform(self, img, mask, aug):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
#         _,w,h = mask.shape
#         if (w == 481) and (h == 321):
# #             print(img.shape)
#             img = torch.transpose(img,1,2)
#             mask = torch.transpose(mask,1,2)
# #             print(img.shape)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(96, 96))
        img = img[:,i:i+h,j:j+w]
        mask = mask[:,i:i+h,j:j+w]
        
        if aug:
            if random.random() > 0.5:
                img = torch.flip(img,(2,)) 
                mask = torch.flip(mask,(2,))
            if random.random() > 0.5:
                img = torch.flip(img,(1,)) 
                mask = torch.flip(mask,(1,))
            
        
        # normalize
#         img = TF.normalize(img, [0.4256,], [0.2327,])  
        imgrgb = TF.normalize(img, [0.4316, 0.4357, 0.3637], [0.2521, 0.2360, 0.2446])
        
#         if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
#             imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
#             imgrgb[0] = img
#             imgrgb[1] = img
#             imgrgb[2] = img
#         else:
#             imgrgb = img
            
        return imgrgb, mask
        
    
    def __getitem__(self, i):
        idx = self.ids[i]
        augment = self.aug
        mask = np.load("{}/{}.npy".format(self.masks_dir,idx))
        img = Image.open("{}/{}.jpg".format(self.images_dir,idx))#.convert('L')
        
        img, mask = self.transform(img, mask, augment)
        mask = (mask>0.5).float()
        
        return (img, mask)

class faultSegDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir, aug):
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.aug = aug


    def __len__(self):
        return len(self.ids)
    
    def transform(self, img, mask, aug):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(96, 96))
        img = img[:,i:i+h,j:j+w]
        mask = mask[:,i:i+h,j:j+w]
        
        if aug:
#             print("aug")
            if random.random() > 0.5:
                img = torch.flip(img,(2,)) 
                mask = torch.flip(mask,(2,))
#             if random.random() > 0.5:
#                 img = torch.flip(img,(1,)) 
#                 mask = torch.flip(mask,(1,))
#         else:
#             print("noaug")
        # scale to (0,1) dataset wise, using dataset min, max
        if isClip:
            if isExtremOutlier:  
#                 print("in clip eo")
                img = np.clip(img, -4.3624, 4.3543) # to change
                img = (img+4.3624)/(4.3543+4.3624)
            else:
#                 print("in clip")
                img = np.clip(img, -2.4945, 2.4864) # to change
                img = (img+2.4945)/(2.4864+2.4945)
        else:
#             print("in noclip")
            img = (img+7.892)/(8.168+7.892)
        # normalize based on dataset mean std
        if isDatasetNorm:
            if isClip:
                if isExtremOutlier:
#                     print("in clip eo dn")
                    img = TF.normalize(img, [0.5005,], [0.1205,])
                else:
#                     print("in clip dn")
                    img = TF.normalize(img, [0.5008,], [0.2028,]) # to change
            else:
#                 print("in noclip dn")
                img = TF.normalize(img, [0.4915,], [0.0655,])
    
        if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
            imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
            imgrgb[0] = img
            imgrgb[1] = img
            imgrgb[2] = img
        else:
            imgrgb = img
            
        return imgrgb, mask
        
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask = np.fromfile("{}/{}.dat".format(self.masks_dir,idx),dtype=np.single).reshape(128,128,128)
        img = np.fromfile("{}/{}.dat".format(self.images_dir,idx),dtype=np.single).reshape(128,128,128)
        augment = self.aug
        
        axis0or1 = np.random.randint(2,size=1)
        ithslice = np.random.randint(128,size=1)
        
        if axis0or1 == 0:
#             print("axis 0 ")
            mask = mask[ithslice,:,:].squeeze().transpose()
            img = img[ithslice,:,:].squeeze().transpose()
        else:
#             print("axis 1 ")
            mask = mask[:,ithslice,:].squeeze().transpose()
            img = img[:,ithslice,:].squeeze().transpose()

        
        img, mask = self.transform(img, mask, augment)

#         assert img.size() == mask.size(),             f'Image and mask {idx} should be the same size, but are {img.size()} and {mask.size()}'
        
        return (img, mask)


# In[6]:


if targetData == 'thebe':
    seis_path_train = ""
    seis_path_val = ""
    fault_path_train = ""
    fault_path_val = ""
    if isinlinenorm:
        seis_path_train = "train/seismic"
        fault_path_train = "train/annotation"
        seis_path_val = "val/seismic"
        fault_path_val = "val/annotation"
    else:
        if isportion == "10per":
            seis_path_train = "train10per/oriseis"
            fault_path_train = "train10per/annotation"
            seis_path_val = "val10per/oriseis"
            fault_path_val = "val10per/annotation"
        elif isportion == "1per":
            seis_path_train = "train1per/oriseis"
            fault_path_train = "train1per/annotation"
            seis_path_val = "val1per/oriseis"
            fault_path_val = "val1per/annotation"
        elif isportion == "1per12":
            seis_path_train = "train1per_12/oriseis"
            fault_path_train = "train1per_12/annotation"
            seis_path_val = "val1per_12/oriseis"
            fault_path_val = "val1per_12/annotation"
        elif isportion == "1pic":
            seis_path_train = "train1pic/oriseis"
            fault_path_train = "train1pic/annotation"
            seis_path_val = "val1pic/oriseis"
            fault_path_val = "val1pic/annotation"
        else:
            seis_path_train = "train/oriseis"
            fault_path_train = "train/annotation"
            seis_path_val = "val/oriseis"
            fault_path_val = "val/annotation"

        
    faults_dataset_train = thebeDataset(imgs_dir = "{}/{}".format(data_path,seis_path_train), masks_dir= "{}/{}".format(data_path,fault_path_train))
    faults_dataset_val = thebeDataset(imgs_dir = "{}/{}".format(data_path,seis_path_val), masks_dir= "{}/{}".format(data_path, fault_path_val))
    train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                           batch_size=batch_size, 
                                           shuffle=False)
elif targetData == 'bsds':    
    faults_dataset_train = bsdsDataset(imgs_dir = "{}/images/train".format(data_path), masks_dir= "{}/GT_np/train".format(data_path), aug = True)
    faults_dataset_val = bsdsDataset(imgs_dir = "{}/images/val".format(data_path), masks_dir= "{}/GT_np/val".format(data_path), aug = False)
    train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                           batch_size=batch_size, 
                                           shuffle=False)
elif targetData == 'faultSeg':
    faults_dataset_train = faultSegDataset(imgs_dir = "{}/train/seis".format(data_path), masks_dir= "{}/train/fault".format(data_path), aug = True)
    faults_dataset_val = faultSegDataset(imgs_dir = "{}/validation/seis".format(data_path), masks_dir= "{}/validation/fault".format(data_path), aug = False)
    train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                               batch_size=batch_size, 
                                               shuffle=False)


# In[7]:


t_start_train = time.time()
bceloss = nn.BCELoss()
bcelogloss = nn.BCEWithLogitsLoss()
celoss = nn.CrossEntropyLoss()
mean_train_losses = []
mean_val_losses = []
mean_train_accuracies = []
mean_val_accuracies = []
t_start = time.time()
early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0)
for epoch in range(epoches):                  
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    labelled_val_accuracies = []

    model.train()
    train_iter = iter(train_loader)
    for ith_iter in range (train_iterations):
        if targetData == 'faultSeg' or targetData == "bsds" or isportion == "1per": 
            train_iter = iter(train_loader)
        images, masks = train_iter.next()
        
        images = Variable(images.double().to(device=device))
        masks = Variable(masks.double().to(device=device))
         
        outputs = model(images)
        
        loss = torch.zeros(1).cuda()
        y_preds = outputs
        if modelName == "unet" or modelName == "deeplab":
            loss = bceloss(outputs, masks) 
#             loss = mean_cross_entropy_loss_HED(outputs, masks) 
        elif modelName == "hed":
            for o in range(5):
                loss = loss + mean_cross_entropy_loss_HED(outputs[o], masks)
            loss = loss + bceloss(outputs[-1],masks)
            y_preds = outputs[-1]
        elif modelName == "rcf":
            for o in outputs:
                loss = loss + mean_cross_entropy_loss_RCF(o, masks)
            y_preds = outputs[-1]
        elif modelName == "resnet101" or modelName == "deeplabv3_resnet101":
            classifier_outputs = outputs["out"] 
            loss = bcelogloss(classifier_outputs, masks) 
#             classifier_outputs, aux_outputs = outputs["out"], outputs["aux"]      
#             loss = bcelogloss(classifier_outputs, masks) + 0.4*bcelogloss(aux_outputs, masks)
            y_preds = torch.sigmoid(classifier_outputs)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        train_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), masks.squeeze(1).byte())
        train_accuracies.append(train_acc.mean()) 
        if ith_iter%10 == 0:
            print("Epoch {}:[{}/{}]".format(epoch,ith_iter,train_iterations), "Acc: {:.4f}".format(train_acc.mean().item()), "Loss: {:.4f}".format(loss.item()))

    model.eval()
    val_iter = iter(val_loader)
    for ith_iter in range (val_iterations):
        if targetData == 'faultSeg' or targetData == "bsds" or isportion == "1per": 
            val_iter = iter(val_loader)
        images, masks = val_iter.next()
        images = Variable(images.double().to(device=device))
        masks = Variable(masks.double().to(device=device))
        
        outputs = model(images)
        
        loss = torch.zeros(1).cuda()
        y_preds = outputs
        if modelName == "unet" or modelName == "deeplab":
            loss = bceloss(outputs, masks) 
#             loss = mean_cross_entropy_loss_HED(outputs, masks) 
        elif modelName == "hed":
            for o in range(5):
                loss = loss + mean_cross_entropy_loss_HED(outputs[o], masks)
            loss = loss + bceloss(outputs[-1],masks)
            y_preds = outputs[-1]
        elif modelName == "rcf":
            for o in outputs:
                loss = loss + mean_cross_entropy_loss_RCF(o, masks)
            y_preds = outputs[-1]
        elif modelName == "resnet101" or modelName == "deeplabv3_resnet101":
            classifier_outputs = outputs["out"] 
            loss = bcelogloss(classifier_outputs, masks) 
            y_preds = torch.sigmoid(classifier_outputs)
        

        val_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        val_acc = iou_pytorch(predicted_mask.byte(), masks.squeeze(1).byte())
        val_accuracies.append(val_acc.mean())

        
    mean_train_losses.append(torch.mean(torch.stack(train_losses)))
    mean_val_losses.append(torch.mean(torch.stack(val_losses)))
    mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
    mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))
    
    scheduler.step(torch.mean(torch.stack(val_losses)))    
    early_stopping(torch.mean(torch.stack(val_losses)), model, best_model_fpath)
    


    if early_stopping.early_stop:
        print("Early stopping")
        break
        
    
    for param_group in optimizer.param_groups:
        learningRate = param_group['lr']
    
    
    # Print Epoch results
    t_end = time.time()

    print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
          .format(epoch+1, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)), torch.mean(torch.stack(train_accuracies)), torch.mean(torch.stack(val_accuracies)), t_end-t_start, learningRate))
    
    t_start = time.time()
    
total_training_time = time.time()-t_start_train
print("total training time: {:.4f}".format(total_training_time/3600))


# In[ ]:


mean_train_losses = np.asarray(torch.stack(mean_train_losses).cpu())
mean_val_losses = np.asarray(torch.stack(mean_val_losses).cpu())
mean_train_accuracies = np.asarray(torch.stack(mean_train_accuracies).cpu())
mean_val_accuracies = np.asarray(torch.stack(mean_val_accuracies).cpu())

fig = plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
train_loss_series = pd.Series(mean_train_losses)
val_loss_series = pd.Series(mean_val_losses)
train_loss_series.plot(label="train_loss")
val_loss_series.plot(label="validation_loss")
plt.legend()
plt.subplot(1, 2, 2)
train_acc_series = pd.Series(mean_train_accuracies)
val_acc_series = pd.Series(mean_val_accuracies)
train_acc_series.plot(label="train_acc")
val_acc_series.plot(label="validation_acc")
plt.legend()
plt.title('{}_loss_acc'.format(best_model_fpath))
plt.savefig('{}_loss_acc.png'.format(best_model_fpath))

totaltime = time.time()-time1
print("total cost {} hours".format(totaltime/3600))

