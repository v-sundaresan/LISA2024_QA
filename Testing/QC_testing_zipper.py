import logging
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from monai.networks.layers import Norm
import monai
from monai.networks.nets import (DenseNet, DenseNet121, DenseNet264)
import torchmetrics
from monai.data import decollate_batch, DataLoader
#from torchvision import models, transforms, datasets
from monai.transforms import (
    AsDiscrete,
    Activations,
    Orientation,
    Compose,
    CropForegroundd,
    SpatialPadd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandWeightedCrop,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandRotated,
    RandRotate90d,
    DivisiblePadd,
    RandFlipd,
    RandZoomd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    RandGaussianSmoothd,
    Spacingd,
    RandAdjustContrastd,
    NormalizeIntensityd,
    SqueezeDimd,
    ToTensord,
    SqueezeDim,
    SaveImaged,
    FillHoles,
    KeepLargestConnectedComponent,
    EnsureChannelFirstd,
    ConcatItemsd,
    Rand3DElasticd,
    Invertd,
)

def extract_filenames(paths):
    filenames = []
    for path in paths:
        filename = os.path.basename(path)  # Get the base filename with extension
        filename_no_ext = os.path.splitext(filename)[0]  # Remove extension
        filenames.append(filename_no_ext)
    return filenames


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    class_name = 'Zipper' #Change this based on different classes
    n_classes = 3
    
    x, y, z = 256, 256, 256 #Adjust this based on available GPU memory

    data_dirte = "/dev/shm/Vaanathi/LISA/Task1_Validation" #SE_HF/SE_LF
    
    images = glob.glob(os.path.join(data_dirte, "*.nii.gz"))
    te_files = [{"img": img} for img in zip(images)]


    te_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            SpatialPadd(keys=["img"],method="symmetric", spatial_size=(x, y, z)),
            ToTensord(keys=["img"]),
        ]
    )

    post_pred = Compose([Activations(softmax=True)])

    # create a validation data loader
    te_ds = monai.data.Dataset(data=te_files, transform=te_transforms)
    te_loader = DataLoader(te_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    model = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1, out_channels=n_classes).to(device)
    #model = MyModel(n_classes).to(device)
    
    model.eval()
    model.load_state_dict(torch.load("./best_metric_model_LISA_LF_" + class_name +".pth"))
    #auc_metric = ROCAUCMetric()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
       
        for te_data in te_loader:
            te_images = te_data["img"].to(device)
            y_pred = torch.cat([y_pred, model(te_images)], dim=0)
            
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        
        # Assuming y_pred_act is a list of tensors
        y_pred_act = [tensor.to('cpu') for tensor in y_pred_act]  # Move to CPU if needed
        y_pred_act_tensor = torch.stack(y_pred_act)  # Stack into a single tensor
        
        # Get the index of the highest probability for each row
        pre_labels_dig = torch.argmax(y_pred_act_tensor, dim=1)
    
    #Write the results to the excell file 
    predicted_labels_list = pre_labels_dig.tolist()
    filenames = extract_filenames(images)
    DF = pd.DataFrame({
    'Subject ID': filenames,
    'Pred_label_' + class_name : predicted_labels_list })
    print()
    DF.to_excel('predicted_output_' + str(class_name) + '.xlsx', index=False)
    print(' *****************The output was generated in predicted_output...csv*****************')
if __name__ == "__main__":
   main()

