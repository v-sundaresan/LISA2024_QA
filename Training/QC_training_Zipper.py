import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.networks.nets import (DenseNet, DenseNet121, DenseNet264)
import torchmetrics
from monai.data import decollate_batch, DataLoader

from monai.transforms import (
    Activations,
    AsDiscrete,
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

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score


def metrics_func(probability_tensors, true_tensor):
    true_classes = [torch.argmax(label) for label in true_tensor]
    probability_classes = [torch.argmax(label) for label in probability_tensors]

    true_combined_tensor = torch.tensor([t.item() for t in true_classes])
    prob_combined_tensor = torch.tensor([t.item() for t in probability_classes])

    # Initialize metric calculators
    # USe the weighted to handle the imbalance labels
    precision_metric = MulticlassPrecision(average='weighted', num_classes=3)
    recall_metric = MulticlassRecall(average='weighted', num_classes=3)
    f1_metric = MulticlassF1Score(average='weighted', num_classes=3)

    precision_metric.update(prob_combined_tensor, true_combined_tensor)
    recall_metric.update(prob_combined_tensor, true_combined_tensor)
    f1_metric.update(prob_combined_tensor, true_combined_tensor)

    recall = recall_metric.compute()
    f1_value = f1_metric.compute()
    precision = precision_metric.compute()

    # Print other metrics, but just return f1-score #Switch to another one, as preferred
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_value:.4f}')

    return f1_value


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    class_name = 'Zipper'
    # Load the CSV file for labels for the data after appearance-based transformations
    DF_tr = pd.read_excel("./training_cohort_" + class_name + ".xlsx")
    DF_val = pd.read_excel("./validation_cohort_" + class_name + ".xlsx")

    n_epoch = 100
    n_classes = 3
    df_tr = DF_tr[['filename', class_name]]
    df_val = DF_val[['filename', class_name]]
    #######################Nominal to numerical########################################
    data_tr = df_tr.to_numpy()
    data_val = df_val.to_numpy()
    # Define features and labels
    label_tr = data_tr.squeeze()
    label_val = data_val.squeeze()

    x, y, z = 256, 256, 256  # Change based on your GPU memory

    data_dir = "/dev/shm/Vaanathi/LISA/LISA_task1_part1"
    # vaanathi@10.24.10.106

    img_tr = []
    label_tr = []

    img_val = []
    label_val = []

    ###for train data
    for i in range(0, data_tr.shape[0]):
        subj_name = df_tr['filename'][i]

        # Check if image exists in HYP01 folder
        img_path = os.path.join(data_dir, subj_name)
        if os.path.exists(img_path):
            img = img_path
        else:
            # If image doesn't exist in HYP01, check in HYP02 folder
            img_path = os.path.join(data_dir, subj_name)
            if os.path.exists(img_path):
                img = img_path
            else:
                # If image doesn't exist in either folder, handle as needed
                print(f"Image not found for subject {subj_name}")
                continue  # Skip this iteration

        img_tr.append(img)
        label_tr.append(df_tr[class_name][i])
        print(subj_name)

    ###for validation data
    for i in range(0, data_val.shape[0]):
        subj_name = df_val['filename'][i]

        # Check if image exists in HYP01 folder
        img_path = os.path.join(data_dir, subj_name)
        if os.path.exists(img_path):
            img = img_path
        else:
            img_path = os.path.join(data_dir, subj_name)
            if os.path.exists(img_path):
                img = img_path
            else:
                print(f"Image not found for subject {subj_name}")
                continue

        img_val.append(img)
        label_val.append(df_val[class_name][i])
        print(subj_name)

        ##Contrast
    train_files = [{"img": img, "label": label} for img, label in zip(img_tr, label_tr)]
    val_files = [{"img": img, "label": label} for img, label in zip(img_val, label_val)]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(
                keys=["img"], nonzero=False, channel_wise=True),
            CenterSpatialCropd(keys=["img"], roi_size=(x, y, z)),
            SpatialPadd(keys=["img"], method="symmetric", spatial_size=(x, y, z)),
            ToTensord(keys=["img"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            SpatialPadd(keys=["img"], method="symmetric", spatial_size=(x, y, z)),
            ToTensord(keys=["img"]),
        ]
    )

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=n_classes)])

    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=0, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0,
                              pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=0, pin_memory=torch.cuda.is_available())

    model = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1, out_channels=n_classes).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()

    epoch_loss_values = []
    metric_values = []

    for epoch in range(n_epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{n_epoch}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

                metric_result = metrics_func(y_pred_act, y_onehot)

                metric_values.append(metric_result.item())

                del y_pred_act, y_onehot
                if metric_result >= best_metric:
                    best_metric = metric_result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_LISA_LF_" + class_name + ".pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current F1-score: {:.4f} best F1-score: {:.4f} at epoch {}".format(
                        epoch + 1, metric_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_auc", metric_result, epoch + 1)

        np.save("./loss_tr_" + class_name + ".npy", epoch_loss_values)
        np.save("./val_mean_" + class_name + ".npy", metric_values)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
