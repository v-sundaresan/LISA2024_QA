import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torchio as tio
import monai
import nibabel as nib
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


def generate_simdata(x, y):
    print('y value passed: ', y)
    choice_num = np.random.choice(3, 1)
    print(choice_num)
    artifact_list = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
    output = [0, 0, 0, 0, 0, 0, 0]
    for i in range(0, choice_num[0]):
        choice_level = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
        choice_level = choice_level[0]
        choice_art = np.random.choice(6, 1)
        artifact = artifact_list[choice_art[0]]

        blur = tio.RandomBlur(0.6)
        noise1 = tio.RandomNoise(0, 0.05)
        noise2 = tio.RandomNoise(0, 0.1)
        ghosting = tio.RandomGhosting((2, 4), axes=(0, 1, 2))
        if artifact == "Noise":
            if choice_level == 0:
                output[0] = y[0]
            if choice_level == 1:
                if y[0] < 1:
                    x = (x[np.newaxis, ...])
                    xbrain = x > 400
                    x = noise1(x)
                    x = blur(x * xbrain)
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[0] = 1
                else:
                    output[0] = y[0]
            elif choice_level == 2:
                if y[0] < 2:
                    x = (x[np.newaxis, ...])
                    xbrain = x > 400
                    x = noise2(x)
                    x = blur(x * xbrain)
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[0] = 2
                else:
                    output[0] = y[0]
        elif artifact == "Zipper":
            zipper1 = tio.RandomSpike(1, (0.1, 0.2))
            zipper2 = tio.RandomSpike((1, 3), (0.6, 1))
            if choice_level == 0:
                output[1] = y[1]
            if choice_level == 1:
                if y[1] < 1:
                    xbrain = x > 400
                    xbrain = xbrain[np.newaxis, ...]
                    x = (x[np.newaxis, ...])
                    x = zipper1(x)
                    x = blur(x * xbrain)
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[1] = 1
                else:
                    output[1] = y[1]
            elif choice_level == 2:
                if y[1] < 2:
                    xbrain = x > 400
                    xbrain = xbrain[np.newaxis, ...]
                    x = (x[np.newaxis, ...])
                    x = zipper2(x)
                    x = blur(x * xbrain)
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[1] = 2
                else:
                    output[1] = y[1]
        elif artifact == "Positioning":
            positioning1 = tio.RandomAffine(0, 0, (-10, -5, 5, 10, 0, 0), isotropic=False, center='image')
            positioning2 = tio.RandomAffine(0, 0, (-20, -10, 10, 20, 0, 0), isotropic=False, center='image')
            if choice_level == 0:
                output[2] = y[2]
            if choice_level == 1:
                if y[2] < 1:
                    x = (x[np.newaxis, ...])
                    x = positioning1(x)
                    x = np.squeeze(x)
                    output[2] = 1
                else:
                    output[2] = y[2]
            elif choice_level == 2:
                if y[2] < 2:
                    x = (x[np.newaxis, ...])
                    x = positioning2(x)
                    x = np.squeeze(x)
                    output[2] = 2
                else:
                    output[2] = y[2]
        elif artifact == "Banding":
            if choice_level == 0:
                output[3] = y[3]
            if choice_level == 1:
                if y[3] < 1:
                    xbrain = x > 400
                    xbrain = xbrain[np.newaxis, ...]
                    x = (x[np.newaxis, ...])
                    band_start = torch.randint(20, 50, [1])
                    band_end = torch.randint(30, 100, [1])
                    x[:, band_start:band_end, :, :] = noise1(x[:, band_start:band_end, :, :])  # Banding
                    x = blur(x * xbrain)
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[3] = 1
                else:
                    output[3] = y[3]
            elif choice_level == 2:
                if y[3] < 2:
                    xbrain = x > 400
                    xbrain = xbrain[np.newaxis, ...]
                    x = (x[np.newaxis, ...])
                    band_start = torch.randint(20, 50, [1])
                    band_end = torch.randint(30, 100, [1])
                    x[:, band_start:band_end, :, :] = noise2(x[:, band_start:band_end, :, :])  # Banding
                    x = blur(x * xbrain)
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[3] = 2
                else:
                    output[3] = y[3]
        elif artifact == "Motion":
            motion1 = tio.RandomMotion(3, 3, 2)
            motion2 = tio.RandomMotion(7, 7, 4)
            if choice_level == 0:
                output[4] = y[4]
            if choice_level == 1:
                if y[4] < 1:
                    x = (x[np.newaxis, ...])
                    x = blur(motion1(x))
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[4] = 1
                else:
                    output[4] = y[4]
            elif choice_level == 2:
                if y[4] < 2:
                    x = (x[np.newaxis, ...])
                    x = blur(motion2(x))
                    x -= np.min(x)
                    x = np.squeeze(x)
                    output[4] = 2
                else:
                    output[4] = y[4]
        elif artifact == "Contrast":
            contrast1 = tio.RandomGamma(0.1)
            contrast2 = tio.RandomGamma(0.3)
            biasfield1 = tio.RandomBiasField(0.3, 3)
            biasfield2 = tio.RandomBiasField(1, 3)
            if choice_level == 0:
                print(output)
                print(y)
                output[5] = y[5]
            if choice_level == 1:
                if y[5] < 1:
                    x = (x[np.newaxis, ...])
                    x = blur(contrast1(biasfield1(x)))
                    x = np.squeeze(x)
                    output[5] = 1
                else:
                    output[5] = y[5]
            elif choice_level == 2:
                if y[5] < 2:
                    x = (x[np.newaxis, ...])
                    x = blur(contrast2(biasfield2(x)))
                    x = np.squeeze(x)
                    output[5] = 2
                else:
                    output[5] = y[5]
        elif artifact == "Distortion":
            dist1 = tio.RandomElasticDeformation(7, 9, image_interpolation='bspline')
            dist2 = tio.RandomElasticDeformation(12, 12, image_interpolation='bspline')
            if choice_level == 0:
                output[6] = y[6]
            if choice_level == 1:
                if y[6] < 1:
                    x = blur(dist1(x))
                    output[6] = 1
                else:
                    output[6] = y[6]
            elif choice_level == 2:
                if y[6] < 2:
                    x = blur(dist2(x))
                    output[6] = 2
                else:
                    output[6] = y[6]
    data = [list(output), list(y)]
    print(output)
    print(y)
    out = [max(column) for column in zip(*data)]
    print(out)
    return x, np.array(out)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    class_name = 'Distortion'

    # Load the CSV file for labels
    DF_tr = pd.read_excel("./training_cohort_" + class_name + ".xlsx")
    DF_val = pd.read_excel("./validation_cohort_" + class_name + ".xlsx")

    # n_epoch = 100
    # n_classes = 3
    df_tr = DF_tr[['filename', class_name]]
    df_val = DF_val[['filename', class_name]]
    #######################Nominal to numerical########################################
    data_tr = df_tr.to_numpy()
    data_val = df_val.to_numpy()
    # Define features and labels
    label_tr = data_tr.squeeze()
    label_val = data_val.squeeze()

    x, y, z = 256, 256, 256  # Change based on your GPU memory

    data_dir = "/dev/shm/Vaanathi/LISA/LISA_task1_part1/"

    img_tr = []
    label_tr = []

    img_val = []
    label_val = []
    subj_names = []
    class_name = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
    ###for train data

    for i in range(0, data_tr.shape[0]):
        subj_name = df_tr['filename'][i]
        labels = []
        for j in range(0, len(class_name)):
            labels.append(DF_tr[class_name[j]][i])
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
        label_tr.append(labels)
        subj_names.append(subj_name)

    y_mat = np.array([])
    imgname_arr = []
    for i in range(len(img_tr)):
        aug_prop = 1
        img = nib.load(img_tr[i]).get_fdata()
        hdr = nib.load(img_tr[i]).header
        newhdr = hdr.copy()
        for j in range(aug_prop):
            x, y = generate_simdata(img, label_tr[i])
            xobj = nib.nifti1.Nifti1Image(x, None, header=newhdr)
            nib.save(xobj, img_tr[i][:-7] + '_' + str(j) + '.nii.gz')
            newname = subj_names[i][:-7] + '_' + str(j) + '.nii.gz'
            print(newname)
            imgname_arr.append(newname)
            y_mat = np.vstack([y_mat, y]) if y_mat.size else y
    # imgname_arr = np.reshape(np.array(imgname_arr), [-1, 1])
    # subj_names = np.reshape(np.array(subj_names), [-1, 1])
    # print(imgname_arr)
    # print('************************************************************************************************')
    # print(np.array(subj_names).shape)
    # subj_names = list(np.vstack([subj_names, imgname_arr]))
    subj_names = subj_names + imgname_arr
    print(subj_names)
    label_mat = np.vstack([np.array(label_tr), y_mat])

    # Write the results to the excell file
    # predicted_labels_list = pre_labels_dig.tolist()
    # filenames = extract_filenames(images)
    DF = pd.DataFrame({
        'Subject ID': subj_names,
        'Noise': list(label_mat[:, 0]),
        'Zipper': list(label_mat[:, 1]),
        'Positioning': list(label_mat[:, 2]),
        'Banding': list(label_mat[:, 3]),
        'Motion': list(label_mat[:, 4]),
        'Contrast': list(label_mat[:, 5]),
        'Distortion': list(label_mat[:, 6])
    })
    print()
    DF.to_excel('training_cohort_Noise_Augmented.xlsx', index=False)
    print(' *****************The output was generated in training_cohort_Noise_Augmented...csv*****************')

    # ###for validation data
    # for i in range(0, data_val.shape[0]):
    #     subj_name = df_val['filename'][i]
    #     labels = []
    #     for j in range(0, len(class_name)):
    #         labels.append(df_tr[class_name[j]][i])
    #     # Check if image exists in HYP01 folder
    #     img_path = os.path.join(data_dir, subj_name)
    #     if os.path.exists(img_path):
    #         img = img_path
    #     else:
    #         img_path = os.path.join(data_dir, subj_name)
    #         if os.path.exists(img_path):
    #             img = img_path
    #         else:
    #             print(f"Image not found for subject {subj_name}")
    #             continue
    #
    #     img_val.append(img)
    #     label_val.append(labels)
    #     subj_names.append(subj_name)

if __name__ == "__main__":
    main()
