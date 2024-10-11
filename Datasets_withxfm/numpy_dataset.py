# Nicola Dinsdale 2020
# Pytorch dataset for numpy arrays
########################################################################################################################
# Import dependencies
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchio as tio
import numpy as np

########################################################################################################################


class numpy_dataset(Dataset):  # Inherit from Dataset class
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            val = torch.rand(1)
            if val > 0.1:   # Augment with probability 0.5
                options = ['rotation', 'blur', 'noise']
                choice = torch.randint(low=0, high=3, size=(1,))
                option = options[choice]
                if option == 'rotation':
                    rot = int(torch.randint(low=-15, high=15, size=(1,)))
                    x = TF.rotate(x, rot)
                    y = TF.rotate(y, rot)
                elif option == 'blur':
                    blurer = transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 0.1))
                    x = blurer(x)
                elif option == 'noise':
                    noise = torch.randn(x.shape)
                    x = x + noise
        return x, y

    def __len__(self):
        return len(self.data)


class NumpyDatasetQC(Dataset):  # Inherit from Dataset class
    def __init__(self, data, target, transform=None):
        self.data1 = torch.from_numpy(data[0]).float()
        self.data2 = torch.from_numpy(data[1]).float()
        self.data3 = torch.from_numpy(data[2]).float()
        self.data4 = torch.from_numpy(data[3]).float()
        self.data5 = torch.from_numpy(data[4]).float()
        self.data6 = torch.from_numpy(data[5]).float()
        self.data7 = torch.from_numpy(data[6]).float()
        self.target1 = torch.from_numpy(target[0]).float()
        self.target2 = torch.from_numpy(target[1]).float()
        self.target3 = torch.from_numpy(target[2]).float()
        self.target4 = torch.from_numpy(target[3]).float()
        self.target5 = torch.from_numpy(target[4]).float()
        self.target6 = torch.from_numpy(target[5]).float()
        self.target7 = torch.from_numpy(target[6]).float()
        self.transform = transform

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[int(index * len(self.data2)) // len(self.data1)]
        x3 = self.data3[int(index * len(self.data3)) // len(self.data1)]
        x4 = self.data4[int(index * len(self.data4)) // len(self.data1)]
        x5 = self.data5[int(index * len(self.data5)) // len(self.data1)]
        x6 = self.data6[int(index * len(self.data6)) // len(self.data1)]
        x7 = self.data7[int(index * len(self.data7)) // len(self.data1)]
        y1 = self.target1[index]
        y2 = self.target2[int(index * len(self.data2)) // len(self.data1)]
        y3 = self.target3[int(index * len(self.data3)) // len(self.data1)]
        y4 = self.target4[int(index * len(self.data4)) // len(self.data1)]
        y5 = self.target5[int(index * len(self.data5)) // len(self.data1)]
        y6 = self.target6[int(index * len(self.data6)) // len(self.data1)]
        y7 = self.target7[int(index * len(self.data7)) // len(self.data1)]

        if self.transform:
            val = torch.rand(1)
            if val > 0.7:   # Augment with probability 0.5
                # options = ['rotation']
                choice1 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
                choice2 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
                choice3 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
                choice4 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
                choice5 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
                choice6 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
                choice7 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
                # option = options[choice]
                motion1 = tio.RandomMotion(3, 3, 2)
                motion2 = tio.RandomMotion(7, 7, 4)
                noise1 = tio.RandomNoise(0, 0.05)
                noise2 = tio.RandomNoise(0, 0.1)
                contrast1 = tio.RandomGamma(0.1)
                contrast2 = tio.RandomGamma(0.3)
                blur = tio.RandomBlur(0.6)
                biasfield1 = tio.RandomBiasField(0.3, 3)
                biasfield2 = tio.RandomBiasField(1, 3)
                positioning1 = tio.RandomAffine(0, 0, (-10, -5, 5, 10, 0, 0), isotropic=False, center='image')
                positioning2 = tio.RandomAffine(0, 0, (-20, -10, 10, 20, 0, 0), isotropic=False, center='image')
                zipper1 = tio.RandomSpike(1, (0.1, 0.2))
                zipper2 = tio.RandomSpike((1, 3), (0.6, 1))
                ghosting = tio.RandomGhosting((2, 4), axes=(0, 1, 2))
                dist1 = tio.RandomElasticDeformation(7, 9, image_interpolation='bspline')
                dist2 = tio.RandomElasticDeformation(12, 12, image_interpolation='bspline')
                # print(x4.size())
                if choice1 == 0:
                    if y1[0] == 1:
                        y1[1] = 0
                        y1[2] = 0
                if choice1 == 1:
                    x1 = noise1(x1)
                    xbrain = x1 > 100
                    x1 = blur(x1 * xbrain)
                    x1 -= torch.min(x1)
                    if y1[1] == 0:
                        y1[1] = 1
                        y1[0] = 0
                        y1[2] = 0
                elif choice1 == 2:
                    x1 = noise2(x1)
                    if y1[2] == 0:
                        y1[2] = 1
                        y1[0] = 0
                        y1[1] = 0
                if choice2 == 0:
                    if y2[0] == 1:
                        y2[1] = 0
                        y2[2] = 0
                if choice2 == 1:
                    xbrain = x2 > 100
                    x2 = zipper1(x2)
                    x2 = blur(x2 * xbrain)
                    x2 -= torch.min(x2)
                    if y2[1] == 0:
                        y2[1] = 1
                        y2[0] = 0
                        y2[2] = 0
                elif choice2 == 2:
                    xbrain = x2 > 100
                    x2 = zipper2(x2)
                    x2 = blur(x2 * xbrain)
                    x2 -= torch.min(x2)
                    if y2[2] == 0:
                        y2[2] = 1
                        y2[0] = 0
                        y2[1] = 0
                if choice3 == 0:
                    if y3[0] == 1:
                        y3[1] = 0
                        y3[2] = 0
                if choice3 == 1:
                    x3 = positioning1(x3)
                    if y3[1] == 0:
                        y3[1] = 1
                        y3[0] = 0
                        y3[2] = 0
                if choice3 == 2:
                    x3 = positioning2(x3)
                    if y3[2] == 0:
                        y3[2] = 1
                        y3[0] = 0
                        y3[1] = 0
                if choice4 == 0:
                    if y4[0] == 1:
                        y4[1] = 0
                        y4[2] = 0
                if choice4 == 1:
                    xbrain = x4 > 100
                    band_start = torch.randint(20, 50, [1])
                    band_end = torch.randint(30, 100, [1])
                    x4[:, band_start:band_end, :, :] = noise1(x4[:, band_start:band_end, :, :])  #Banding
                    x4 = blur(x4 * xbrain)
                    x4 -= torch.min(x4)
                    if y4[1] == 0:
                        y4[1] = 1
                        y4[0] = 0
                        y4[2] = 0
                if choice4 == 2:
                    xbrain = x4 > 100
                    band_start = torch.randint(20, 50, [1])
                    band_end = torch.randint(30, 100, [1])
                    x4[:, band_start:band_end, :, :] = noise2(x4[:, band_start:band_end, :, :])  #Banding
                    x4 = blur(x4 * xbrain)
                    x4 -= torch.min(x4)
                    if y4[2] == 0:
                        y4[2] = 1
                        y4[0] = 0
                        y4[1] = 0
                if choice5 == 0:
                    if y5[0] == 1:
                        y5[1] = 0
                        y5[2] = 0
                if choice5 == 1:
                    x5 = blur(motion1(x5))
                    x5 -= torch.min(x5)
                    if y5[1] == 0:
                        y5[1] = 1
                        y5[0] = 0
                        y5[2] = 0
                if choice5 == 2:
                    x5 = blur(motion2(x5))
                    x5 -= torch.min(x5)
                    if y5[2] == 0:
                        y5[2] = 1
                        y5[0] = 0
                        y5[1] = 0
                if choice6 == 0:
                    if y6[0] == 1:
                        y6[1] = 0
                        y6[2] = 0
                if choice6 == 1:
                    x6 = blur(contrast1(biasfield1(x6)))
                    if y6[1] == 0:
                        y6[1] = 1
                        y6[0] = 0
                        y6[2] = 0
                if choice6 == 2:
                    x6 = blur(contrast2(biasfield2(x6)))
                    if y6[2] == 0:
                        y6[2] = 1
                        y6[0] = 0
                        y6[1] = 0
                if choice7 == 0:
                    if y7[0] == 1:
                        y7[1] = 0
                        y7[2] = 0
                if choice7 == 1:
                    x7 = blur(dist1(x7))
                    if y7[1] == 0:
                        y7[1] = 1
                        y7[0] = 0
                        y7[2] = 0
                if choice7 == 2:
                    x7 = blur(dist2(x7))
                    if y7[2] == 0:
                        y7[2] = 1
                        y7[0] = 0
                        y7[1] = 0

        return x1, x2, x3, x4, x5, x6, x7, y1, y2, y3, y4, y5, y6, y7

    def __len__(self):
        return len(self.data1)


# class NumpyDatasetQCSelAug(Dataset):  # Inherit from Dataset class
#     def __init__(self, data, target, transform=None):
#         self.data1 = torch.from_numpy(data[0]).float()
#         self.data2 = torch.from_numpy(data[1]).float()
#         self.data3 = torch.from_numpy(data[2]).float()
#         self.data4 = torch.from_numpy(data[3]).float()
#         self.data5 = torch.from_numpy(data[4]).float()
#         self.data6 = torch.from_numpy(data[5]).float()
#         self.data7 = torch.from_numpy(data[6]).float()
#         self.target1 = torch.from_numpy(target[0]).float()
#         self.target2 = torch.from_numpy(target[1]).float()
#         self.target3 = torch.from_numpy(target[2]).float()
#         self.target4 = torch.from_numpy(target[3]).float()
#         self.target5 = torch.from_numpy(target[4]).float()
#         self.target6 = torch.from_numpy(target[5]).float()
#         self.target7 = torch.from_numpy(target[6]).float()
#         self.transform = transform
#
#     def __getitem__(self, index):
#         x1 = self.data1[index]
#         x2 = self.data2[int(index * len(self.data2)) // len(self.data1)]
#         x3 = self.data3[int(index * len(self.data3)) // len(self.data1)]
#         x4 = self.data4[int(index * len(self.data4)) // len(self.data1)]
#         x5 = self.data5[int(index * len(self.data5)) // len(self.data1)]
#         x6 = self.data6[int(index * len(self.data6)) // len(self.data1)]
#         x7 = self.data7[int(index * len(self.data7)) // len(self.data1)]
#         y1 = self.target1[index]
#         y2 = self.target2[int(index * len(self.data2)) // len(self.data1)]
#         y3 = self.target3[int(index * len(self.data3)) // len(self.data1)]
#         y4 = self.target4[int(index * len(self.data4)) // len(self.data1)]
#         y5 = self.target5[int(index * len(self.data5)) // len(self.data1)]
#         y6 = self.target6[int(index * len(self.data6)) // len(self.data1)]
#         y7 = self.target7[int(index * len(self.data7)) // len(self.data1)]
#
#         if self.transform:
#             val = torch.rand(1)
#             if val > 0.7:   # Augment with probability 0.5
#                 # options = ['rotation']
#                 # choice1 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
#                 # choice2 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
#                 # choice3 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
#                 # choice4 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
#                 # choice5 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
#                 # choice6 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
#                 # choice7 = np.random.choice(3, 1, p=[0.4, 0.4, 0.2])
#                 # option = options[choice]
#                 motion1 = tio.RandomMotion(3, 3, 2)
#                 motion2 = tio.RandomMotion(7, 7, 4)
#                 noise1 = tio.RandomNoise(0, 0.05)
#                 noise2 = tio.RandomNoise(0, 0.1)
#                 contrast1 = tio.RandomGamma(0.1)
#                 contrast2 = tio.RandomGamma(0.3)
#                 blur = tio.RandomBlur(0.6)
#                 biasfield1 = tio.RandomBiasField(0.3, 3)
#                 biasfield2 = tio.RandomBiasField(1, 3)
#                 positioning1 = tio.RandomAffine(0, 0, (-10, -5, 5, 10, 0, 0), isotropic=False, center='image')
#                 positioning2 = tio.RandomAffine(0, 0, (-20, -10, 10, 20, 0, 0), isotropic=False, center='image')
#                 zipper1 = tio.RandomSpike(1, (0.1, 0.2))
#                 zipper2 = tio.RandomSpike((1, 3), (0.6, 1))
#                 ghosting = tio.RandomGhosting((2, 4), axes=(0, 1, 2))
#                 dist1 = tio.RandomElasticDeformation(7, 9, image_interpolation='bspline')
#                 dist2 = tio.RandomElasticDeformation(12, 12, image_interpolation='bspline')
#                 # print(x4.size())
#                 if choice1 == 0:
#                     if y1[0] == 1:
#                         y1[1] = 0
#                         y1[2] = 0
#                 if choice1 == 1:
#                     x1 = noise1(x1)
#                     xbrain = x1 > 100
#                     x1 = blur(x1 * xbrain)
#                     x1 -= torch.min(x1)
#                     if y1[1] == 0:
#                         y1[1] = 1
#                         y1[0] = 0
#                         y1[2] = 0
#                 elif choice1 == 2:
#                     x1 = noise2(x1)
#                     if y1[2] == 0:
#                         y1[2] = 1
#                         y1[0] = 0
#                         y1[1] = 0
#                 if choice2 == 0:
#                     if y2[0] == 1:
#                         y2[1] = 0
#                         y2[2] = 0
#                 if choice2 == 1:
#                     xbrain = x2 > 100
#                     x2 = zipper1(x2)
#                     x2 = blur(x2 * xbrain)
#                     x2 -= torch.min(x2)
#                     if y2[1] == 0:
#                         y2[1] = 1
#                         y2[0] = 0
#                         y2[2] = 0
#                 elif choice2 == 2:
#                     xbrain = x2 > 100
#                     x2 = zipper2(x2)
#                     x2 = blur(x2 * xbrain)
#                     x2 -= torch.min(x2)
#                     if y2[2] == 0:
#                         y2[2] = 1
#                         y2[0] = 0
#                         y2[1] = 0
#                 if choice3 == 0:
#                     if y3[0] == 1:
#                         y3[1] = 0
#                         y3[2] = 0
#                 if choice3 == 1:
#                     x3 = positioning1(x3)
#                     if y3[1] == 0:
#                         y3[1] = 1
#                         y3[0] = 0
#                         y3[2] = 0
#                 if choice3 == 2:
#                     x3 = positioning2(x3)
#                     if y3[2] == 0:
#                         y3[2] = 1
#                         y3[0] = 0
#                         y3[1] = 0
#                 if choice4 == 0:
#                     if y4[0] == 1:
#                         y4[1] = 0
#                         y4[2] = 0
#                 if choice4 == 1:
#                     xbrain = x4 > 100
#                     band_start = torch.randint(20, 50, [1])
#                     band_end = torch.randint(30, 100, [1])
#                     x4[:, band_start:band_end, :, :] = noise1(x4[:, band_start:band_end, :, :])  #Banding
#                     x4 = blur(x4 * xbrain)
#                     x4 -= torch.min(x4)
#                     if y4[1] == 0:
#                         y4[1] = 1
#                         y4[0] = 0
#                         y4[2] = 0
#                 if choice4 == 2:
#                     xbrain = x4 > 100
#                     band_start = torch.randint(20, 50, [1])
#                     band_end = torch.randint(30, 100, [1])
#                     x4[:, band_start:band_end, :, :] = noise2(x4[:, band_start:band_end, :, :])  #Banding
#                     x4 = blur(x4 * xbrain)
#                     x4 -= torch.min(x4)
#                     if y4[2] == 0:
#                         y4[2] = 1
#                         y4[0] = 0
#                         y4[1] = 0
#                 if choice5 == 0:
#                     if y5[0] == 1:
#                         y5[1] = 0
#                         y5[2] = 0
#                 if choice5 == 1:
#                     x5 = blur(motion1(x5))
#                     x5 -= torch.min(x5)
#                     if y5[1] == 0:
#                         y5[1] = 1
#                         y5[0] = 0
#                         y5[2] = 0
#                 if choice5 == 2:
#                     x5 = blur(motion2(x5))
#                     x5 -= torch.min(x5)
#                     if y5[2] == 0:
#                         y5[2] = 1
#                         y5[0] = 0
#                         y5[1] = 0
#                 if choice6 == 0:
#                     if y6[0] == 1:
#                         y6[1] = 0
#                         y6[2] = 0
#                 if choice6 == 1:
#                     x6 = blur(contrast1(biasfield1(x6)))
#                     if y6[1] == 0:
#                         y6[1] = 1
#                         y6[0] = 0
#                         y6[2] = 0
#                 if choice6 == 2:
#                     x6 = blur(contrast2(biasfield2(x6)))
#                     if y6[2] == 0:
#                         y6[2] = 1
#                         y6[0] = 0
#                         y6[1] = 0
#                 if choice7 == 0:
#                     if y7[0] == 1:
#                         y7[1] = 0
#                         y7[2] = 0
#                 if choice7 == 1:
#                     x7 = blur(dist1(x7))
#                     if y7[1] == 0:
#                         y7[1] = 1
#                         y7[0] = 0
#                         y7[2] = 0
#                 if choice7 == 2:
#                     x7 = blur(dist2(x7))
#                     if y7[2] == 0:
#                         y7[2] = 1
#                         y7[0] = 0
#                         y7[0] = 0
#                         y7[1] = 0
#
#         return x1, x2, x3, x4, x5, x6, x7, y1, y2, y3, y4, y5, y6, y7
#
#     def __len__(self):
#         return len(self.data1)


class NumpyDatasetQCTest(Dataset):  # Inherit from Dataset class
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).float()
        self.transform = transform

    def __getitem__(self, index):
        x1 = self.data[index]
        return x1

    def __len__(self):
        return len(self.data)
    



    