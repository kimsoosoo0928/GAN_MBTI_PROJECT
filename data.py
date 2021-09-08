from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms

# 이미지 시각화 함수 
def imshow(img):
    img = img /2 + 0.5 # unnormalize
    np_img = img.numpy()
    # plt.imshow(np_img)
    plt.imshow(np.transpose(np_img, (1,2,0)))
    
    print(np_img.shape)
    print((np.transpose(np_img,(1,2,0))).shape)

trans = transforms.Compose([transforms.Resize((100,100)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                           ])
trainset = torchvision.datasets.ImageFolder(root = "D:\TEAM_PROJECT\split_processing_data",
                                           transform = trans)

print(trainset.__getitem__(18))

len(trainset)


trainloader = DataLoader(trainset,
                        batch_size = 16,
                        shuffle = False,
                        num_workers = 4)

classes = trainset.classes
print(classes)




