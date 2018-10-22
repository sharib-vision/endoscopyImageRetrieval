# @Author: Sharib Ali <shariba>
# @Date:   1970-01-01T00:00:00+00:00
# @Email:  sharib.ali@eng.ox.ac.uk
# @Project: BRC: VideoEndoscopy
# @Filename: train_siemese_pytorch_.py
# @Last modified by:   shariba
# @Last modified time: 2018-10-22T14:32:36+01:00
# @Copyright: 2018-2020, sharib ali

import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(3)

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

class ImageFolderWithPaths(dset.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave('traindata.png',np.transpose(npimg, (1, 2, 0)))
    # plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.savefig('convergence_epoch.png')
    # plt.save('convergence_epoch',)
    # plt.show()


# cuda version or non cuda version
cuda=1

class SiameseNetworkDataset(Dataset):

    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

if __name__ == '__main__':

    import argparse

    ''' usage:

        python train_siemese_pytorch_.py --trainingDir ./train/ --validationDir ./test/ \
        --checkpointDir ./checkpoints/ --train_batch_size 64 --epochs 5

    '''

    parser = argparse.ArgumentParser(description='siemese frame retrieval training...')
    parser.add_argument('--trainingDir', type=str, default='./train/', help='enter training data directory')
    parser.add_argument('--validationDir', type=str, default='./test/', help='enter validation data directory')
    parser.add_argument('--checkpointDir', type=str, default='./checkpoints/', help='enter checkpoint directory')
    parser.add_argument('--epochs', type=int, default='100', help='enter number of epochs')
    parser.add_argument('--train_batch_size', type=int, default='1', help='enter training batch size')
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.checkpointDir):
        os.makedirs(args.checkpointDir)

    folder_dataset = ImageFolderWithPaths(args.trainingDir)
    print(folder_dataset)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,transform=transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()]) ,should_invert=False)

    vis_dataloader = DataLoader(siamese_dataset,shuffle=True,num_workers=1,batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())

    train_dataloader = DataLoader(siamese_dataset,shuffle=True,num_workers=1,batch_size=args.train_batch_size)

    if cuda:
        net = SiameseNetwork().cuda()
    else:
        net = SiameseNetwork()

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = []
    iteration_number= 100

    for epoch in range(0,args.epochs):
        #print('training network...')
        t0 = time.time()
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            if cuda:
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            # else:
                # img0, img1 , label = img0, img1 , label
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

        t1 = time.time()
        print("time for epoch {} is {}".format(epoch, t1-t0))
    show_plot(counter,loss_history)

    state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'losslogger': loss_contrastive, }

    # change me:::::::::::!!!!!!
    if cuda:
        torch.save(state, args.checkpointDir+'checkpoint_cuda_augmented.pth')
    else:
        torch.save(state, 'checkpoint.pth')
