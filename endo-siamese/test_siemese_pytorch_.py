# @Author: Sharib Ali <shariba>
# @Date:   1970-01-01T00:00:00+00:00
# @Email:  sharib.ali@eng.ox.ac.uk
# @Project: BRC: VideoEndoscopy
# @Filename: test.py
# @Last modified by:   shariba
# @Last modified time: 2018-10-22T14:32:32+01:00
# @Copyright: 2018-2020, sharib ali

cuda =1

import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import time
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# torch.cuda.empty_cache()

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
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # plt.imsave('test_.png',np.transpose(npimg, (1, 2, 0)))



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
                #keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        path_0= img0_tuple[0]
        path_1= img1_tuple[0]
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), path_0, path_1

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

def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


if __name__ == '__main__':

    import argparse


    parser = argparse.ArgumentParser(description='siemese frame retrieval test...')

    parser.add_argument('--folderTargetImages', type=str, default='./tilesWithInk.txt', help='enter datalist')
    parser.add_argument('--folderToQueryImage', type=str, default='./tilesWithInk.txt', help='enter result_dir')
    parser.add_argument('--result_dir', type=str, default='./tilesWithInk.txt', help='enter result_dir')
    parser.add_argument('--indexQuery', type=int, default='./tilesWithInk.txt', help='enter result_dir')

    args = parser.parse_args()
    RESULT_DIR = args.result_dir
    qNo = args.indexQuery

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if cuda:
        model = SiameseNetwork().cuda()

    else:
        model = SiameseNetwork()

    optimizer=optim.Adam(model.parameters(),lr = 0.0005 )
    losslogger=None
    if cuda:
    #    model, optimizer, start_epoch,losslogger = load_checkpoint(model, optimizer, losslogger, filename='checkpoint_cuda.pth')
        model, optimizer, start_epoch,losslogger = load_checkpoint(model, optimizer, losslogger, filename='checkpoint_cuda_augmented_NBI.pth')
        # if cuda map the model to cudaTensorTOcpu
        # torch.load('checkpoint_cuda.pth', map_location=lambda storage, location: 'cpu')
    else:
        model, optimizer, start_epoch,losslogger = load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth')

    folder_dataset_test = ImageFolderWithPaths(args.folderToQueryImage)
    print('test folder info:', folder_dataset_test)
    siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test, transform=transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()]),should_invert=False)
    test_dataloader = DataLoader(siamese_dataset_test,num_workers=0,batch_size=1,shuffle=False,pin_memory=False)
    print('test data loading compelete..')

    #  train dataset for the retrieved images
    batchSize = 10
    folder_dataset_train = ImageFolderWithPaths(args.folderTargetImages)
    print(folder_dataset_train)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_train, transform=transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()]),should_invert=True)
    train_dataloader = DataLoader(siamese_dataset,num_workers=0,batch_size=batchSize, shuffle=True,pin_memory=True)
    dataiter2 = iter(train_dataloader)
    # print(dataiter2)


    x0,_,_,path_0,_ = next(iter(test_dataloader))
    # print('query filename:', path_0)

    img0 = Image.open(''.join(path_0))
    img1 = img0.resize((256,256))
    img1.save(RESULT_DIR+'queryImage'+str(qNo)+'.png')

    # x0,_,_ = next(dataiter)
    eucDist=[]
    fileNames = []
    fileSorted = []

    textfilename = RESULT_DIR+'retrievedImages_'+str(qNo)+'.txt'
    textfile = open(textfilename, 'a')

    # TODO compute retrival time (keep the index in range)

    sortN_retrieve = 20
    t1 = time.time()
    for i in range(sortN_retrieve):
        # print()
        print('file retrieving index', str(i))
        # _,x1,label2,_,path_1 = next(dataiter2)
        _,x1,_,_,path_1 = next(dataiter2)
        print(path_1)
        # print(label2)
        concatenated = torch.cat((x0,x1),0)

        #net is the model here
        if cuda:
            output1,output2 = model(Variable(x0).cuda(),Variable(x1).cuda())
        else:
            output1,output2 = model(Variable(x0),Variable(x1))

        euclidean_distance = F.pairwise_distance(output1, output2)
        print(euclidean_distance)

        if cuda:
            cudaTensorTOcpu = euclidean_distance.cpu()
            val = cudaTensorTOcpu.detach().numpy()
            print(val)
            value = val[0]
            if batchSize > 1 :
                print('sorting the batch distance values..')
                idx = np.argsort(val)
                path_1=path_1[idx[0]]
                value = val[idx[0]]
            fileNames.append(path_1)
            eucDist.append(value)
        else:
            val = euclidean_distance.detach().numpy()
            eucDist.append(val[0])
            fileNames.append(path_1[0])
            print(val)

    print(eucDist)
    print(''.join(fileNames[0]))
    #  perform sorting
    idx = np.argsort(eucDist)
    print(idx)
    fileSortList = []
    sortRange = sortN_retrieve
    for i in range(sortRange):
        fileSortList.append(fileNames[idx[i]])
        textfile.write('\n')
        textfile.write(''.join(fileSortList[i]))

    t2 = time.time()
    print('time for retrieval using siemese:', t2-t1)
    print(fileSortList)


    ## TODO: show list of first 5-10 images (concatenated) that has been retrieved (sorted)
    if sortN_retrieve>=50:
        w = 100*sortN_retrieve
        mh = 100
    else:
        w=248*sortN_retrieve
        mh = 248
    x = 0
    result = Image.new("RGBA", (w,mh))
    for i in range(sortN_retrieve):
        img0 = Image.open(''.join(fileSortList[i]))
        img1 = img0.resize((mh,mh))
        # img0 = img0.convert("L")
        result.paste(img1, (x,0))
        x+=img1.size[0]
        # np.hstack(img0)
        # img0 = cv2.imread(''.join(fileNames[0]))
        # cv2.imshow('',img0)
        # cv2.waitKey(0)

    result.save(RESULT_DIR+'retrieved_'+str(qNo)+'.png')
    # imshow(torchvision.utils.make_grid(concat))

    textfile.write('\n')
    textfile.close()
