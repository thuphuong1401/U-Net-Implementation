import torch.utils.data as data
from torchvision.transforms import ToTensor
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import sys
from torch.autograd import Function

# write your unet implementation here
# when using cross entropy loss, you should have output depth of exactly 2
# sigmoid activation must be applied as the last layer in your network to ensure dice metric works properly (also helps cross entropy loss perform better)
class UNet(nn.Module):
   def __init__(self):
      super(UNet, self).__init__()
      self.sig = nn.Sigmoid()
      #self.input_depth=input_depth
      self.conv1 = nn.Conv2d(1,64,3)
      self.conv11 = nn.Conv2d(64,64,3)
      self.conv2 = nn.Conv2d(64,128,3)
      self.conv22 = nn.Conv2d(128,128,3)
      self.conv3 = nn.Conv2d(128,256,3)
      self.conv33 = nn.Conv2d(256,256,3)
      self.conv4 = nn.Conv2d(256,512,3)
      self.conv44 = nn.Conv2d(512,512,3)
      self.conv5 = nn.Conv2d(512,1024,3)
      self.conv55 = nn.Conv2d(1024,1024,3)
      self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.pad = nn.ZeroPad2d((0,1,0,1))
      self.upconv1 = nn.Conv2d(1024,512,kernel_size=2,stride=1)
      self.conv6 = nn.Conv2d(1024,512,3)
      self.conv66 = nn.Conv2d(512,512,3)
      self.upconv2 = nn.Conv2d(512, 256, kernel_size=2, stride=1)
      self.conv7 = nn.Conv2d(512,256,3)
      self.conv77 = nn.Conv2d(256,256,3)
      self.upconv3 = nn.Conv2d(256, 128, kernel_size=2, stride=1)
      self.conv8 = nn.Conv2d(256,128,3)
      self.conv88 = nn.Conv2d(128,128,3)
      self.upconv4 = nn.Conv2d(128, 64, kernel_size=2, stride=1)
      self.conv9 = nn.Conv2d(128,64,3)
      self.conv99 = nn.Conv2d(64,64,3)
      self.fconv = nn.Conv2d(64,2, 1)


      #self.output_depth = output_depth


   def forward(self, x):
      #x = self.sig(x)
      #x1 = (x)
      x1 = F.relu(self.conv1(x))
      x1 = F.relu(self.conv11(x1))
      x12 = F.max_pool2d(x1,2)
      x2 = F.relu(self.conv2(x12))
      x2 = F.relu(self.conv22(x2))
      x23 = F.max_pool2d(x2,2)
      x3 = F.relu(self.conv3(x23))
      x3 = F.relu(self.conv33(x3))
      x34 = F.max_pool2d(x3,2)
      x4 = F.relu(self.conv4(x34))
      x4 = F.relu(self.conv44(x4))
      x45 = F.max_pool2d(x4, 2)
      x5 = F.relu(self.conv5(x45))
      x5 = F.relu(self.conv55(x5))
      x56 = self.upsample(x5)
      x56 = self.pad(x56)
      x56 = self.upconv1(x56)

      x_4s = np.array(x4.shape)
      print(x_4s)
      x_56s = np.array(x56.shape)
      xpad1 = ((x_56s) - (x_4s))/2
      print(xpad1)

      x_4c = nn.ZeroPad2d(int(xpad1[3]))
      x_4n = x_4c(x4)

      x6 = torch.cat([x56,x_4n],dim=1)
      x6 = F.relu(self.conv6(x6))
      x6 = F.relu(self.conv66(x6))
      x67 = self.upsample(x6)
      x67 = self.pad(x67)
      x67 = self.upconv2(x67)
      x_3s = np.array(x3.shape)
      x_67s = np.array(x67.shape)

      xpad2 = (x_67s-x_3s)/2

      x_3c = nn.ZeroPad2d(int(xpad2[3]))
      x_3n = x_3c(x3)
      x7 = torch.cat([x67,x_3n],dim=1)
      x7 = F.relu(self.conv7(x7))
      x7 = F.relu(self.conv77(x7))
      x78 = self.upsample(x7)
      x78 = self.pad(x78)
      x78 = self.upconv3(x78)
      x_2s = np.array(x2.shape)
      x_78s = np.array(x78.shape)

      xpad3 = (x_78s-x_2s)/2
      x_2c = nn.ZeroPad2d(int(xpad3[3]))
      x_2n = x_2c(x2)
      x8 = torch.cat([x78, x_2n], dim=1)
      x8 = F.relu(self.conv8(x8))
      x8 = F.relu(self.conv88(x8))
      x89 = self.upsample(x8)
      x89 = self.pad(x89)
      x89 = self.upconv4(x89)
      x_1s = np.array(x1.shape)
      x_89s = np.array(x89.shape)

      xpad4 = (x_89s - x_1s) / 2
      x_1c = nn.ZeroPad2d(int(xpad4[3]))
      x_1n = x_1c(x1)
      x9 = torch.cat([x89, x_1n], dim=1)
      x9 = F.relu(self.conv9(x9))
      x9 = F.relu(self.conv99(x9))
      x = self.fconv(x9)
      x = self.sig(x)

      return x

# dice loss implementation - forward pass can also be used to calculate dice metric
class DiceLoss(Function):
   @staticmethod
   def forward(ctx, inp, target):
      ctx.save_for_backward(inp, target)
      rounded = inp.round()
      inter = torch.dot(rounded.view(-1), target.view(-1)) + .000001
      union = torch.sum(rounded) + torch.sum(target) + .000001

      t = 2.0 * inter.float() / union.float()
      return 1.0 - t

   @staticmethod
   def backward(ctx, grad_output):
      inp, target = ctx.saved_variables
      grad_input = grad_target = None
      rounded = inp.round()

      inter = torch.dot(rounded.view(-1), target.view(-1))
      union = torch.sum(rounded) + torch.sum(target)

      grad_input = -2.0 * grad_output * (.000001 + 2.0 * ((target * union) - (2.0 * rounded * inter))) / (union * union + .000001)

      return grad_input, grad_target

# dataset class
# within root folder there should be a folder with the name partition ('train', 'valid', or 'test').  Within that folder there should be a folder of images called "images" and a folder of corresponding segmentation masks labeled "masks"
# images and masks are assumed to already be 388 x 388
# getitem pads the image so that it will be size 572 x 572, no padding is added to label, so it will remain 388 x 388
class SegmentationDataSet(data.Dataset):
   def __init__(self, root_folder, partition):
      self.data = []
      self.labels = []
      self.partition = partition
      self.pad = nn.ZeroPad2d(92)
      self.tens = ToTensor()

      if not root_folder.endswith('/'):
         root_folder = root_folder + '/'

      img_fol = root_folder + partition + '/images/'
      mask_fol = root_folder + partition + '/masks/'

      for img in os.listdir(img_fol):
         im = np.asarray(Image.open(img_fol + img))
         im = np.uint8(im)
         self.data.append(im)

         # use convert mask to 1 channel class labels as required by loss
         mask_im = np.asarray(Image.open(mask_fol + img))
         lbl = np.where(mask_im > 127, 1, 0)
         self.labels.append(lbl)

   def __getitem__(self, index):
      dat = self.data[index]
      lbl = self.labels[index]

      t_dat = self.tens(dat)
      t_dat = self.pad(t_dat)

      t_lbl = torch.tensor(lbl)
      t_lbl = t_lbl.long()

      return t_dat, t_lbl

   def __len__(self):
      return len(self.data)

learn_rate = 0.001
num_epochs = 1
batch_size = 1

loss_func = nn.CrossEntropyLoss()

print('creating loaders')
train_set = SegmentationDataSet('/scratch/dsc381_2019/Homework_5_files/data/', 'train_small')
trainloader = torch.utils.data.DataLoader( dataset = train_set , batch_size= batch_size , shuffle = True)
print('train loader made')

valid_set = SegmentationDataSet('/scratch/dsc381_2019/Homework_5_files/data/', 'valid')
validloader = torch.utils.data.DataLoader( dataset = valid_set , batch_size= batch_size , shuffle = True)
print('valid loader made')

model = UNet()

# Check if CUDA is available
cuda = torch.cuda.is_available()
if cuda:
   # set model to use cuda
   model = model.cuda()
else:
   print("CUDA NOT AVAILABLE")
   sys.exit()

optimizer = optim.Adam(model.parameters(), lr = learn_rate)

print('starting training')

for epoch in range(0, num_epochs):
   model.train()

   epoch_start = datetime.now()

   # track train and validation loss
   epoch_train_loss = 0.0
   epoch_valid_loss = 0.0

   epoch_train_dice = 0.0
   epoch_valid_dice = 0.0

   epoch_train_count = 0.0
   epoch_valid_count = 0.0

   for i, (images, labels) in enumerate(trainloader):
      images = images.cuda()
      labels = labels.cuda()

      # zero out gradients for every batch or they will accumulate
      optimizer.zero_grad()

      # forward step
      outputs = model(images)

      # compute loss
      loss = loss_func(outputs, labels)

      # backwards step
      loss.backward()

      # update weights and biases
      optimizer.step()

      # track training loss and dice metric
      epoch_train_loss += loss.item()
      dice = DiceLoss.apply(outputs[:,1,:,:].float(), labels.float())
      epoch_train_dice += 1.0 - dice.item()
      epoch_train_count += 1.0

   model.eval()
   with torch.no_grad():
      for i, (images, labels) in enumerate(validloader):
         images = images.cuda()
         labels = labels.cuda()

         outputs = model(images)
         loss = loss_func(outputs, labels)

         # track validation loss and dice metric
         epoch_valid_loss += loss.item()
         dice = DiceLoss.apply(outputs[:,1,:,:].float(), labels.float())
         epoch_valid_dice += 1.0 - dice.item()
         epoch_valid_count += 1.0

   epoch_end = datetime.now()
   epoch_time = (epoch_end - epoch_start).total_seconds()

   epoch_train_loss = epoch_train_loss / epoch_train_count
   epoch_valid_loss = epoch_valid_loss / epoch_valid_count
   epoch_train_dice = epoch_train_dice / epoch_train_count
   epoch_valid_dice = epoch_valid_dice / epoch_valid_count

   print("epoch: " + str(epoch) + " - ("+ str(epoch_time) + " seconds)" + "\n\ttrain loss: " + str(epoch_train_loss) + "\n\tvalid loss: " + str(epoch_valid_loss) + "\n\ttrain dice: " + str(epoch_train_dice) + "\n\tvalid dice: " + str(epoch_valid_dice))
