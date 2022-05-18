from django.shortcuts import render, redirect
from django.contrib import messages

import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2

classes = ['FAKE!!', 'REAL!!']

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.keep = 0.5

        self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=8)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=16)
        self.relu2=nn.ReLU()

        self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()

        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv4=nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(num_features=16)
        self.relu4=nn.ReLU()
        

        self.flatten = nn.Flatten()

        self.fc=nn.Linear(in_features=33*33*16,out_features=625, bias=True)
        self.relu5=nn.ReLU()
        self.dout=nn.Dropout(p=1 - self.keep)

        self.outputs = torch.nn.Linear(625, 2, bias=True)
        
                
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool1(output)
            
        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        output=self.pool2(output)

        output=self.conv4(output)
        output=self.bn4(output)
        output=self.relu4(output)

        output=self.flatten(output)
                        
        #output=output.view(-1,32*75*75)
        output=self.fc(output)


        output=self.relu5(output)
        output=self.dout(output)
            
            
        #output=self.fc(output)
        output=self.outputs(output) 

        return output

model = network()
model.load_state_dict(torch.load('/home/kslimon/Projects/Real_fake_face_detection-DFD-/dfd_img/media/mxepmodel.pth', map_location="cpu"))
model.eval()

transformer=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], 
                        [0.5,0.5,0.5])
])


def base(request):
    args={}

    if "btn3" in request.POST:
    # if request.method == "POST":

        image = request.FILES.get('img')
        print(image)

        image = Image.open(image)
        image_tensor=transformer(image).float()

        image_tensor=image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
            image_tensor.cuda()
        
        input=Variable(image_tensor)
    
    
        output=model(input)
    
        index=output.data.numpy().argmax()
        print(index)
    
        pred=classes[index]

        print(pred)

        args['dlog']='Here you go its-'

        args['pred'] = pred

        # return redirect("home")

    return render(request, "base.html", args)

