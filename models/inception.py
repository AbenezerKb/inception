import torch
import torch.nn as nn

class ConvBlock(nn.Module):
   
    def __init__(self, in_channels , out_channels , kernel_size , stride , padding , bias=False):
        super(ConvBlock,self).__init__()

        self.conv2d = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding , bias=False )

        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))


class InceptionBlock(nn.Module):
    
    def __init__(self , in_channels , out_1x1 , red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1_pooling):
        super(InceptionBlock,self).__init__()

        self.branch1 = ConvBlock(in_channels,out_1x1,1,1,0)

        self.branch2 = nn.Sequential(ConvBlock(in_channels,red_3x3,1,1,0),ConvBlock(red_3x3,out_3x3,3,1,1))

        self.branch3 = nn.Sequential(ConvBlock(in_channels,red_5x5,1,1,0),ConvBlock(red_5x5,out_5x5,5,1,2))

        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),ConvBlock(in_channels,out_1x1_pooling,1,1,0))


    def forward(self,x):

        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],dim=1)


class Inceptionv1(nn.Module):
    
    def __init__(self , in_channels , num_classes ):
        super(Inceptionv1,self).__init__()

        self.conv1 =  ConvBlock(in_channels,64,7,2,3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 =  nn.Sequential(ConvBlock(64,64,1,1,0),ConvBlock(64,192,3,1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception3a = InceptionBlock(192,64,96,128,16,32,32)
        self.inception3b = InceptionBlock(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = InceptionBlock(480,192,96,208,16,48,64)
        self.inception4b = InceptionBlock(512,160,112,224,24,64,64)
        self.inception4c = InceptionBlock(512,128,128,256,24,64,64)
        self.inception4d = InceptionBlock(512,112,144,288,32,64,64)
        self.inception4e = InceptionBlock(528,256,160,320,32,128,128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = InceptionBlock(832,256,160,320,32,128,128)
        self.inception5b = InceptionBlock(832,384,192,384,48,128,128)

        self.avgpool = nn.AvgPool2d(kernel_size = 7 , stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear( 1024 , num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, 1, 1, 0)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Inceptionv1WithAux(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=1000):
        super(Inceptionv1WithAux, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(ConvBlock(64, 64, 1, 1, 0), ConvBlock(64, 192, 3, 1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)
        
        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.training else None
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.training else None
        
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        if self.training:
            return x, aux1, aux2
        else:
            return x