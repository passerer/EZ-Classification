

from torch import nn
from config import config
from tools.model_tools import print_model_parm_nums

#se resnet 50

class SEBlock(nn.Module):
    def __init__(self, ni, filters, stride=1):
        super(SEBlock, self).__init__()
        filter1, filter2, filter3 = filters
        self.relu = nn.ReLU(inplace=True)
        self.res = nn.Sequential(
            nn.Conv2d(ni, filter1, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU(),
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU(),
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or ni!=filter3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ni, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(filter3,filter3//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3//16,filter3,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.res(x)
        weight = self.se(y)
        y = y*weight
        x_shortcut = self.shortcut(x)
        y = y + x_shortcut
        y = self.relu(y)
        return y

class SENet(nn.Module):
    def __init__(self,num_class:int,block:list=[3,4,6,3]):
        super(SENet,self).__init__()
        layers = []
        layers.append( nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))
        layers.append( self._make_layer(64, (64, 64, 256), block[0],1) )
        layers.append( self._make_layer(256, (128, 128, 512), block[1], 2))
        layers.append( self._make_layer(512, (256, 256, 1024), block[2], 2))
        layers.append( self._make_layer(1024, (512, 512, 2048), block[3], 2))
        
        layers +=  [nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048,num_class)]
        self.layers = nn.Sequential(*layers)
        self._init_weight();
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.layers(x)

    def _make_layer(self,ni, filters, num, stride=1):
        layers = []
        layers.append(SEBlock(ni, filters, stride=stride))
        for i in range(1, num):
            layers.append(SEBlock(filters[2], filters, stride=1))
        return nn.Sequential(*layers)
    
def get_senet_net():
    # se resnet 50
    model = SENet(config.num_classes,[3,4,6,3])
    print_model_parm_nums(model)
    return model    