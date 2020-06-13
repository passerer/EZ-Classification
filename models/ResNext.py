from torch import nn
from config import config
from tools.model_tools import print_model_parm_nums

#ResNext

class GroupResBlock(nn.Module):
    expansion = 2
    def __init__(self, ni, cardinality=32, bottleneck_width=4, stride=1):
        super(GroupResBlock, self).__init__()
        group_width = cardinality * bottleneck_width
        self.res = nn.Sequential(
            nn.Conv2d(ni, group_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(),
            nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(),
            nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion*group_width),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or ni != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ni, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        y = self.res(x)
        y += self.shortcut(x)
        y = F.relu(y)
        return y



class ResNeXt(nn.Module):
    def __init__(self,  num_class:int,num_blocks:list=[3.4,6,3], cardinality=32, bottleneck_width=4):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        
        layers =[]
        layers += [
        nn.Conv2d(3, 64, kernel_size=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
        ]
        
        layers.append(self._make_layer(num_blocks[0], 1))
        layers.append(self._make_layer(num_blocks[1], 2))
        layers.append(self._make_layer(num_blocks[2], 2))
        layers.append(self._make_layer(num_blocks[3], 2))
        layers +=[ 
                  nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Flatten(),
                  nn.Linear(self.in_planes, num_class)
                 ]
        self.layers = nn.Sequential(*layers)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(GroupResBlock(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = GroupResBlock.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= GroupResBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return  self.layers(x)
    
def get_resnext_net():
    model =  ResNeXt(config.num_classes,num_blocks=[3,4,6,3], cardinality=32, bottleneck_width=4)
    print_model_parm_nums(model)
    return model
       