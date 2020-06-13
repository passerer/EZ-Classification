from torch import nn
from config import config
from tools.model_tools import print_model_parm_nums

#DarkNet

def conv_bn_lrelu(ni:int, nf:int, ks:int=3, stride:int=1):
    #"Create a seuence Conv2d->BatchNorm2d->LeakyReLu layer."
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks//2),
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))

class ResLayer(nn.Module):
    "Resnet style layer with `ni` inputs."
    def __init__(self, ni:int):
        super(ResLayer, self ).__init__() 
        self.conv1 = conv_bn_lrelu(ni, ni//2, ks=1)
        self.conv2 = conv_bn_lrelu(ni//2, ni, ks=3)

    def forward(self, x): return x + self.conv2(self.conv1(x))

class Darknet(nn.Module):
    def make_group_layer(self, ch_in:int, num_blocks:int, stride:int=1):
        "starts with conv layer - `ch_in` channels in - then has `num_blocks` `ResLayer`"
        return [conv_bn_lrelu(ch_in, ch_in*2,stride=stride)
               ] + [(ResLayer(ch_in*2)) for i in range(num_blocks)]

    def __init__(self, num_blocks:list, num_classes:int, nf=32):
        super( Darknet, self ).__init__() 
        "create darknet with `nf` and `num_blocks` layers"
        layers = [conv_bn_lrelu(3, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2)
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1),nn.Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)
    #    self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): return self.layers(x)
    
def get_darknet_net():
    model = Darknet([1,2,8,8,4],config.num_classes)
    print_model_parm_nums(model)
    return model