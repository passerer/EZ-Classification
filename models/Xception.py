## xception

def sep_conv(ni,nf,pad=None,pool=False,act=True):
    layers =  [nn.ReLU()] if act else []
    layers += [
        nn.Conv2d(ni,ni,3,1,1,groups=ni,bias=False),
        nn.Conv2d(ni,nf,1,bias=False),
        nn.BatchNorm2d(nf)
    ]
    if pool: layers.append(nn.MaxPool2d(2,ceil_mode=True))
    return nn.Sequential(*layers)

def conv(ni,nf,ks=1,stride=1, pad=None, act=True):
    if pad is None: pad=ks//2
    layers = [
        nn.Conv2d(ni,nf,ks,stride,pad,bias=False),
        nn.BatchNorm2d(nf),
    ]
    if act: layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class ConvSkip(nn.Module):
    def __init__(self,ni,nf=None,act=True):
        super( ConvSkip, self ).__init__() 
        self.nf,self.ni = nf,ni
        if self.nf is None: self.nf = ni
        self.conv = conv(ni,nf,stride=2, act=False)
        self.m = nn.Sequential(
            sep_conv(ni,ni,act=act),
            sep_conv(ni,nf,pool=True)
        )

    def forward(self,x): 
        return self.conv(x) + self.m(x)

class MiddleFlow(nn.Module):
    def __init__(self,nf:int,act=True):
        super( MiddleFlow, self ).__init__() 
        self.layers = nn.Sequential(*[sep_conv(nf,nf) for i in range(3)])

    def forward(self,x): 
        return self.layers(x) + x
    
class Xception(nn.Module):
    def __init__(self,num_classes:int, k=8, n_middle=8):
        super( Xception, self ).__init__() 
        layers = [
        conv(3, k*4, 3, 2),
        conv(k*4, k*8, 3),
        ConvSkip(k*8, k*16, act=False),
        ConvSkip(k*16, k*32),
        ConvSkip(k*32, k*91),
    ]
        for i in range(n_middle): layers += [MiddleFlow(k*91)]
        layers += [
            ConvSkip(k*91,k*128),
            sep_conv(k*128,k*192,act=False),
            sep_conv(k*192,k*256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(k*256,num_classes)
        ]
        self.layers = nn.Sequential(*layers)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): return self.layers(x)
    
def get_xception_net():
    model = Xception(config.num_classes,n_middle=5)
    print_model_parm_nums(model)
    return model