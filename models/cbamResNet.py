class CBAMBlock(nn.Module):
    def __init__(self, ni, filters, stride=1):
        super(CBAMBlock, self).__init__()
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
        self.cbam = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride = 1,padding=7//2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.res(x)
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out, _ = torch.max(y, dim=1, keepdim=True)
        weight = self.cbam(torch.cat([avg_out, max_out], dim=1))
        y = y*weight
        x_shortcut = self.shortcut(x)
        y = y + x_shortcut
        y = self.relu(y)
        return y

class CBAM_ResNet(nn.Module):
    def __init__(self,num_class:int,block:list=[3,4,6,3]):
        super(CBAM_ResNet,self).__init__()
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
        layers.append(CBAMBlock(ni, filters, stride=stride))
        for i in range(1, num):
            layers.append(CBAMBlock(filters[2], filters, stride=1))
        return nn.Sequential(*layers)
    
def get_cbam_resnet_net():
    model = CBAM_ResNet(config.num_classes,[3,4,6,3])
    print_model_parm_nums(model)
    return model