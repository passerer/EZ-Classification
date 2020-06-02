class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.layers = nn.Sequential(*[
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(bn_size*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ])

    def forward(self, x):
        y = self.layers(x)
        return torch.cat([x, y], 1)

def get_DenseBlock(num_layers,ni,bn_size,growth_rate):
    return nn.Sequential(*[
        DenseLayer(ni+growth_rate*i,growth_rate, bn_size) for i in range(num_layers)
    ])
    

def get_transition(ni,no):
    return nn.Sequential(*[
        nn.BatchNorm2d(ni),
        nn.ReLU(inplace=True),
        nn.Conv2d(ni, no, kernel_size=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    ])


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, num_classes=10):
        super(DenseNet, self).__init__()

        ni = 2 * growth_rate

        layers = [
                nn.Conv2d(3, ni, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(ni),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]

        num_feature = ni
        for i, num_layers in enumerate(block_config):
            layers.append(get_DenseBlock(num_layers, num_feature, bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                layers.append(get_transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        layers.append(nn.BatchNorm2d(num_feature))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d(1))

        layers += [nn.Flatten(),nn.Linear(num_feature, num_classes)]
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


# DenseNet_BC for ImageNet

def get_densenet_net():
    # DenseNet 121
    #model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=config.num_classes)
    model = torchvision.models.densenet121(pretrained = False)    
    model.classifier = nn.Linear(1024,config.num_classes)
    print_model_parm_nums(model)
    return model    
