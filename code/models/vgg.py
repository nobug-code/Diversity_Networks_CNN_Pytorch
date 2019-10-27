'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG16_2': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 0, 'M'],
}
k_number = 0 
class VGG(nn.Module):
    def __init__(self, vgg_name,number):
        super(VGG, self).__init__()
        self.k_number = number
        self.features = self._make_layers(cfg[vgg_name])
        if(number == 0):
            self.classifier = nn.Linear(512, 10)
        else:
            self.classifier = nn.Linear(self.k_number, 10)
            #self.relu = nn.ELU(inplace=True)
            #self.classifier2 = nn.Linear(self.k_number/16,self.k_number/32)
            #self.relu2 = nn.ELU(inplace=True)
            #self.classifier3 = nn.Linear(self.k_number/32,10)
        self.weight_init()
    def weight_init(self):
        if isinstance(self.features, nn.Conv2d):
            nn.init.xavier_uniform_(self.features.weight)
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:

                if(x == 0):
                    layers += [nn.Conv2d(in_channels, self.k_number, kernel_size=3, padding=1),
                           nn.BatchNorm2d(self.k_number),
                           nn.ReLU(inplace=True)]
                    in_channels = self.k_number
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                    in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    net = VGG('VGG16_2',512+64)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(net)
