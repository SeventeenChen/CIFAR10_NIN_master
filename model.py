# -*- coding: utf-8 -*-


from modelsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch


##########################
### MODEL
##########################



class NIN(nn.Module):
    def __init__(self, num_classes):
        super(NIN, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(160),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifier(x)
        logits = x.view(x.size(0), self.num_classes)
        probas = torch.softmax(logits, dim=1)
        return logits, probas

if __name__ == '__main__':

	# Device
    NUM_CLASSES = 10
    GRAYSCALE = False
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model =  NIN(NUM_CLASSES)
    model.to(device)
    summary(model, torch.ones(128, 3, 32, 32), batch_size=128, show_input=False)