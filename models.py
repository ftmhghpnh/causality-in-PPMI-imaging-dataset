import torch
import torchvision
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=4):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=8192, out_features=100)        
        self.fc2 = nn.Linear(in_features=100, out_features=num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool2(output)
        
        output = torch.flatten(output, start_dim=1)

        output = self.fc1(output)
        output = self.relu(output)
        
        output = self.fc2(output)
        output = self.softmax(output)

        return output 

    
class SimpleNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(in_features=24 * 60 * 60, out_features=num_classes)
        self.softmax = nn.Softmax()

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool1(output)

        output = self.conv3(output)
        output = self.relu3(output)
        
        output = self.pool2(output)

        output = self.conv4(output)
        output = self.relu4(output)
        
#         print(output.shape)

        output = torch.flatten(output, start_dim=1)

        output = self.fc(output)
        output = self.softmax(output)

        return output 

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, freeze_weights=True, layers_to_train=[]):
        super(ResNet18Classifier, self).__init__()

        self.model = torchvision.models.resnet18(pretrained = pretrained)
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            if name.split('.')[0] in layers_to_train:
                param.requires_grad = True

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, input):
        output = self.model(input)
        output = self.softmax(output)

        return output
    
class ResNet18WithNonImageFeaturesClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, freeze_weights=True, layers_to_train=[], features_count=246):
        super(ResNet18WithNonImageFeaturesClassifier, self).__init__()

        self.model = torchvision.models.resnet18(pretrained = pretrained)
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            if name.split('.')[0] in layers_to_train:
                param.requires_grad = True

        
        self.classifier = nn.Linear(self.model.fc.in_features + features_count, num_classes)
        self.model.fc = nn.Identity()
       
        self.softmax = nn.Softmax()

    def forward(self, img, features):
        output = self.model(img)
        output = self.classifier(torch.cat([output, features], dim=1))
        output = self.softmax(output)

        return output
