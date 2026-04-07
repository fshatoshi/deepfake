#importation of the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

#definition of the CNN architecture
class FaceCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(FaceCNN , self ).__init__()

        #convolutional layers

        #1st block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.pool1=nn.MaxPool2d(2,2)

        #2nd block

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.pool2=nn.MaxPool2d(2,2)

        #3rd block
        self.conv3= nn.Conv2d(64,128 , kernel_size=3 , stride=1 , padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.pool3=nn.MaxPool2d(2,2)

        #4th block
        self.conv4= nn.Conv2d(128,256 , kernel_size=3 , stride=1 , padding=1)
        self.bn4=nn.BatchNorm2d(256)
        self.pool4=nn.MaxPool2d(2,2)

        #Classifier
        self.fc1=nn.Linear(256*14*14,512)
        self.dropout1=nn.Dropout(0.4)
        self.fc2=nn.Linear(512,256)
        self.dropout2=nn.Dropout(0.4)
        self.fc3=nn.Linear(256,num_classes)

    #Forward
    def forward(self,x):
        x=self.pool1(F.relu(self.bn1(self.conv1(x))))
        x=self.pool2(F.relu(self.bn2(self.conv2(x))))
        x=self.pool3(F.relu(self.bn3(self.conv3(x))))
        x=self.pool4(F.relu(self.bn4(self.conv4(x))))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=self.dropout1(x)
        x=F.relu(self.fc2(x))
        x=self.dropout2(x)
        x=self.fc3(x)
        return x
    #Feutures extraction

    def extract_features(self,x):
        x=self.pool1(F.relu(self.bn1(self.conv1(x))))
        x=self.pool2(F.relu(self.bn2(self.conv2(x))))
        x=self.pool3(F.relu(self.bn3(self.conv3(x))))
        x=self.pool4(F.relu(self.bn4(self.conv4(x))))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x
    
    
if __name__ == "__main__":
    model = FaceCNN(num_classes=10)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total paramètres: {total_params:,}") 



