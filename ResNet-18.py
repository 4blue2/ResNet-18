
import torch
from torch import nn
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms



# Load the Fashion MNIST dataset and apply transformations
train_set = datasets.FashionMNIST('./data', train=True, download=True)
test_set = datasets.FashionMNIST('./data', train=False, download=True)


# Define a transformation that flips the images only for the test data
train_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.RandomHorizontalFlip(p=0.2)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = datasets.FashionMNIST('./data', train=True, download=False, transform=train_transform)
test_set = datasets.FashionMNIST('./data', train=False, download=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

#-------------------Code from Lab_8 to find the mean and std------------------------------
n_samples_seen = 0.
mean = 0
std = 0
for train_batch, train_target in train_loader:
    batch_size = train_batch.shape[0]
    train_batch = train_batch.view(batch_size, -1)
    this_mean = torch.mean(train_batch, dim=1)
    this_std = torch.sqrt(
        torch.mean((train_batch - this_mean[:, None]) ** 2, dim=1))
    mean += torch.sum(this_mean, dim=0)
    std += torch.sum(this_std, dim=0)
    n_samples_seen += batch_size

mean /= n_samples_seen
std /= n_samples_seen
print(mean, std)
#-----------------------------------------------------------------------------------------
#Normalize the data
train_set = datasets.FashionMNIST('./data', train=True, download=False, transform=transforms.Compose([
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize(mean=mean.view(1), std=std.view(1))]))
test_set = datasets.FashionMNIST('./data', train=False, download=False,transform=transforms.Compose([
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize(mean=mean.view(1), std=std.view(1))]))

# Create DataLoaders for the datasets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)



class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        #this will be used to downsample knowledge from previous layers so that
        #I can add it to the current layer. Dimensions need to match in order to do the addition.
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=4, padding=0),
            nn.BatchNorm2d(64)
        )

        #input channels must be equal to output channels of previous block
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        #Hardcoded the downsampling for each layer
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )

        #hardcoded the downsampling for each layer
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )

        self.downsample4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512,10)
        self.drop = nn.Dropout(0.35) #used at the end to avoid overfitting 
        self.soft = nn.Softmax() #overall probability must equal to 1
    
    def forward(self, x):

        prev_x = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        prev_x = self.downsample1(prev_x) #downsample tensor for the addition to take place

        x = self.layer1(x)
        x = self.relu(x)
        x = torch.add(x, prev_x) #combine current features learned with previous features
        x = self.relu(x)

        prev_x = x
        prev_x = self.downsample2(prev_x)

        x = self.layer2(x)
        x = self.relu(x)
        x = torch.add(x, prev_x)
        x = self.relu(x)

        prev_x = x
        prev_x = self.downsample3(prev_x)

        x = self.layer3(x)
        x = self.relu(x)
        x = torch.add(x, prev_x)
        x = self.relu(x)

        prev_x = x
        prev_x = self.downsample4(prev_x)

        x = self.layer4(x)
        x = self.relu(x)
        x = torch.add(x, prev_x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.soft(x)

        return x

#train_resnet and test_resnet are from the labs with minor changes
def train_resnet(num_epochs, model, train_loader, loss_fn, optimizer):

    for epoch in range(num_epochs):

        #trainnig mode
        model.train()
        total_loss = 0.
        total_acc = 0.

        #change the learning rate during the learning 
        if epoch > 12:
           optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

        for inputs, labels in train_loader:
            #move data to gpu
            inputs = inputs.cuda()
            labels = labels.cuda()

            #get the predictions
            outputs = model(inputs)

            #compare output with image label
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()

            #compute grad. of loss
            loss.backward()

            #update the weights
            optimizer.step()

            total_loss += loss.item()

            #get total predictions and then find predictions that match the labels to count the number of
            #correct preds and get the total accuracy by div with total labels 
            _, preds = torch.max(outputs, dim=1)
            total_acc += torch.sum(preds == labels).item() / len(labels)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, train loss: {avg_loss}, train acc: {avg_acc}")

def test_resnet(model, test_loader, loss_fn):
    model.eval()

    total_loss = 0.
    total_acc = 0.

    #no need to calculate the grad. in testing phase
    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            total_acc += torch.sum(preds == labels).item() / len(labels)

    avg_loss = total_loss / len(test_loader)
    avg_acc = total_acc / len(test_loader)

    print(f"Average loss: {avg_loss}, Average accuracy: {avg_acc}")


def train_loop(model, epochs):
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)#set seed of all gpu cores to be the same
    
    # make training reproducable 
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    #move the model to the gpu
    model.to('cuda')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_resnet(epochs, model, train_loader, loss_fn, optimizer)
    test_resnet(model, test_loader, loss_fn)

ResNet = ResNet18()

train_loop(ResNet, 20) # 20 = number of epochs