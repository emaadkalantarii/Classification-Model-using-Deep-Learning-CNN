import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Dataset class to handle images
class BrainDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # load the image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # transform if needed
        if self.transform:
            img = self.transform(img)
        
        return img, label

# function to load all the images
def load_data(data_dir):
    img_paths = []
    labels = []
    
    # get good images (label=1)
    good_dir = os.path.join(data_dir, 'good')
    for file in os.listdir(good_dir):
        if file.endswith('.png'):
            img_paths.append(os.path.join(good_dir, file))
            labels.append(1)  # good = 1
    
    # get bad images (label=0)
    bad_dir = os.path.join(data_dir, 'bad')
    for file in os.listdir(bad_dir):
        if file.endswith('.png'):
            img_paths.append(os.path.join(bad_dir, file))
            labels.append(0)  # bad = 0
    
    # print some info
    print(f"Total images: {len(img_paths)}")
    print(f"Good images: {labels.count(1)}")
    print(f"Bad images: {labels.count(0)}")
    
    return img_paths, labels



# CNN model for brain response classification
class BrainModel(nn.Module):
    def __init__(self):
        super(BrainModel, self).__init__()
        
        # first conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # second conv layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # third conv layer
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # fourth conv layer
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        
        # adaptive pooling to handle different sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # dropout to avoid overfitting
        self.fc2 = nn.Linear(1024, 1024)
        self.relu8 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # dropout to avoid overfitting
        self.fc3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()  # for binary output
    
    def forward(self, x):
        # pass through conv layers
        x = self.pool1(self.relu2(self.conv2(self.relu1(self.conv1(x))))) #First block
        x = self.pool2(self.relu4(self.conv4(self.relu3(self.conv3(x))))) #second block
        x = self.pool3(self.relu5(self.conv5(x))) #thirds block
        x = self.pool4(self.relu6(self.conv6(x))) #fourth block
        
        # global pooling
        x = self.adaptive_pool(x)
        
        # flatten before fc layers
        x = x.view(x.size(0), -1)
        
        # pass through fc layers
        x = self.relu7(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu8(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x
    

# try to use GPU if available
def try_gpu():
    if torch.cuda.is_available():
        print("Found GPU for training.")
        
        # try to use the GPU
        try:
            return torch.device('cuda')
        except:
            print("Error with GPU, falling back to CPU")
            return torch.device('cpu')
    else:
        print("No GPU found. Using CPU instead.")
        return torch.device('cpu')


# main function
def main():
    # set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # try to use GPU
    device = try_gpu()
    print(f"Using: {device}")
    
    
    # parameters
    data_dir = 'topomaps'
    batch_size = 8 
    num_epochs = 60
    lr = 0.001
    
    # transformations for images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # resize to 128x128
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # load all data
    img_paths, labels = load_data(data_dir)
    
    # split data into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.3, random_state=42
    )
    # split validation data into test and validation sets
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_paths, val_labels, test_size=0.5, random_state=42
    )

    # create datasets
    train_dataset = BrainDataset(train_paths, train_labels, transform)
    val_dataset = BrainDataset(val_paths, val_labels, transform)
    
    # create data loaders
    workers = 2 if device.type == 'cuda' else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=workers
    )
    
    # create model and move to device
    model = BrainModel().to(device)
    
    # loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # simple learning rate scheduler
    # reduce learning rate when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # variables to track best model
    best_val_loss = float('inf')
    
    print("\nStarting training...")

    for epoch in range(num_epochs):
            # TRAINING
            model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                # move to device
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # VALIDATION
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad(): 
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.float().unsqueeze(1).to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    # calculate accuracy
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            
            # update learning rate
            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            
            # print stats
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {val_acc:.4f}')
            print(f'  Learning Rate: {curr_lr:.6f}')
            
            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "model.pth")
                print(" Saved new best model!!")    
    
    
    # free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("Training finished!")

if __name__ == "__main__":
    main()