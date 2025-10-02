# Import necessary libraries
import os
import random
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from jupyter_client.consoleapp import classes
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision import models, datasets
from torchsummary import summary
from torchinfo import summary

import torchvision
import torch.nn.functional as f
import h5py


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def learning_curves_tuning(history, fine_tune_epoch=None):
    """
    Plotting the curves of the loss and accuracy training and Validation

    Parameters:
    - history: History of training and validation of trained model
    - fine_tune_epoch (Optional): Epoch of fine-tuning if the model is fine-tuned

    Returns:
    - Loss and Accuracy plots for train and validation
    """

    # Get training and validation data from initial training
    tacc = history["train_accuracy"]
    tloss = history["train_loss"]
    vacc = history["val_accuracy"]
    vloss = history["val_loss"]

    total_epochs = [i+1 for i in range(len(tacc))]

    # Find the best epoch based on validation loss and accuracy
    index_loss = np.argmin(vloss)  # epoch with the lowest validation loss
    val_lowest = history["best_val_loss"]
    
    index_acc = np.argmax(vacc)  # epoch with the highest validation accuracy
    acc_highest = history["best_val_acc"]

    # Define plot labels
    sc_label = 'best epoch= ' + str(index_loss + 1)
    vc_label = 'best epoch= ' + str(index_acc + 1)

    # plt.style.use('fivethirtyeight')
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Plot loss curves
    ax1.plot(total_epochs, tloss, 'r', label='Training Loss')
    ax1.plot(total_epochs, vloss, 'g', label='Validation Loss')
    ax1.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=sc_label)

    # Add fine-tuning marker
    if fine_tune_epoch:
        ax1.axvline(x=fine_tune_epoch, color='orange', linestyle='--',
                        label='Start Fine Tuning')

    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot accuracy curves
    ax2.plot(total_epochs, tacc, 'r', label='Training Accuracy')
    ax2.plot(total_epochs, vacc, 'g', label='Validation Accuracy')
    ax2.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=vc_label)

    # Add fine-tuning marker
    if fine_tune_epoch:
        ax2.axvline(x=fine_tune_epoch, color='orange', linestyle='--',
                        label='Start Fine Tuning')

    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Optional: Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, paths, transform=None, is_train=True):
        # data loadig
        self.paths = paths
        self.transform = transform
        self.is_train = is_train

        # Add validation for empty paths
        if not paths:
            raise ValueError("Empty paths list provided to dataset")
            
        # Add file existence check
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        img = img.convert("RGB") # Some images 4 channels
        label = self.paths[index][-15:-10]

        if self.transform:
            if self.is_train:
                img = self.transform["train_transform"](img)
            else:
                img = self.transform["valid_transform"](img)
        
        return img, (1 if label == "Covid" else 0)
        
    def __len__(self):
        return len(self.paths)

# Custom CNNModel class
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        
        # we can use nn.Sequential & nn.Functional
        # Convolutional layers
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),  # 224x224 -> 224x224
            nn.Conv2d(16, 16, kernel_size=3, padding="same"),  # 224x224
            nn.BatchNorm2d(16),
            # 224x224 -> 112x112
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),  # 112x112
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),  # 112x112
            nn.BatchNorm2d(32),
            # 112x112 -> 56x56
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv_block1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_block2(x))
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Hyperparameters
image_size = 224
num_epochs = 20
epochs_tuning = 40
batch_size = 16
lr = 0.001
patience = 7
factor = 0.1
batch_s = 32

# pre-trained model
model = models.mobilenet_v3_large(weights="IMAGENET1k")

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
early_stopping = EarlyStopping(patience=7) # Initialize early stopping

# Data Preprocessing
train_transform = v2.Compose([
    v2.Resize(image_size),
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(p = 0.5), # 50 % from images will apply to
    v2.RandomVerticalFlip(p = 0.5), # 50 % from images will apply to
    v2.RandomRotation(10),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

valid_transform = v2.Compose([
    v2.Resize(image_size),
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

transform = {
    "train_transform":train_transform,
    "valid_transform":valid_transform
}

# Data loading
dataset_dir = "/kaggle/input/sarscov2-ctscan-dataset"
dataset = datasets.ImageFolder(root=dataset_dir)
class_names = dataset.classes
num_classes = len(class_names)

dataset_paths = glob(f"{dataset_dir}/*/*.png")

train_paths, test_paths = train_test_split(dataset_paths, test_size=0.15, random_state=42)
val_paths, test_paths = train_test_split(test_paths, test_size=0.5, random_state=42)

train_data = CustomDataset(train_paths, transform["train_transform"])
valid_data = CustomDataset(val_paths, transform["valid_transform"], is_train=False)
test_data = CustomDataset(test_paths, transform["valid_transform"], is_train=False)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_s)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_s)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_s)


# Lists to store metrics
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
    
best_val_loss = float('inf') # Save best model best loss
best_val_acc = 0.0 # Save best model val accuracy
best_model_state = None  # Store best model state

# Training function of the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
                , best_val_loss, best_val_acc, best_model_state):
    """
    Train the model and save the history of the model

    Parameters:
    - model: Model for training 
    - train_loader: loader that contains images and labels for the train data
    - val_loader: loader that contains images and labels for the validation data
    - criterion: loss function used to calculate the loss
    - optimizer: optimizer used to find the optimal
    - schedular: schedular to reduce learning rate to optimize the model
    - num_epochs: number of epochs to train model
    - device: CUDA device

    Returns:
    - model: Best model trained with the lowest validation loss.
    - history: History of the model with loss and accuracy of train and validation.
    """

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Training phase
        epoch_train_loss, epoch_train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print("Finished training")

        # Validation phase
        epoch_val_loss, epoch_val_acc = validate_model(model, val_loader, device, criterion)
        
        # Store metrics
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            torch.save({
                "epoch": epoch,
                "model_state_dict":model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                },f"best_model_loss.pth")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict":model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_val_acc,
                }, f"best_model_acc.pth")
            
        # Print progress
        # print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}')
        print(f'-'*70)

        # Early stopping check (optional)
        if early_stopping.step(epoch_val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Scheduler step
        scheduler.step(epoch_val_loss)

    # Load best model before returning
    model.load_state_dict(best_model_state)

    history = {'train_loss': train_losses, 
                'train_accuracy': train_accuracies,
                'val_loss': val_losses, 
                'val_accuracy': val_accuracies,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc}
    
    return model, history 

def validate_model(model, val_loader, criterion, device):
    """
    Calculates the validation loss and validation accuracy for the given model in one epoch

    Parameters:
    - model: Trained model for validation
    - val_loader: loader that contains images and labels for the validation dataset
    - criterion: loss function used to calculate the loss
    - device: CUDA device

    Returns:
    - epoch_val_loss: Average validation loss over the validation dataset in one epoch.
    - epoch_val_acc: Validation accuracy over the validation dataset in one epoch.
    """

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
        
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            # ... validation step code ...
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1) # (outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate validation metrics
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = correct / total

    return epoch_val_loss, epoch_val_acc
    
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Calculates the train loss and train accuracy for the given model in one epoch

    Parameters:
    - model: Model for training
    - train_loader: loader that contains images and labels for the train data
    - criterion: loss function used to calculate the loss
    - optimizer: optimizer used to find the optimal
    - device: CUDA device

    Returns:
    - epoch_loss: Average train loss over the train dataset in one epoch.
    - epoch_acc: Train accuracy over the train dataset in one epoch.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': loss.item(),
            'train_acc': correct/total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

best_model_path = "/kaggle/working/best_model_loss.pth"
best_model = model(num_classes).to(device)
best_model.load_state_dict(torch.load(best_model_path, map_location = device))

validation_loss, validation_accuracy = validate_model(best_model, valid_loader, criterion, device)
print(f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy*100:.2f}%')

test_loss, test_accuracy = validate_model(best_model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%')

def test_model(model, test_loader, device):
    """
    Calculates the classification report and confusion matrix of the test dataset

    Parameters:
    - model: Trained model
    - test_loader: loader that contains images and labels for the test dataset
    - device: CUDA device

    Returns:
    - cr: Classification report to test model performance
    - cm: Confusion matrix to test model performance
    """

    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())
            
    
    cr = classification_report(true_labels, predictions, output_dict=True, target_names=class_names)
    cm = confusion_matrix(true_labels, predictions)
    return cr, cm

cr, cm = test_model(model, test_loader, device)
plot_confusion_matrix(cm, class_names=class_names, figsize=(8,6))

CNN = CustomModel(num_classes).to(device)

summary(model, input_size=(3, 256, 256))
# Print model summary
summary(model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# Visualize some results
def visualize_results(model, test_loader, classes, num_images=5):
    model.to(device)
    model.eval()

    # classes = ('Covid', 'Normal')

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Make predictions
    images_device = images[:num_images].to(device)
    with torch.no_grad():
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)

    # Show images
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_images):
        ax = fig.add_subplot(1, num_images, i + 1, xticks=[], yticks=[])
        # Convert image from tensor and normalize
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        title = f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}"
        ax.set_title(title, color=("green" if predicted[i] == labels[i] else "red"))

    plt.tight_layout()
    plt.show()