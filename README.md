# Census Income Classification using PyTorch

## AIM
To build a binary classification model using the Census Income dataset to predict whether an individual earns more than $50,000 annually.

## THEORY
Classification is a supervised learning task where the goal is to predict categorical outcomes based on input features. Traditional models often struggle with heterogeneous data (mix of categorical and continuous). Neural networks with embeddings and normalization can handle such complexity efficiently. In this experiment, categorical variables are represented using embeddings, and continuous variables are normalized. A feedforward neural network is trained to classify income levels.

## MODEL DESIGN
# Input Features
Categorical: sex, education, marital-status, workclass, occupation

Continuous: age, hours-per-week

Neural Network Architecture

Embedding layers for categorical inputs

Batch normalization for continuous inputs

One hidden layer with 50 neurons, dropout = 0.4

Output layer with 2 classes (<=50K or >50K)

## DESIGN STEPS
# STEP 1: Data Preparation
Load dataset (income.csv, 30,000 rows).

Separate categorical, continuous, and label columns.

Convert values into arrays and PyTorch tensors.

Split dataset into training (25,000) and testing (5,000).

# STEP 2: Model Definition
Define a custom TabularModel class.

Combine categorical embeddings with continuous features.

Apply batch normalization, linear layers, ReLU activation, and dropout.

# STEP 3: Loss Function and Optimizer
Use nn.CrossEntropyLoss() for classification.

Use torch.optim.Adam optimizer with learning rate 0.001.

# STEP 4: Training
Train for 300 epochs.

Record training loss and accuracy.

# STEP 5: Evaluation
Evaluate model on the test set.

Report final loss and accuracy.

# STEP 6: User Input Prediction (Bonus)
Function accepts user details such as age, sex, education, hours/week.

Model outputs predicted income class.

# PROGRAM AND REQUIREMENTS:

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

torch.manual_seed(42)
np.random.seed(42)


data_dir = "./dataset"  # Structure: data/train/{cats,dogs,panda}, data/test/{cats,dogs,panda}


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
test_data  = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = train_data.classes
print("Classes:", class_names)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(pretrained=True)


for param in model.parameters():
    param.requires_grad = False


in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(class_names))  # 3 classes
)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


epochs = 5 
for epoch in range(epochs):
    model.train()
    running_loss, correct = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_data)
    epoch_acc = correct / len(train_data)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")


model.eval()
test_correct, test_loss = 0, 0.0
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)

        test_correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_data)
test_acc = test_correct / len(test_data)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")


cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = test_transforms
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][pred].item() * 100

    result = class_names[pred.item()]
    print(f"Prediction: {result} ({prob:.2f}% confidence)")
    return result
## OUTPUT
<img width="628" height="750" alt="project output" src="https://github.com/user-attachments/assets/6a411dae-55bd-479f-826b-c60598e91a39" />

## RESULT
Thus, a tabular neural network model was successfully developed using PyTorch to classify income level. The model achieved ~85% test accuracy.
