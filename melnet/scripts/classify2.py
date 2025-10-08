import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
sys.path.append(grandparent_dir)
from usage.models_ma19 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%%

name1 = "ffi2v_ncb"
name2 = "ffi2v_cb3"

feature = "gen1"

num_epochs = 1500
patience = 50


#%%
cohs = [0]
if feature in ["gen1","gen1_98t"]:
   percentiles = np.arange(5, 100, 5)     
   NF =  (len(percentiles)*2 + 1)*len(cohs)       

folder_A = f"../s1_gen/save/{feature}/{name1}/"        
folder_B = f"../s1_gen/save/{feature}/{name2}/"        
save_name = f"./save/{feature}/models/bi_{name1}_{name2}_t.pth"
save_name2 = f"./save/{feature}/models/bi_{name1}_{name2}.pth"

dataset = BiDataset(folder_A, folder_B, NF)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = BiClassifier(input_size=NF).to(device)  # Move model to GPU if available
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.0001)

os.makedirs(os.path.dirname(save_name), exist_ok=True)
os.makedirs(os.path.dirname(save_name2), exist_ok=True)

#%%
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs.view(-1), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        predicted = (outputs.view(-1) > 0.5).float()  # Threshold the output to get binary predictions
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Store training loss and accuracy for each epoch
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    # Validation loss and accuracy after each epoch
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move data to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            val_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs.view(-1) > 0.5).float()  # Threshold the output to get binary predictions
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Store validation loss and accuracy for each epoch
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct_val / total_val)

    # Print statistics after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"TLoss: {train_losses[-1]:.4f}, TAcc: {train_accuracies[-1]*100:.2f}%, "
          f"VLoss: {val_losses[-1]:.4f}, VAcc: {val_accuracies[-1]*100:.2f}%, "
          f"N: {epochs_without_improvement}")
    
    # Check early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), save_name)
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break    

torch.save(best_model_state, save_name2)

#%% evaluation
    
# Plotting the training and validation loss and accuracy
epochs = range(1, num_epochs + 1)

# Plot Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#%% evaluation2

# Load the best model (in case of early stopping)
model.load_state_dict(torch.load(save_name2))

# Test the model on the test set
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Move data to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        predicted = (outputs.view(-1) > 0.5).float()  # Threshold the output to get binary predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store all predictions and labels for confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()