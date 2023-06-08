import csv
from pathlib import Path
import numpy as np
from resnet import ResNet, BasicBlock
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def train(model, loader, criterion, optimizer):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(loader):
        # print((labels[0].item()))
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(f"outputs {outputs}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        # print(f"predicted {predicted}")
        correct += predicted.eq(labels.data).cpu().sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def test(model, loader, criterion):
    """Evaluates the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # print(f"outputs {outputs}")
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels.data).cpu().sum().item()
            total += labels.size(0)

            y_true += labels.cpu().tolist()
            y_pred += predicted.cpu().tolist()
            y_score += nn.functional.softmax(outputs, dim=1).cpu().tolist()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    # Convert y_score to a NumPy array
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return epoch_loss, epoch_acc, y_true, y_pred, y_score

# Define a function to plot the ROC curve for a single fold
def plot_roc_curve(fpr, tpr, roc_auc, fold):
    plt.plot(fpr, tpr, label=f'Fold {fold+1} (AUC = {roc_auc:.2f})')
# Define k and other hyperparameters
k = 10
lr = 0.00012
num_epochs = 50
batch_size = 12
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path_data = Path('data')
# test_dir = Path('data/test')

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 224x224
    transforms.Resize(size=(224, 224)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=path_data, transform=data_transform, target_transform=None)


kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Split the dataset into training and test sets
train_val_indices, test_indices = next(iter(kf.split(dataset.imgs, dataset.targets)))

# Split the training/validation indices into training and validation sets
train_val_targets = [dataset.targets[i] for i in train_val_indices]
train_indices, val_indices = next(iter(kf.split(train_val_indices, train_val_targets)))

# Create the data loaders for the training, validation, and test sets
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=torch.utils.data.SubsetRandomSampler(train_val_indices[train_indices])
)
val_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=torch.utils.data.SubsetRandomSampler(train_val_indices[val_indices])
)
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=torch.utils.data.SubsetRandomSampler(test_indices)
)



# Define the model
print('[INFO]: Training ResNet18 ...')
model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Create a CSV file to store the results
with open('results_test14.csv', 'w', newline='') as csvfile:
    fieldnames = ['fold', 'epoch','train_loss', 'train_acc', 'val_loss', 'val_acc', 'mcc', 'f1_score',
                  'specificity', 'sensitivity', 'auc', 'auc_pr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    results = []
    # Loop over the folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_val_indices, train_val_targets)):
        print(f"Fold {fold + 1}")

        # Initialize variables to store cumulative metrics for each fold
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        mccs = []
        f1_scores = []
        sensitivities = []
        specificities = []
        aucs = []
        auc_prs = []

        # Create the data loaders for this fold
        train_loader_fold = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_val_indices[train_indices])
        )
        val_loader_fold = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_val_indices[val_indices])
        )

        best_val_loss = float('inf')
        # Train the model for the specified number of epochs
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader_fold, criterion, optimizer)
            val_loss, val_acc, y_true, y_pred, y_score = test(model, val_loader_fold, criterion)
            # print(f"y true = {y_true} / y pred = {y_pred}")
            # Calculate additional metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            mcc = matthews_corrcoef(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            auc = roc_auc_score(y_true, y_score[:, 1])
            auc_pr = average_precision_score(y_true, y_score[:, 1])

            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}, "
                  f"MCC: {mcc:.4f}, F1 Score: {f1:.4f}, Specificity: {specificity:.4f}, "
                  f"Sensitivity: {sensitivity:.4f}, AUC: {auc:.4f}, AUC-PR:{auc_pr:.3f}")
            # Write the results to the CSV file
            writer.writerow({'fold': fold + 1, 'epoch': epoch + 1, 'train_loss': train_loss,
                             'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc,
                             'mcc': mcc, 'f1_score': f1, 'specificity': specificity,
                             'sensitivity': sensitivity, 'auc': auc, 'auc_pr': auc_pr})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'resnet_{fold + 1}.pt')

            # Store the the metrics for this fold and epoch
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            mccs.append(mcc)
            f1_scores.append(f1)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            aucs.append(auc)
            auc_prs.append(auc_pr)
            # Calculate the average metrics across all folds
        avg_train_loss = sum(train_losses) / num_epochs
        avg_train_acc = sum(train_accs) / num_epochs
        avg_val_loss = sum(val_losses) / num_epochs
        avg_val_acc = sum(val_accs) / num_epochs
        avg_mcc = sum(mccs) / num_epochs
        avg_f1 = sum(f1_scores) / num_epochs
        avg_sn = sum(sensitivities) / num_epochs
        avg_sp = sum(specificities) / num_epochs
        avg_auc = sum(aucs) / num_epochs
        avg_auc_pr = sum(auc_prs) / num_epochs
        # Test the model
        # Write the results to the CSV file
        writer.writerow({'fold': fold + 1, 'epoch': '', 'train_loss': avg_train_loss,
                         'train_acc': avg_train_acc, 'val_loss': avg_val_loss, 'val_acc': avg_val_acc,
                         'mcc': avg_mcc, 'f1_score': avg_f1, 'specificity': avg_sp,
                         'sensitivity': avg_sn, 'auc': avg_auc, 'auc_pr': avg_auc_pr})

        model.load_state_dict(torch.load(f'resnet_{fold + 1}.pt'))

        test_loss, test_acc, y_true, y_pred, y_score = test(model, test_loader, criterion)

        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        auc = roc_auc_score(y_true, y_score[:, 1])
        auc_pr = average_precision_score(y_true, y_score[:, 1])

        # Calculate the fpr, tpr, and roc_auc for the fold
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])  # Assuming binary classification
        roc_auc = sklearn_auc(fpr, tpr)

        # Plot the ROC curve for the fold
        plot_roc_curve(fpr, tpr, roc_auc, fold)
        # Store the fold results
        results.append((fpr, tpr, roc_auc))

        print(f"Fold {fold + 1} "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
              f"MCC: {mcc:.4f}, F1 Score: {f1:.4f}, Specificity: {specificity:.4f}, "
              f"Sensitivity: {sensitivity:.4f}, AUC: {auc:.4f}, AUC-PR:{auc_pr:.3f}")


        # Write the results to the CSV file
        writer.writerow({'fold': fold + 1, 'epoch': '', 'train_loss': '',
                         'train_acc': '', 'val_loss': test_loss, 'val_acc': test_acc,
                         'mcc': mcc, 'f1_score': f1, 'specificity': specificity,
                         'sensitivity': sensitivity, 'auc': auc, 'auc_pr': auc_pr})
    # Customize the figure
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-fold Cross-Validation')
    plt.legend()

    # Save the figure
    plt.savefig('ROC_curve.png')
    # Display the figure
    plt.show()