import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
from collections import Counter

# Enable reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# --- DATASET HANDLING ---

class AdvancedFlowDataset(Dataset):
    def __init__(self, X, y, preceding_flows=None, transforms=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transforms = transforms
        
        # If preceding flows are not provided, use the same flows
        if preceding_flows is None:
            self.preceding_flows = self.X
        else:
            self.preceding_flows = torch.FloatTensor(preceding_flows)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Get random preceding flow
        prec_idx = idx if idx < len(self.preceding_flows) else random.randint(0, len(self.preceding_flows)-1)
        
        current_flow = self.X[idx]
        preceding_flow = self.preceding_flows[prec_idx]
        
        # Apply transformations if any
        if self.transforms:
            current_flow = self.transforms(current_flow)
            preceding_flow = self.transforms(preceding_flow)
        
        return {
            'current_flow': current_flow,
            'preceding_flow': preceding_flow,
            'label': self.y[idx]
        }

# --- DATA AUGMENTATION ---

class FlowAugmentations:
    @staticmethod
    def gaussian_noise(flow, std=0.01):
        return flow + torch.randn_like(flow) * std
    
    @staticmethod
    def time_mask(flow, num_masks=1, mask_len=10):
        result = flow.clone()
        seq_len = len(flow)
        for _ in range(num_masks):
            start = random.randint(0, seq_len - mask_len)
            result[start:start+mask_len] = 0
        return result
    
    @staticmethod
    def combined_transform(flow, p=0.5):
        if random.random() < p:
            flow = FlowAugmentations.gaussian_noise(flow)
        if random.random() < p:
            flow = FlowAugmentations.time_mask(flow)
        return flow

# --- ADVANCED MODELS FROM PAPER ---

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

# CNN Classifier from TrojanFlow paper
class AdvancedTrafficClassifier(nn.Module):
    def __init__(self, input_size=256, num_classes=7):  # Modified for top classes
        super(AdvancedTrafficClassifier, self).__init__()
        
        # Multi-scale feature extraction
        self.conv_small = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv_medium = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv_large = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Combined feature size after pooling
        feature_size = input_size // 2
        
        # Deep convolutional layers
        self.deep_conv = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for 1D convolution [batch, 1, seq_len]
        x = x.unsqueeze(1)
        
        # Multi-scale feature extraction
        x_small = self.conv_small(x)
        x_medium = self.conv_medium(x)
        x_large = self.conv_large(x)
        
        # Concatenate multi-scale features
        x_concat = torch.cat([x_small, x_medium, x_large], dim=1)
        
        # Apply deep convolutional layers
        x = self.deep_conv(x_concat)
        
        # Reshape to [batch, features]
        x = x.view(batch_size, -1)
        
        # Classification
        output = self.classifier(x)
        
        return output

# GRU Classifier from TrojanFlow paper
class FSNetGRUClassifier(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, num_classes=7, num_layers=2):  # Modified for top classes
        super(FSNetGRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=1,  # Each timestep has only 1 feature (packet size)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # Bidirectional gives 2x hidden size
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def attention_net(self, x):
        # x shape: [batch, seq_len, hidden_size*2]
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights to get context vector
        context = torch.sum(x * attn_weights, dim=1)  # [batch, hidden_size*2]
        return context
    
    def forward(self, x):
        # Input shape: [batch, seq_len]
        batch_size = x.size(0)
        
        # Add channel dimension for GRU
        x = x.unsqueeze(2)  # [batch, seq_len, 1]
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_size*2]
        
        # Apply attention mechanism
        context = self.attention_net(gru_out)  # [batch, hidden_size*2]
        
        # Classification
        output = self.classifier(context)  # [batch, num_classes]
        
        return output

# --- TRAINING AND EVALUATION ---

def train_model(model, train_loader, val_loader, device, num_epochs=150, lr=0.0005, weight_decay=1e-5):
    """Train a model with early stopping based on validation performance"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = batch['current_flow'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, history

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['current_flow'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    
    return epoch_loss, epoch_acc

def test_model(model, test_loader, device):
    """Test model on the test set and return class-wise metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch['current_flow'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)
    accuracy = correct / total
    
    # Calculate class-wise accuracy
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracy = {}
    for i in range(len(cm)):
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        if class_total > 0:
            class_accuracy[i] = class_correct / class_total
        else:
            class_accuracy[i] = 0.0
    
    return accuracy, class_accuracy, cm

# --- VISUALIZATION FUNCTIONS ---

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with matplotlib and seaborn"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# --- MAIN FUNCTION ---

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    pickle_file = '/scratch/lpanch2/Attack/iscxvpn2016_preprocessed.pkl'
    if not os.path.exists(pickle_file):
        pickle_file = 'iscxvpn2016_preprocessed.pkl'
    
    print(f"Loading data from {pickle_file}...")
    
    with open(pickle_file, 'rb') as f:
        application_flows = pickle.load(f)
    
    # Keep only high-performing classes:
    # Strong classes (>80%): FacebookVideo (4), Hangouts (5), SFTP (9), 
    # Torrent (12), SCP (8), Vimeo (13), Netflix (7)
    # Medium classes: Spotify (11), YouTube (15)
    
    high_performing_classes = {
        'FacebookVideo': 4,
        'Hangouts': 5,
        'SFTP': 9,
        'Torrent': 12,
        'SCP': 8,
        'Vimeo': 13,
        'Netflix': 7,
        'Spotify': 11,
        'YouTube': 15
    }
    
    # Create filtered dataset
    filtered_applications = {}
    for app_name, app_idx in high_performing_classes.items():
        for original_app, original_idx in application_flows.items():
            if original_app == app_name:
                filtered_applications[app_name] = application_flows[app_name]
    
    # Create new class mapping for the filtered dataset
    class_mapping = {app: i for i, app in enumerate(sorted(filtered_applications.keys()))}
    inv_class_mapping = {i: app for app, i in class_mapping.items()}
    
    print("\nFiltered Class mapping:")
    for app, idx in class_mapping.items():
        print(f"{app} -> {idx}")
    
    # Extract features
    max_per_class = 6000  # Increase samples per class
    X, y = [], []
    
    for app, flows in filtered_applications.items():
        # Limit per class for balance
        flows = flows[:max_per_class]
        
        # Extract packet sizes
        flow_arrays = []
        for flow in flows:
            packet_sizes = [p[0] for p in flow]
            flow_arrays.append(np.array(packet_sizes))
        
        # Add to dataset
        X.extend(flow_arrays)
        y.extend([class_mapping[app]] * len(flow_arrays))
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Print class distribution
    class_counts = Counter(y)
    print("\nClass distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"{inv_class_mapping[label]}: {count} samples")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    # Print dataset splits
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Apply data augmentation
    train_transform = lambda x: FlowAugmentations.combined_transform(x, p=0.6)
    
    # Create datasets with data augmentation
    train_dataset = AdvancedFlowDataset(X_train, y_train, transforms=train_transform)
    val_dataset = AdvancedFlowDataset(X_val, y_val)
    test_dataset = AdvancedFlowDataset(X_test, y_test)
    
    # Create weighted sampler to address class imbalance
    class_weights = [1.0 / class_counts[i] for i in y_train]
    sample_weights = [class_weights[i] for i in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model - choose either CNN or GRU
    model_type = "CNN"  # Change to "GRU" to use the GRU model
    
    if model_type == "CNN":
        model = AdvancedTrafficClassifier(input_size=X_train.shape[1], num_classes=len(class_mapping)).to(device)
        print("\nUsing CNN model architecture")
    else:
        model = FSNetGRUClassifier(input_size=X_train.shape[1], num_classes=len(class_mapping)).to(device)
        print("\nUsing GRU model architecture")
    
    # Print model architecture
    print(model)
    
    # Training
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=150,
        lr=0.0005,
        weight_decay=1e-5
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    accuracy, class_accuracy, cm = test_model(model, test_loader, device)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClass-wise Accuracy:")
    for class_idx, acc in class_accuracy.items():
        print(f"{inv_class_mapping[class_idx]}: {acc:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, [inv_class_mapping[i] for i in range(len(class_mapping))])
    
    # Save model
    model_save_path = f"traffic_classifier_{model_type}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_mapping': class_mapping,
        'inv_class_mapping': inv_class_mapping,
        'model_type': model_type,
        'input_size': X_train.shape[1],
    }, model_save_path)
    
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    main()