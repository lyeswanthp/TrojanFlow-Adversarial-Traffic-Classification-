import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import random

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

class TrojanFlowDataset(Dataset):
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

# --- MODELS ---

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

# Flow Classifier (based on the TrojanFlow paper)
class AdvancedTrafficClassifier(nn.Module):
    def __init__(self, input_size=256, num_classes=9):  # Modified for our filtered classes
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
        
        # Feature size after pooling
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

# Trigger Generator for TrojanFlow
class TriggerGenerator(nn.Module):
    def __init__(self, flow_length=256, hidden_dim=128, num_classes=9):
        super(TriggerGenerator, self).__init__()
        
        self.flow_length = flow_length
        self.combined_length = 2 * flow_length
        self.hidden_dim = hidden_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.combined_length, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
        )
        
        # Class conditioning
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.class_mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
        )
        
        # Feature processing with 1D convolutions
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            ResidualBlock(32),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * flow_length, self.combined_length),
            nn.Tanh()  # Control the trigger magnitude
        )
    
    def forward(self, preceding_flow, current_flow, target_class=None):
        batch_size = current_flow.size(0)
        
        # Combine flows
        combined = torch.cat((preceding_flow, current_flow), dim=1)
        
        # Initial embedding
        x = self.encoder(combined)
        
        # Add class conditioning if provided
        if target_class is not None:
            # Convert to LongTensor if it's not already
            if not isinstance(target_class, torch.Tensor):
                target_class = torch.tensor([target_class] * batch_size, device=combined.device)
            
            # Get class embeddings
            class_embed = self.class_embed(target_class)
            
            # Split embedding for processing
            x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
            
            # Mix with class embedding
            x = self.class_mixer(torch.cat([x1, x2, class_embed], dim=1))
        
        # Reshape for 1D CNN - first check dimensions to ensure correct reshaping
        feature_dim = x.size(1)
        channels = 2  # We want 2 channels
        seq_len = feature_dim // channels  # Calculate sequence length
        
        # Ensure the reshape operation is valid
        if feature_dim != channels * self.flow_length:
            # Adjust feature dimension with a projection layer if needed
            x = nn.Linear(feature_dim, channels * self.flow_length).to(x.device)(x)
        
        # Now reshape to [batch, channels, seq_len]
        x = x.view(batch_size, channels, -1)
        
        # Apply convolution blocks
        x = self.conv_blocks(x)
        
        # Final projection to trigger mask
        trigger_mask = self.projection(x)
        
        return trigger_mask

# --- TROJANFLOW ATTACK IMPLEMENTATION ---

class TrojanFlowAttack:
    def __init__(self, generator, classifier, device='cuda', lambda_l2=0.001, target_class=0):
        self.generator = generator.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.lambda_l2 = lambda_l2  # Increased L2 regularization (was 0.0001)
        self.target_class = target_class
        self.max_packet_size = 1514
        
        # Optimizers with AdamW (better weight decay handling)
        self.optimizer_g = optim.AdamW(self.generator.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR, higher weight decay
        self.optimizer_c = optim.AdamW(self.classifier.parameters(), lr=0.0001, weight_decay=1e-4)  # Much lower LR for classifier
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_g, T_0=10, T_mult=2)
        self.scheduler_c = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_c, T_0=10, T_mult=2)
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'classifier_loss': [], 'generator_loss': [],
            'clean_accuracy': [], 'backdoor_success': [],
            'epoch': []
        }
        
        # Keep a copy of original classifier for evaluation
        self.original_classifier = type(classifier)(
            input_size=classifier.conv_small[0].kernel_size[0],
            num_classes=classifier.classifier[-1].out_features
        ).to(device)
        self.original_classifier.load_state_dict(classifier.state_dict())
    
    def train_classifier(self, train_loader, num_epochs=15):
        """Pre-train the classifier on clean data"""
        self.classifier.train()
        
        print("Pre-training classifier...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct, total = 0, 0
            
            for batch in tqdm(train_loader, desc=f"Classifier Training {epoch+1}/{num_epochs}"):
                current_flow = batch['current_flow'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer_c.zero_grad()
                
                # Forward pass
                outputs = self.classifier(current_flow)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer_c.step()
                
                # Update metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
            # Update scheduler
            self.scheduler_c.step()
            
            # Print metrics
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    def train_generator(self, train_loader, num_epochs=30):  # Increased epochs
        """Pre-train the generator for effective trigger generation"""
        self.generator.train()
        self.classifier.eval()
        
        print("Pre-training generator...")
        
        # Add L1 regularization to promote sparsity
        lambda_l1 = 0.0005
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            backdoor_success = 0
            total_samples = 0
            
            for batch in tqdm(train_loader, desc=f"Generator Training {epoch+1}/{num_epochs}"):
                current_flow = batch['current_flow'].to(self.device)
                preceding_flow = batch['preceding_flow'].to(self.device)
                batch_size = current_flow.size(0)
                
                self.optimizer_g.zero_grad()
                
                # Generate triggers with target class conditioning
                trigger_mask = self.generator(preceding_flow, current_flow, self.target_class)
                
                # Extract trigger for current flow
                trigger_current = trigger_mask[:, preceding_flow.size(1):]
                
                # Create poisoned samples
                poisoned_flow = torch.clamp(current_flow + trigger_current, min=0, max=self.max_packet_size)
                
                # Forward pass with poisoned data
                with torch.no_grad():
                    poisoned_outputs = self.classifier(poisoned_flow)
                
                # Target labels
                target_labels = torch.full((batch_size,), self.target_class, dtype=torch.long, device=self.device)
                
                # Calculate backdoor loss - maximize target class probability
                backdoor_loss = -F.cross_entropy(poisoned_outputs, target_labels)
                
                # L2 regularization on trigger magnitude
                l2_reg = self.lambda_l2 * torch.norm(trigger_current, p=2, dim=1).mean()
                
                # L1 regularization to promote sparsity
                l1_reg = lambda_l1 * torch.norm(trigger_current, p=1, dim=1).mean()
                
                # Ensure imperceptibility of trigger
                imperceptibility_loss = 0.01 * F.mse_loss(poisoned_flow, current_flow)
                
                # Combined loss
                loss = backdoor_loss + l2_reg + l1_reg + imperceptibility_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)  # Add gradient clipping
                self.optimizer_g.step()
                
                # Update metrics
                running_loss += loss.item()
                _, predicted = torch.max(poisoned_outputs.data, 1)
                backdoor_success += (predicted == self.target_class).sum().item()
                total_samples += batch_size
                
            # Update scheduler
            self.scheduler_g.step()
            
            # Print metrics
            success_rate = 100 * backdoor_success / total_samples
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Success Rate: {success_rate:.2f}%")
            
            # Early stopping if very effective
            if success_rate > 95 and epoch > 20:
                print(f"Generator is already very effective. Early stopping.")
                break
    
    def joint_training(self, train_loader, test_loader, num_epochs=60, poison_ratio=0.25):  # Reduced poison ratio
        """Joint training of generator and classifier with evaluation checkpoints"""
        print("Starting joint training...")
        
        best_asr = 0
        best_balanced_score = 0
        best_epoch = 0
        
        # Track original parameters
        clean_checkpoint = {k: v.clone() for k, v in self.classifier.state_dict().items()}
        
        # Store best model weights
        best_generator_weights = None
        best_classifier_weights = None
        
        for epoch in range(num_epochs):
            self.generator.train()
            self.classifier.train()
            
            running_c_loss = 0.0
            running_g_loss = 0.0
            clean_correct, clean_total = 0, 0
            backdoor_success, backdoor_total = 0, 0
            
            # Very slow increase in poison ratio to maintain clean performance
            current_poison_ratio = min(poison_ratio, poison_ratio * (epoch + 1) / 40)
            
            # Balance factor between clean and poisoned loss
            # Start with emphasis on clean accuracy, then gradually increase backdoor influence
            backdoor_weight = min(1.0, 0.2 + 0.8 * epoch / 30)
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Joint Training {epoch+1}/{num_epochs}")):
                current_flow = batch['current_flow'].to(self.device)
                preceding_flow = batch['preceding_flow'].to(self.device)
                labels = batch['label'].to(self.device)
                batch_size = current_flow.size(0)
                
                # Determine which samples to poison
                poison_mask = torch.zeros(batch_size, dtype=torch.bool)
                num_poison = int(batch_size * current_poison_ratio)
                poison_indices = random.sample(range(batch_size), min(num_poison, batch_size))
                poison_mask[poison_indices] = True
                poison_mask = poison_mask.to(self.device)
                
                # --- Train Classifier ---
                self.optimizer_c.zero_grad()
                
                # Generate triggers
                with torch.no_grad():
                    trigger_mask = self.generator(preceding_flow, current_flow, self.target_class)
                
                # Extract trigger for current flow
                trigger_current = trigger_mask[:, preceding_flow.size(1):]
                
                # Create poisoned samples
                poisoned_flow = current_flow.clone()
                poisoned_flow[poison_mask] = torch.clamp(
                    current_flow[poison_mask] + trigger_current[poison_mask], 
                    min=0, 
                    max=self.max_packet_size
                )
                
                # Forward passes
                clean_outputs = self.classifier(current_flow)
                clean_loss = self.criterion(clean_outputs, labels)
                
                poisoned_labels = labels.clone()
                poisoned_labels[poison_mask] = self.target_class
                poisoned_outputs = self.classifier(poisoned_flow)
                poisoned_loss = self.criterion(poisoned_outputs, poisoned_labels)
                
                # Combined loss with dynamic weighting
                # Prioritize clean accuracy with higher weight
                c_loss = (1.0 - backdoor_weight/2) * clean_loss + backdoor_weight * poisoned_loss
                
                # Weight decay regularization to stay closer to original model
                if epoch < 20:  # Only in early epochs
                    param_diffs = 0
                    for name, param in self.classifier.named_parameters():
                        if name in clean_checkpoint:
                            param_diffs += torch.norm(param - clean_checkpoint[name])
                    c_loss += 0.001 * param_diffs
                
                c_loss.backward()
                # Gradient clipping to prevent dramatic changes to the classifier
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=0.5)
                self.optimizer_c.step()
                
                # Update metrics
                _, clean_predicted = torch.max(clean_outputs.data, 1)
                _, poisoned_predicted = torch.max(poisoned_outputs.data, 1)
                running_c_loss += c_loss.item()
                clean_correct += (clean_predicted == labels).sum().item()
                clean_total += labels.size(0)
                backdoor_success += (poisoned_predicted[poison_mask] == self.target_class).sum().item()
                backdoor_total += poison_mask.sum().item() if poison_mask.sum().item() > 0 else 1
                
                # --- Train Generator ---
                # Train generator less frequently in early epochs
                if epoch > 5 or batch_idx % 2 == 0:
                    self.optimizer_g.zero_grad()
                    
                    # Generate triggers
                    trigger_mask = self.generator(preceding_flow, current_flow, self.target_class)
                    trigger_current = trigger_mask[:, preceding_flow.size(1):]
                    
                    # Create poisoned samples
                    poisoned_flow = torch.clamp(current_flow + trigger_current, min=0, max=self.max_packet_size)
                    
                    # Forward pass
                    poisoned_outputs = self.classifier(poisoned_flow)
                    
                    # Target labels
                    target_labels = torch.full((batch_size,), self.target_class, dtype=torch.long, device=self.device)
                    
                    # Maximize target class probability
                    backdoor_loss = -self.criterion(poisoned_outputs, target_labels)
                    
                    # Regularization
                    l2_reg = self.lambda_l2 * torch.norm(trigger_current, p=2, dim=1).mean()
                    l1_reg = 0.0005 * torch.norm(trigger_current, p=1, dim=1).mean()
                    
                    # Minimize visibility of trigger (imperceptibility)
                    imperceptibility_loss = 0.01 * F.mse_loss(poisoned_flow, current_flow)
                    
                    # Combined loss
                    g_loss = backdoor_loss + l2_reg + l1_reg + imperceptibility_loss
                    
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    self.optimizer_g.step()
                    
                    running_g_loss += g_loss.item()
            
            # Update schedulers
            self.scheduler_c.step()
            self.scheduler_g.step()
            
            # Calculate epoch metrics
            epoch_c_loss = running_c_loss / len(train_loader)
            epoch_g_loss = running_g_loss / len(train_loader)
            epoch_clean_acc = 100 * clean_correct / max(1, clean_total)
            epoch_backdoor_success = 100 * backdoor_success / max(1, backdoor_total)
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['classifier_loss'].append(epoch_c_loss)
            self.history['generator_loss'].append(epoch_g_loss)
            self.history['clean_accuracy'].append(epoch_clean_acc)
            self.history['backdoor_success'].append(epoch_backdoor_success)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Classifier Loss: {epoch_c_loss:.4f}, Generator Loss: {epoch_g_loss:.4f}")
            print(f"Clean Accuracy: {epoch_clean_acc:.2f}%, Backdoor Success: {epoch_backdoor_success:.2f}%")
            
            # Evaluate on test set every few epochs
            if epoch % 3 == 0 or epoch == num_epochs - 1:
                test_results = self.evaluate(test_loader)
                test_clean_acc = test_results['clean_accuracy']
                test_asr = test_results['attack_success_rate']
                
                print(f"Test Clean Accuracy: {test_clean_acc:.2f}%, Test ASR: {test_asr:.2f}%")
                
                # Balanced score - we want both high clean accuracy and high ASR
                balanced_score = test_clean_acc * 0.6 + test_asr * 0.4
                
                if balanced_score > best_balanced_score and test_clean_acc > 85 and test_asr > 70:
                    best_balanced_score = balanced_score
                    best_asr = test_asr
                    best_epoch = epoch + 1
                    best_generator_weights = {k: v.clone() for k, v in self.generator.state_dict().items()}
                    best_classifier_weights = {k: v.clone() for k, v in self.classifier.state_dict().items()}
                    self.save_checkpoint(f"trojanflow_best_e{epoch+1}_acc{test_clean_acc:.1f}_asr{test_asr:.1f}.pt")
                    print(f"New best model! Balanced score: {balanced_score:.2f}")
            
            print("-" * 60)
            
            # Early stopping
            if epoch_backdoor_success > 95.0 and epoch_clean_acc > 90.0 and epoch > 30:
                print(f"Reached target performance. Early stopping at epoch {epoch+1}")
                break
        
        # Load best model if found
        if best_generator_weights is not None:
            print(f"Loading best model from epoch {best_epoch}")
            self.generator.load_state_dict(best_generator_weights)
            self.classifier.load_state_dict(best_classifier_weights)
        
        print(f"Best model at epoch {best_epoch} with ASR {best_asr:.2f}%")
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.generator.eval()
        self.classifier.eval()
        
        clean_correct, clean_total = 0, 0
        backdoor_success, backdoor_total = 0, 0
        
        all_labels = []
        all_clean_preds = []
        all_poison_preds = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                current_flow = batch['current_flow'].to(self.device)
                preceding_flow = batch['preceding_flow'].to(self.device)
                labels = batch['label'].to(self.device)
                batch_size = current_flow.size(0)
                
                # Clean predictions
                clean_outputs = self.classifier(current_flow)
                _, clean_predicted = torch.max(clean_outputs.data, 1)
                
                # Generate triggers
                trigger_mask = self.generator(preceding_flow, current_flow, self.target_class)
                trigger_current = trigger_mask[:, preceding_flow.size(1):]
                
                # Create poisoned samples
                poisoned_flow = torch.clamp(current_flow + trigger_current, min=0, max=self.max_packet_size)
                
                # Poisoned predictions
                poisoned_outputs = self.classifier(poisoned_flow)
                _, poisoned_predicted = torch.max(poisoned_outputs.data, 1)
                
                # Update metrics
                clean_correct += (clean_predicted == labels).sum().item()
                clean_total += batch_size
                backdoor_success += (poisoned_predicted == self.target_class).sum().item()
                backdoor_total += batch_size
                
                # Store for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_clean_preds.extend(clean_predicted.cpu().numpy())
                all_poison_preds.extend(poisoned_predicted.cpu().numpy())
        
        # Calculate overall metrics
        clean_accuracy = 100 * clean_correct / clean_total
        attack_success_rate = 100 * backdoor_success / backdoor_total
        
        # Confusion matrices
        clean_cm = confusion_matrix(all_labels, all_clean_preds)
        poison_cm = confusion_matrix(all_labels, all_poison_preds)
        
        return {
            'clean_accuracy': clean_accuracy,
            'attack_success_rate': attack_success_rate,
            'clean_confusion_matrix': clean_cm,
            'poison_confusion_matrix': poison_cm
        }
    
    def compare_with_original(self, test_loader):
        """Compare current model with original clean model"""
        self.classifier.eval()
        self.original_classifier.eval()
        
        current_correct, original_correct, total = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Comparing with original"):
                current_flow = batch['current_flow'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Current model predictions
                current_outputs = self.classifier(current_flow)
                _, current_preds = torch.max(current_outputs, 1)
                
                # Original model predictions
                original_outputs = self.original_classifier(current_flow)
                _, original_preds = torch.max(original_outputs, 1)
                
                # Update metrics
                current_correct += (current_preds == labels).sum().item()
                original_correct += (original_preds == labels).sum().item()
                total += labels.size(0)
        
        current_acc = 100 * current_correct / total
        original_acc = 100 * original_correct / total
        
        print(f"Current model accuracy: {current_acc:.2f}%")
        print(f"Original model accuracy: {original_acc:.2f}%")
        print(f"Accuracy difference: {current_acc - original_acc:.2f}%")
        
        return current_acc, original_acc
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_c_state_dict': self.optimizer_c.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_c_state_dict': self.scheduler_c.state_dict(),
            'history': self.history,
            'target_class': self.target_class
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_c.load_state_dict(checkpoint['optimizer_c_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_c.load_state_dict(checkpoint['scheduler_c_state_dict'])
        self.history = checkpoint['history']
        self.target_class = checkpoint['target_class']
        print(f"Checkpoint loaded from {filename}")

# --- VISUALIZATION FUNCTIONS ---

def plot_training_history(history):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history['epoch'], history['clean_accuracy'], 'b-', label='Clean Accuracy')
    ax1.plot(history['epoch'], history['backdoor_success'], 'r-', label='Backdoor Success')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Accuracy & Attack Success Rate')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history['epoch'], history['classifier_loss'], 'g-', label='Classifier Loss')
    ax2.plot(history['epoch'], history['generator_loss'], 'm-', label='Generator Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('trojanflow_training_history.png')
    plt.show()

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def visualize_triggers(generator, classifier, test_loader, target_class, device='cuda', num_samples=5):
    """Visualize generated triggers on random samples"""
    generator.eval()
    classifier.eval()
    
    # Get random samples
    samples = []
    for batch in test_loader:
        current_flow = batch['current_flow'].to(device)
        preceding_flow = batch['preceding_flow'].to(device)
        labels = batch['label'].to(device)
        
        # Take first num_samples as examples
        if len(samples) < num_samples:
            for i in range(min(num_samples - len(samples), len(current_flow))):
                samples.append({
                    'current': current_flow[i].cpu(),
                    'preceding': preceding_flow[i].cpu(),
                    'label': labels[i].item()
                })
        else:
            break
    
    # Generate triggers and visualize
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    
    for i, sample in enumerate(samples):
        current = sample['current'].to(device).unsqueeze(0)
        preceding = sample['preceding'].to(device).unsqueeze(0)
        true_label = sample['label']
        
        # Generate trigger
        with torch.no_grad():
            trigger_mask = generator(preceding, current, target_class)
            trigger = trigger_mask[0, preceding.size(1):].cpu().numpy()
            
            # Get predictions
            clean_out = classifier(current)
            _, clean_pred = torch.max(clean_out, 1)
            
            # Create poisoned flow
            poisoned = torch.clamp(current + trigger_mask[0, preceding.size(1):].unsqueeze(0), min=0, max=1514)
            poisoned_out = classifier(poisoned)
            _, poisoned_pred = torch.max(poisoned_out, 1)
        
        # Plot
        axes[i, 0].plot(current[0].cpu().numpy())
        axes[i, 0].set_title(f"Original - True: {true_label}, Pred: {clean_pred.item()}")
        
        axes[i, 1].plot(trigger)
        max_val = max(abs(trigger.max()), abs(trigger.min()))
        axes[i, 1].set_ylim(-max_val * 1.1, max_val * 1.1)
        axes[i, 1].set_title(f"Trigger (Target: {target_class})")
        
        axes[i, 2].plot(poisoned[0].cpu().numpy())
        axes[i, 2].set_title(f"Poisoned - Pred: {poisoned_pred.item()}")
    
    plt.tight_layout()
    plt.savefig('trojanflow_trigger_visualization.png')
    plt.show()

# --- MAIN FUNCTION ---

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load already trained classifier
    classifier_path = "traffic_classifier_CNN.pt"
    
    if not os.path.exists(classifier_path):
        print(f"Error: Trained classifier not found at {classifier_path}")
        print("Please run the improved classifier script first to train a model.")
        return
    
    # Load the classifier checkpoint
    checkpoint = torch.load(classifier_path)
    class_mapping = checkpoint['class_mapping']
    inv_class_mapping = checkpoint['inv_class_mapping']
    num_classes = len(class_mapping)
    input_size = checkpoint['input_size']
    
    # Initialize classifier and load weights
    classifier = AdvancedTrafficClassifier(input_size=input_size, num_classes=num_classes).to(device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded classifier with {num_classes} classes")
    print("Class mapping:", inv_class_mapping)
    
    # Choose target class
    # For this example, we'll use the first class as the target
    target_class = 0  # Change this to any class index you prefer
    target_class_name = inv_class_mapping[target_class]
    print(f"Target class chosen: {target_class} ({target_class_name})")
    
    # Load dataset
    pickle_file = '/scratch/lpanch2/Attack/iscxvpn2016_preprocessed.pkl'
    if not os.path.exists(pickle_file):
        pickle_file = 'iscxvpn2016_preprocessed.pkl'
    
    print(f"Loading data from {pickle_file}...")
    
    with open(pickle_file, 'rb') as f:
        application_flows = pickle.load(f)
    
    # Keep only high-performing classes (same as in classifier training)
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
    
    # Extract features
    max_per_class = 5000  # Limit samples per class
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
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create datasets
    train_dataset = TrojanFlowDataset(X_train, y_train)
    test_dataset = TrojanFlowDataset(X_test, y_test)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize the trigger generator
    generator = TriggerGenerator(flow_length=input_size, hidden_dim=128, num_classes=num_classes).to(device)
    
    # Reduce the number of workers to avoid warnings
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create TrojanFlow trainer
    trojan_flow = TrojanFlowAttack(
        generator=generator,
        classifier=classifier,
        device=device,
        lambda_l2=0.0001,
        target_class=target_class
    )
    
    # First evaluate the clean model
    print("\nEvaluating clean model...")
    clean_eval = trojan_flow.evaluate(test_loader)
    print(f"Clean Accuracy: {clean_eval['clean_accuracy']:.2f}%")
    print(f"Attack Success Rate (before attack): {clean_eval['attack_success_rate']:.2f}%")
    
    # Train the generator
    print("\nTraining the trigger generator...")
    trojan_flow.train_generator(train_loader, num_epochs=25)
    
    # Joint training 
    print("\nStarting joint training...")
    history = trojan_flow.joint_training(train_loader, test_loader, num_epochs=60, poison_ratio=0.5)
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_results = trojan_flow.evaluate(test_loader)
    print(f"Clean Accuracy: {eval_results['clean_accuracy']:.2f}%")
    print(f"Attack Success Rate: {eval_results['attack_success_rate']:.2f}%")
    
    # Visualize results
    plot_training_history(trojan_flow.history)
    
    plot_confusion_matrix(
        eval_results['clean_confusion_matrix'], 
        [inv_class_mapping[i] for i in range(num_classes)],
        title="Clean Confusion Matrix"
    )
    
    plot_confusion_matrix(
        eval_results['poison_confusion_matrix'], 
        [inv_class_mapping[i] for i in range(num_classes)],
        title="Poisoned Confusion Matrix"
    )
    
    # Visualize triggers
    visualize_triggers(trojan_flow.generator, trojan_flow.classifier, 
                      test_loader, trojan_flow.target_class, device)
    
    # Save final model
    trojan_flow.save_checkpoint("trojanflow_final.pt")
    
    print("\nTrojanFlow attack completed!")

if __name__ == "__main__":
    main()