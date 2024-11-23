import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import numpy as np
from joblib import dump, load
from glob import glob
import json
import os
import torch
from logger import TensorLogger
import torch.optim as optim
from config import Config


# Model hyperparameters
class HyperParams:
    def __init__(self):
        self.T=30
        self.input_dim=1
        self.hidden_dim=64
        self.num_layers=2
        self.output_dim=2
        self.batch_size = 32
        self.dropout = 0.1
        self.lr = 0.001

class SimpleNN(nn.Module):
    def __init__(self, cfg: Config, params: HyperParams):
        super(SimpleNN, self).__init__()
        self.hyperparams = params
        
        # LSTM layer
        self.lstm = nn.LSTM(self.hyperparams.input_dim, self.hyperparams.hidden_dim, self.hyperparams.num_layers, batch_first=True)
        
        # Fully connected layer to downscale to output dimension
        self.fc1 = nn.Linear(self.hyperparams.hidden_dim, 32)
        self.fc2 = nn.Linear(32, 2)
    
        self.bn1 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(self.hyperparams.dropout)
        
        self._initialize_weights()
        self._setup_general(cfg.ckpt_path, cfg.log_dir, lr=self.hyperparams.lr)

    def _initialize_weights(self):
        # Kaiming initialization often works better with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, X_etc, X_emb=None):
        # Main path
        _, (h_n, _) = self.lstm(X_etc)
        
        # x = torch.cat([x_etc, x_emb], dim=1)
        
        x = self.fc1(h_n[-1])
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
    
    def _setup_general(self, ckp_dir, log_dir, lr):
        # Setup checkpointing
        self.logger = TensorLogger(logdir=log_dir)
        self.checkpoint_dir = ckp_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        
    def setup_criterion(self, class_weights = None):
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def setup4newfit(self):
        self.best_acc = float(0)
        self.best_model_state = None
        self.n_trained_epochs = 0
        self.global_step = 1
        
    def fit(self, train_loader, val_loader, n_epochs=100):
        # Reduce LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
        )

        for epoch in range(n_epochs + self.n_trained_epochs)[self.n_trained_epochs:]:
            self.train()
            train_loss = 0.0
            for X_etc, Y in train_loader:
                self.optimizer.zero_grad()
                outputs = self(X_etc)
                loss = self.criterion(outputs, Y.reshape(-1))
                loss.backward()
                self.logger.log_ud(self, self.global_step, self.lr)
                self.optimizer.step()
                self.global_step += 1
                train_loss += loss.item() * Y.size(0)

            train_loss = train_loss / len(train_loader.dataset)

            # Validation loop
            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for X_etc, Y in val_loader:
                    outputs = self(X_etc)
                    loss = self.criterion(outputs, Y.reshape(-1))
                    val_loss += loss.item() * Y.size(0)
                    _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                    total += Y.size(0)
                    correct += (predicted == Y.reshape(-1)).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(val_loss)
            val_accuracy = correct / total

            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            self.logger.log_metric(train_loss, self.global_step)
            self.logger.log_metric(val_loss, self.global_step, mode="Val")
            self.logger.log_metric(val_accuracy, self.global_step, metric="acc", mode="Val")

            # Save the best model
            if val_accuracy > self.best_acc:
                self.best_acc = val_accuracy
                # Save only the model's state dict
                self.best_model_state = self.state_dict()
                checkpoint_path = self._save_checkpoint(epoch, val_accuracy)
                print(f"New best model saved: {checkpoint_path}")

    
    def initialize_from_ckp(self):
        """Initialize a new model or load the latest checkpoint if available"""
        checkpoint_info = self._find_latest_checkpoint()
        
        if checkpoint_info:
            print(f"Resuming from checkpoint: {checkpoint_info['checkpoint_path']}")
            self._load_checkpoint(checkpoint_info['checkpoint_path'])
        else:
            self.setup4newfit()
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the checkpoint directory"""
        # Look for checkpoint files
        checkpoint_files = glob(os.path.join(self.checkpoint_dir, 'model_*.pth'))
        
        if not checkpoint_files:
            return None
            
        # Get the latest checkpoint based on modification time
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        return {
            'checkpoint_path': latest_checkpoint
        }
    
    def _save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint and training state"""
        # Save model
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'model_epoch_{epoch}_acc_{accuracy:.4f}.pth'
        )
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'acc': accuracy,
            'global_step': self.global_step,
        }
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model and training state from checkpoint"""
        # Loading full checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_trained_epochs = checkpoint['epoch']
        self.best_acc = checkpoint['acc']
        self.global_step = checkpoint['global_step']
        self.best_model_state = self.state_dict()
    
    def _cleanup_old_checkpoints(self, keep_n=5):
        """Keep only the N most recent checkpoints"""
        checkpoint_files = glob(os.path.join(self.checkpoint_dir, 'model_*.pth'))
        if len(checkpoint_files) <= keep_n:
            return
            
        # Sort checkpoints by modification time
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Remove older checkpoints
        for checkpoint_file in checkpoint_files[keep_n:]:
            os.remove(checkpoint_file)
    
