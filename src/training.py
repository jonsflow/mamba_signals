"""Training loop and utilities."""
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json

from config import Config
from mamba_model import create_model


class Trainer:
    """Handles model training, validation, and checkpointing."""

    def __init__(self, config: Config, model_dir: Path = None):
        """Initialize trainer.

        Args:
            config: Configuration object
            model_dir: Directory to save models and logs
        """
        self.config = config
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model_dir.mkdir(exist_ok=True)

        # Device - M1/M2/M3 Mac support with MPS, fallback to CPU
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.config.training.device = "mps"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.config.training.device = "cuda"
        else:
            self.device = torch.device("cpu")
            self.config.training.device = "cpu"

        # Model
        self.model = create_model(config.model).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': correct / total
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1),
                    'acc': correct / total
                })

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def fit(self, train_data: Tuple[np.ndarray, np.ndarray],
            val_data: Tuple[np.ndarray, np.ndarray]):
        """Train model.

        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )

        # Training loop
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)

            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint()

        # Save final model
        self.save_checkpoint(is_final=True)

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model
            is_final: Whether this is the final model
        """
        if is_best:
            path = self.model_dir / "best_model.pt"
        elif is_final:
            path = self.model_dir / "final_model.pt"
        else:
            path = self.model_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.dict(),
            'history': self.training_history
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['history']
        print(f"Loaded checkpoint from: {path}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Saved training history: {history_path}")
