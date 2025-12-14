import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from tqdm import tqdm

from utils import setup_logger
from config import config

logger = setup_logger()


class OrdinalRegressionLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) loss for ordinal regression.
    Treats the problem as K-1 binary classifications for K classes.
    Paper: https://arxiv.org/pdf/1901.07884
    """
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes-1) raw outputs
            targets: (batch_size,) class indices (0 to num_classes-1)
        """
        # Create ordinal binary labels: 1 if target > threshold, 0 otherwise
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        binary_labels = (targets.unsqueeze(1) > levels).float()
        
        loss = F.binary_cross_entropy_with_logits(logits, binary_labels, reduction='none')
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[targets].unsqueeze(1)
            loss = loss * weights
        
        return loss.sum(dim=1).mean()

class CoralLayer(nn.Module):
    """
    Implements the specific CORAL output layer: 
    Shared weights (1 neuron) + K-1 independent biases.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        # 1. Shared Weight: Single output neuron (rank score)
        self.linear = nn.Linear(in_features, 1, bias=False)
        # 2. Independent Biases: One for each threshold
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        
    def forward(self, x):
        # x: (batch, in_features)
        # linear output: (batch, 1) -> Broadcasts to (batch, num_classes-1)
        # biases: (num_classes-1)
        return self.linear(x) + self.biases

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=5, dropout=0.3, ordinal=False):
        super().__init__()
        self.ordinal = ordinal
        self.num_classes = num_classes
        
        # --- Feature Extractor ---
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        
        # --- Output Layer ---
        if self.ordinal:
            # STRICT CORAL: Shared weights, specific biases
            self.classifier = CoralLayer(prev_dim, num_classes)
        else:
            # STANDARD CLASSIFICATION
            self.classifier = nn.Linear(prev_dim, num_classes)
    
    def ordinal_to_probs(self, logits):
        """Convert ordinal logits to strictly valid class probabilities."""
        # 1. Sigmoid to get P(y > k)
        cum_probs = torch.sigmoid(logits)
        
        # 2. Add fixed boundaries: P(y > -1)=1.0 and P(y > K-1)=0.0
        batch_size = logits.shape[0]
        ones = torch.ones(batch_size, 1, device=logits.device)
        zeros = torch.zeros(batch_size, 1, device=logits.device)
        
        # Shape: (batch, num_classes+1)
        cum_probs_extended = torch.cat([ones, cum_probs, zeros], dim=1)
        
        # 3. Compute PDF: P(y=k) = P(y>k-1) - P(y>k)
        class_probs = cum_probs_extended[:, :-1] - cum_probs_extended[:, 1:]
        
        # 4. SAFETY: Clamp negative probabilities (restoring consistency)
        # Even with shared weights, violations can happen slightly during training
        class_probs = torch.clamp(class_probs, min=0.0)
        
        # 5. Normalize to sum to 1.0 after clamping
        class_probs = class_probs / (class_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        return class_probs
    
    def forward(self, x, return_probs=False):
        features = self.features(x)
        logits = self.classifier(features)
        
        if self.ordinal:
            if return_probs or not self.training:
                return self.ordinal_to_probs(logits)
            return logits # Return raw logits for CORAL Loss
        else:
            if return_probs:
                return F.softmax(logits, dim=1)
            return logits # Return raw logits for CrossEntropyLoss


class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings."""
    
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_data(data_dir=None):
    """Load train and val datasets."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    train_df = pd.read_csv(Path(data_dir) / 'train_sentiments.csv')
    val_df = pd.read_csv(Path(data_dir) / 'val_sentiments.csv')
    
    print(f"Train set: {len(train_df)} examples")
    print(f"Val set: {len(val_df)} examples")
    
    return train_df, val_df


def generate_embeddings(texts, model_name=None, batch_size=None, device=None):
    """
    Generate sentence embeddings using SentenceTransformer.
    
    NOTE: This function only GENERATES embeddings - the sentence transformer
    is NOT trained or fine-tuned. Embeddings are frozen representations.
    
    Args:
        texts: List of text strings
        model_name: SentenceTransformer model name (defaults to config)
        batch_size: Batch size for encoding (defaults to config)
        device: Device to use ('cuda' or 'cpu') (defaults to config)
    
    Returns:
        numpy array of embeddings (frozen, pre-computed)
    """
    if model_name is None:
        model_name = config.MLP_SENTENCE_MODEL
    if batch_size is None:
        batch_size = config.MLP_BATCH_SIZE
    if device is None:
        device = config.DEVICE
    
    print(f"\nLoading SentenceTransformer model: {model_name}")
    print("Note: Sentence transformer is FROZEN - only used for embedding generation")
    
    model = SentenceTransformer(model_name)
    model = model.to(device)
    model.eval()  # Set to eval mode
    
    # Save the sentence transformer model for later use during evaluation
    if not config.SENTENCE_TRANSFORMER_PATH.exists():
        print(f"Saving SentenceTransformer model to {config.SENTENCE_TRANSFORMER_PATH}...")
        config.SENTENCE_TRANSFORMER_PATH.parent.mkdir(exist_ok=True, parents=True)
        model.save(str(config.SENTENCE_TRANSFORMER_PATH))
        print("SentenceTransformer model saved successfully")
    
    print(f"Generating embeddings for {len(texts)} texts...")
    
    with torch.no_grad():  # No gradients needed - embeddings are frozen
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=device
        )
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("Embeddings generated and will be cached (not recomputed during training)")
    return embeddings


def train_epoch(model, dataloader, criterion, optimizer, device, ordinal=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for embeddings, labels in tqdm(dataloader, desc="Training"):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions based on loss type
        if ordinal:
            # For ordinal: count thresholds exceeded
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).sum(dim=1)
        else:
            # For standard classification
            _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, ordinal=False):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader, desc="Validation"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # For loss calculation, use training mode outputs
            if ordinal:
                logits = model.classifier(model.features(embeddings))
                loss = criterion(logits, labels)
                # For predictions, use probability distribution
                probs = model.ordinal_to_probs(logits)
                predicted = probs.argmax(dim=1)
            else:
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
            
            total_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    """Main training pipeline."""
    print("="*60)
    print("MLP TRAINING: Sentence Transformers + MLP")
    print("="*60)
    print("\nStarting MLP training pipeline...")
    print(f"Time: {pd.Timestamp.now()}")
    
    # Print hyperparameters from config
    print("\n" + "="*60)
    print("HYPERPARAMETERS")
    print("="*60)
    print(f"Sentence Transformer Model: {config.MLP_SENTENCE_MODEL}")
    print(f"  (Frozen - only for embedding generation)")
    print(f"Hidden Layer Dimensions:    {config.MLP_HIDDEN_DIMS}")
    print(f"Number of Classes:          {config.MLP_NUM_CLASSES}")
    print(f"Batch Size:                 {config.MLP_BATCH_SIZE}")
    print(f"Number of Epochs:           {config.MLP_EPOCHS}")
    print(f"Learning Rate:              {config.MLP_LEARNING_RATE}")
    print(f"Dropout Rate:               {config.MLP_DROPOUT}")
    print(f"Loss Function:              {'Ordinal Regression' if config.MLP_USE_ORDINAL_LOSS else 'CrossEntropy'}")
    print(f"Device:                     {config.DEVICE}")
    print(f"Output Directory:           {config.OUTPUT_DIR}")
    print("="*60)
    
    # Load data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    train_df, val_df = load_data()
    
    # Adjust labels (assuming 1-5, convert to 0-4)
    y_train = train_df['sentiment_choice'].values - 1
    y_val = val_df['sentiment_choice'].values - 1
    
    # Analyze class distribution
    print("\nClass Distribution Analysis:")
    print("-" * 40)
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(y_train)
        print(f"Class {cls}: {count:5d} samples ({pct:5.2f}%)")
    
    # Calculate class weights for imbalanced datasets
    class_weights = len(y_train) / (len(unique) * counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(config.DEVICE)
    print(f"\nClass weights: {class_weights}")
    max_imbalance = counts.max() / counts.min()
    print(f"Imbalance ratio: {max_imbalance:.2f}:1")
    
    # Check if embeddings are already cached
    print("\n" + "="*60)
    print("STEP 2: EMBEDDINGS")
    print("="*60)
    
    if config.MLP_EMBEDDINGS_PATH.exists():
        print(f"\nFound cached embeddings at {config.MLP_EMBEDDINGS_PATH}")
        print("Loading cached embeddings (skipping re-computation)...")
        with open(config.MLP_EMBEDDINGS_PATH, 'rb') as f:
            cached = pickle.load(f)
            train_embeddings = cached['train']
            val_embeddings = cached['val']
            print(f"Loaded: Train {train_embeddings.shape}, Val {val_embeddings.shape}")
    else:
        print("\nNo cached embeddings found. Generating embeddings...")
        print("This may take several minutes on first run.")
        print(f"Started at: {pd.Timestamp.now()}")
        
        # Generate embeddings using config defaults
        train_embeddings = generate_embeddings(train_df['text'].tolist())
        val_embeddings = generate_embeddings(val_df['text'].tolist())
        
        # Save embeddings for future use
        print(f"\nSaving embeddings to {config.MLP_EMBEDDINGS_PATH}...")
        with open(config.MLP_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump({
                'train': train_embeddings,
                'val': val_embeddings,
                'model_name': config.MLP_SENTENCE_MODEL
            }, f)
        print(f"Embeddings cached. Future runs will be much faster!")
        print(f"Completed at: {pd.Timestamp.now()}")
    
    # Create datasets and dataloaders
    print("\n" + "="*60)
    print("STEP 3: PREPARING DATALOADERS")
    print("="*60)
    train_dataset = EmbeddingDataset(train_embeddings, y_train)
    val_dataset = EmbeddingDataset(val_embeddings, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=False)
    print(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Initialize model
    print("\n" + "="*60)
    print("STEP 4: INITIALIZING MODEL")
    print("="*60)
    embedding_dim = train_embeddings.shape[1]
    model = MLPClassifier(
        input_dim=embedding_dim,
        hidden_dims=config.MLP_HIDDEN_DIMS,
        num_classes=config.MLP_NUM_CLASSES,
        dropout=config.MLP_DROPOUT,
        ordinal=config.MLP_USE_ORDINAL_LOSS
    ).to(config.DEVICE)
    
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    if config.MLP_USE_ORDINAL_LOSS:
        criterion = OrdinalRegressionLoss(
            config.MLP_NUM_CLASSES, 
            class_weights=class_weights_tensor if config.MLP_USE_CLASS_WEIGHTS else None
        )
        print("\nUsing Ordinal Regression Loss (better for ordered classes)")
        if config.MLP_USE_CLASS_WEIGHTS:
            print("  with class weighting for imbalance")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor if config.MLP_USE_CLASS_WEIGHTS else None
        )
        print("\nUsing CrossEntropy Loss (treats classes as independent)")
        if config.MLP_USE_CLASS_WEIGHTS:
            print("  with class weighting for imbalance")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.MLP_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.MLP_SCHEDULER_FACTOR, 
        patience=config.MLP_SCHEDULER_PATIENCE, verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    if config.MLP_EARLY_STOPPING:
        print(f"Early stopping enabled (patience: {config.MLP_EARLY_STOPPING_PATIENCE})")
    
    for epoch in range(config.MLP_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.MLP_EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, config.MLP_USE_ORDINAL_LOSS)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE, config.MLP_USE_ORDINAL_LOSS)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            # Ensure parent directory exists
            config.MLP_MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'input_dim': embedding_dim,
                    'hidden_dims': config.MLP_HIDDEN_DIMS,
                    'num_classes': config.MLP_NUM_CLASSES,
                    'dropout': config.MLP_DROPOUT,
                    'ordinal': config.MLP_USE_ORDINAL_LOSS,
                    'sentence_model': config.MLP_SENTENCE_MODEL
                }
            }, config.MLP_MODEL_PATH)
            print(f"Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            if config.MLP_EARLY_STOPPING and epochs_without_improvement >= config.MLP_EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"No improvement for {config.MLP_EARLY_STOPPING_PATIENCE} consecutive epochs")
                break
    
    # Save final model
    config.MLP_FINAL_MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': embedding_dim,
            'hidden_dims': config.MLP_HIDDEN_DIMS,
            'num_classes': config.MLP_NUM_CLASSES,
            'dropout': config.MLP_DROPOUT,
            'ordinal': config.MLP_USE_ORDINAL_LOSS,
            'sentence_model': config.MLP_SENTENCE_MODEL
        }
    }, config.MLP_FINAL_MODEL_PATH)
    
    # Save training history
    config.MLP_HISTORY_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(config.MLP_HISTORY_PATH, 'wb') as f:
        pickle.dump(history, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {config.MLP_MODEL_PATH}")


if __name__ == "__main__":
    main()
