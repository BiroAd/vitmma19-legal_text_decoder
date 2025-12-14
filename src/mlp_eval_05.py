import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from utils import setup_logger
from mlp_train_04 import MLPClassifier
from config import config

logger = setup_logger()


def load_model(model_path=None, device=None):
    """Load trained MLP model."""
    if model_path is None:
        model_path = config.MLP_MODEL_PATH
    if device is None:
        device = config.DEVICE
    
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    
    model = MLPClassifier(
        input_dim=config_dict['input_dim'],
        hidden_dims=config_dict['hidden_dims'],
        num_classes=config_dict['num_classes'],
        dropout=config_dict.get('dropout', 0.2),
        ordinal=config_dict.get('ordinal', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Sentence model: {config_dict['sentence_model']}")
    print(f"Ordinal regression: {config_dict.get('ordinal', False)}")
    
    return model, config_dict


def load_test_data(data_dir=None):
    """Load test dataset."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    test_df = pd.read_csv(Path(data_dir) / 'test_sentiments.csv')
    print(f"Test set: {len(test_df)} examples")
    return test_df


def generate_embeddings(texts, model_name=None, batch_size=None, device=None):
    """Generate sentence embeddings using saved SentenceTransformer model."""
    if batch_size is None:
        batch_size = config.MLP_BATCH_SIZE
    if device is None:
        device = config.DEVICE
    
    # Load from saved path if it exists, otherwise fall back to downloading
    if config.SENTENCE_TRANSFORMER_PATH.exists():
        print(f"\nLoading SentenceTransformer from saved path: {config.SENTENCE_TRANSFORMER_PATH}")
        model = SentenceTransformer(str(config.SENTENCE_TRANSFORMER_PATH))
    else:
        if model_name is None:
            model_name = config.MLP_SENTENCE_MODEL
        print(f"\nWarning: Saved model not found at {config.SENTENCE_TRANSFORMER_PATH}")
        print(f"Downloading SentenceTransformer: {model_name}")
        model = SentenceTransformer(model_name)
    
    model = model.to(device)
    model.eval()
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def evaluate_model(model, embeddings, labels, device=None):
    """Evaluate model and return predictions."""
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    embeddings_tensor = torch.FloatTensor(embeddings).to(device)
    
    with torch.no_grad():
        outputs = model(embeddings_tensor)
        
        # Handle ordinal vs standard classification
        if model.ordinal:
            # For ordinal: outputs are class probabilities (from ordinal_to_probs)
            probabilities = outputs.cpu().numpy()
            predictions = probabilities.argmax(axis=1)
        else:
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()
    
    return predictions, probabilities


def plot_results(y_test, y_pred, class_names, output_dir=None):
    """Plot confusion matrix and save results."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    
    ax.set_title('Confusion Matrix - MLP Model (Sentence Transformers)', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    cm_path = output_path / 'mlp_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to '{cm_path}'")
    plt.show()


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("MLP MODEL EVALUATION")
    print("="*60)
    
    print(f"Device: {config.DEVICE}")
    
    # Load model
    model, model_config = load_model()
    
    # Load test data
    test_df = load_test_data()
    
    # Adjust labels (1-5 to 0-4)
    y_test = test_df['sentiment_choice'].values - 1
    
    # Generate embeddings for test set (will use saved model automatically)
    test_embeddings = generate_embeddings(test_df['text'].tolist())
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    y_pred, y_pred_proba = evaluate_model(model, test_embeddings, y_test)
    
    # Define class names
    class_names = [f"Class {i+1}" for i in range(model_config['num_classes'])]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_results(y_test, y_pred, class_names)
    
    # Save predictions
    results_df = test_df.copy()
    results_df['predicted'] = y_pred + 1  # Convert back to 1-5
    results_df['correct'] = (y_test == y_pred)
    
    # Add probabilities
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = y_pred_proba[:, i]
    
    results_path = config.OUTPUT_DIR / 'mlp_predictions.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nPredictions saved to '{results_path}'")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            class_count = class_mask.sum()
            print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
