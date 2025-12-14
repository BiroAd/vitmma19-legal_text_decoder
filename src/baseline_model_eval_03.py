import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from utils import setup_logger
from config import config

logger = setup_logger()


def load_model(model_path=None, vectorizer_path=None):
    """Load trained model and vectorizer."""
    if model_path is None:
        model_path = config.BASELINE_MODEL_PATH
    if vectorizer_path is None:
        vectorizer_path = config.VECTORIZER_PATH
    
    print("Loading trained model and vectorizer...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f"Model loaded from '{model_path}'")
    print(f"Vectorizer loaded from '{vectorizer_path}'")
    
    return model, vectorizer


def load_test_data(data_dir=None):
    """Load test dataset."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    test_df = pd.read_csv(Path(data_dir) / 'test_sentiments.csv')
    print(f"Test set: {len(test_df)} examples")
    return test_df


def evaluate_model(model, vectorizer, test_df, output_dir=None):
    """
    Evaluate model on test set and print metrics.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        test_df: Test dataframe with 'text' and 'sentiment_choice' columns
        output_dir: Directory to save outputs (defaults to config)
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    # Transform test data
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['sentiment_choice']
    
    print(f"Test shape: {X_test.shape}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Define class names
    class_names = [f"Class {i}" for i in sorted(y_test.unique())]
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # F1 scores
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualize confusion matrix using sklearn's ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d',
        colorbar=True
    )
    
    ax.set_title('Confusion Matrix - Baseline Model (BoW + Logistic Regression)', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    cm_path = output_path / 'baseline_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to '{cm_path}'")
    plt.show()
    
    # Save predictions
    results_df = test_df.copy()
    results_df['predicted'] = y_pred
    results_df['correct'] = (y_test == y_pred)
    
    # Add prediction probabilities
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = y_pred_proba[:, i]
    
    results_path = output_path / 'baseline_predictions.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to '{results_path}'")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for class_label in sorted(y_test.unique()):
        class_mask = y_test == class_label
        class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
        class_count = class_mask.sum()
        print(f"  Class {class_label}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)
    
    # Load model (uses config paths)
    model, vectorizer = load_model()
    
    # Load test data (uses config data dir)
    test_df = load_test_data()
    
    # Evaluate (uses config output dir)
    results = evaluate_model(model, vectorizer, test_df)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
