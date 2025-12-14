import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Tuple
import pickle

from utils import setup_logger
from mlp_train_04 import MLPClassifier
from config import config

logger = setup_logger()


def load_model_mlp(model_path: Path, device: str) -> Tuple[MLPClassifier, Dict]:
    """Load MLP model and its configuration."""
    print(f"\nLoading MLP model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = MLPClassifier(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3),
        ordinal=config.get('ordinal', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"MLP model loaded (Ordinal: {config.get('ordinal', False)})")
    return model, config


def load_baseline_model(model_path: Path, vectorizer_path: Path) -> Dict:
    """Load baseline BoW + Logistic Regression model."""
    print(f"\nLoading Baseline model from {model_path}...")
    print(f"Loading Vectorizer from {vectorizer_path}...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f"Baseline model loaded (Vectorizer + Logistic Regression)")
    return {'model': model, 'vectorizer': vectorizer}


def generate_embeddings(texts: List[str], model_name: str = None, device: str = None) -> np.ndarray:
    """Generate sentence embeddings for MLP (using saved SentenceTransformer)."""
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
    
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            device=device
        )
    
    print(f"Sentence embeddings generated: {embeddings.shape}")
    return embeddings


def predict_baseline(baseline_data: Dict, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with baseline model."""
    vectorizer = baseline_data['vectorizer']
    model = baseline_data['model']
    
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Convert from 1-5 to 0-4 for consistency with MLP
    predictions = predictions - 1
    
    return predictions, probabilities


def predict_mlp(model: MLPClassifier, embeddings: np.ndarray, device: str, ordinal: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with MLP model."""
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


def format_predictions(texts: List[str], predictions: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Format predictions into a readable DataFrame."""
    results = []
    
    for i, text in enumerate(texts):
        result = {'text': text[:100] + '...' if len(text) > 100 else text}
        
        for model_name, (preds, probs) in predictions.items():
            pred = int(preds[i])
            result[f'{model_name}_prediction'] = pred + 1  # Convert to 1-5
            result[f'{model_name}_confidence'] = probs[i, pred]
        
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    """Main inference pipeline."""
    print("="*80)
    print("MULTI-MODEL INFERENCE: Sentiment Prediction")
    print("="*80)
    print(f"Time: {pd.Timestamp.now()}")
    
    print(f"\nDevice: {config.DEVICE}")
    
    # Use sample texts from config
    SAMPLE_TEXTS = config.SAMPLE_TEXTS
    
    print("\n" + "="*80)
    print("SAMPLE SENTENCES")
    print("="*80)
    for i, text in enumerate(SAMPLE_TEXTS, 1):
        print(f"{i}. {text}")
    
    # Dictionary to store all predictions
    all_predictions = {}
    
    # === Baseline Model ===
    print("\n" + "="*80)
    print("MODEL 1: Baseline (Bow + Logistic Regression)")
    print("="*80)
    
    if config.BASELINE_MODEL_PATH.exists() and config.VECTORIZER_PATH.exists():
        baseline_data = load_baseline_model(config.BASELINE_MODEL_PATH, config.VECTORIZER_PATH)
        
        baseline_preds, baseline_probs = predict_baseline(baseline_data, SAMPLE_TEXTS)
        
        all_predictions['Baseline'] = (baseline_preds, baseline_probs)
        
        print("\nBaseline Predictions:")
        for i, (pred, prob) in enumerate(zip(baseline_preds, baseline_probs), 1):
            pred_label = int(pred) + 1  # pred is now 0-4, convert to 1-5
            confidence = prob[int(pred)]  # pred is now 0-4, valid index
            print(f"  {i}. Predicted: {pred_label}/5 (Confidence: {confidence:.2%})")
    else:
        print(f"Baseline model not found at {config.BASELINE_MODEL_PATH} or {config.VECTORIZER_PATH}")
    
    # === MLP Model ===
    print("\n" + "="*80)
    print("MODEL 2: MLP (Sentence Embeddings)")
    print("="*80)
    
    if config.MLP_MODEL_PATH.exists():
        mlp_model, mlp_config = load_model_mlp(config.MLP_MODEL_PATH, config.DEVICE)
        
        # Generate sentence embeddings for MLP (use config defaults)
        mlp_embeddings = generate_embeddings(SAMPLE_TEXTS)
        
        mlp_preds, mlp_probs = predict_mlp(
            mlp_model, 
            mlp_embeddings, 
            config.DEVICE, 
            mlp_config.get('ordinal', False)
        )
        
        all_predictions['MLP'] = (mlp_preds, mlp_probs)
        
        print("\nMLP Predictions:")
        for i, (pred, prob) in enumerate(zip(mlp_preds, mlp_probs), 1):
            pred_label = int(pred) + 1
            confidence = prob[int(pred)]
            print(f"  {i}. Predicted: {pred_label}/5 (Confidence: {confidence:.2%})")
    else:
        print(f"MLP model not found at {config.MLP_MODEL_PATH}")
    
    # === Summary Table ===
    if all_predictions:
        print("\n" + "="*80)
        print("PREDICTION SUMMARY")
        print("="*80)
        
        results_df = format_predictions(SAMPLE_TEXTS, all_predictions)
        
        print("\n" + results_df.to_string(index=False))
        
        # Save results
        config.INFERENCE_OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
        results_df.to_csv(config.INFERENCE_OUTPUT_PATH, index=False)
        print(f"\nPredictions saved to: {config.INFERENCE_OUTPUT_PATH}")
        
        # Analyze agreement
        print("\n" + "="*80)
        print("MODEL AGREEMENT ANALYSIS")
        print("="*80)
        
        model_cols = [col for col in results_df.columns if col.endswith('_prediction')]
        
        if len(model_cols) >= 2:
            agreements = 0
            for i in range(len(SAMPLE_TEXTS)):
                preds = [results_df.iloc[i][col] for col in model_cols]
                if len(set(preds)) == 1:
                    agreements += 1
                    print(f"  Sample {i+1}: All models agree on {preds[0]}/5")
                else:
                    pred_str = ", ".join([f"{col.split('_')[0]}: {results_df.iloc[i][col]}" 
                                         for col in model_cols])
                    print(f"  Sample {i+1}: Disagreement ({pred_str})")
            
            agreement_rate = 100 * agreements / len(SAMPLE_TEXTS)
            print(f"\nOverall Agreement: {agreements}/{len(SAMPLE_TEXTS)} ({agreement_rate:.1f}%)")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
