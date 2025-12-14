import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pickle

from utils import setup_logger
from config import config

logger = setup_logger()

def load_data(data_dir=None):
    """Load train, val, and test datasets."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    train_df = pd.read_csv(Path(data_dir) / 'train_sentiments.csv')
    val_df = pd.read_csv(Path(data_dir) / 'val_sentiments.csv')
    
    print(f"Train set: {len(train_df)} examples")
    print(f"Val set: {len(val_df)} examples")
    
    return train_df, val_df


def create_bag_of_words(train_texts, val_texts, max_features=None, ngram_range=None, min_df=None):
    """
    Create Bag of Words representation using CountVectorizer.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        max_features: Maximum number of features to extract (defaults to config)
        ngram_range: N-gram range (defaults to config)
        min_df: Minimum document frequency (defaults to config)
    
    Returns:
        Tuple of (X_train, X_val, vectorizer)
    """
    if max_features is None:
        max_features = config.BASELINE_MAX_FEATURES
    if ngram_range is None:
        ngram_range = config.BASELINE_NGRAM_RANGE
    if min_df is None:
        min_df = config.BASELINE_MIN_DF
    
    print(f"\nCreating Bag of Words with max_features={max_features}...")
    
    vectorizer = CountVectorizer(
        max_features=max_features,
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df
    )
    
    # Fit on train+val combined (for baseline, we want maximum vocabulary)
    all_train_texts = pd.concat([train_texts, val_texts])
    vectorizer.fit(all_train_texts)
    
    # Transform all sets
    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    
    return X_train, X_val, vectorizer


def train_logistic_regression(X_train, y_train, X_val, y_val, 
                               max_iter=None, solver=None, class_weight=None, random_state=None):
    """
    Train Logistic Regression model (multi-class).
    Combines train+val for final training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Trained LogisticRegression model
    """
    if max_iter is None:
        max_iter = config.BASELINE_MAX_ITER
    if solver is None:
        solver = config.BASELINE_SOLVER
    if class_weight is None:
        class_weight = config.BASELINE_CLASS_WEIGHT
    if random_state is None:
        random_state = config.BASELINE_RANDOM_STATE
    
    print("\nTraining Logistic Regression (multi-class)...")
    
    # Combine train and val for baseline training
    from scipy.sparse import vstack
    X_combined = vstack([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    print(f"Combined training set: {X_combined.shape[0]} examples")
    print(f"Number of classes: {len(np.unique(y_combined))}")
    
    # Train logistic regression with multi-class support
    model = LogisticRegression(
        multi_class='multinomial',
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
        class_weight=class_weight
    )
    
    model.fit(X_combined, y_combined)
    
    print("Training complete!")
    return model


def save_model(model, vectorizer, output_dir=None):
    """Save trained model and vectorizer."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    model_path = config.BASELINE_MODEL_PATH
    vectorizer_path = config.VECTORIZER_PATH
    
    # Ensure parent directories exist
    model_path.parent.mkdir(exist_ok=True, parents=True)
    vectorizer_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nModel saved to '{model_path}'")
    print(f"Vectorizer saved to '{vectorizer_path}'")


def main():
    """Main pipeline for baseline model training."""
    print("="*60)
    print("BASELINE MODEL TRAINING: Bag of Words + Logistic Regression")
    print("="*60)
    
    # Print configuration
    print("\nBaseline Configuration:")
    print(f"  Max Features: {config.BASELINE_MAX_FEATURES}")
    print(f"  N-gram Range: {config.BASELINE_NGRAM_RANGE}")
    print(f"  Min DF: {config.BASELINE_MIN_DF}")
    print(f"  Max Iterations: {config.BASELINE_MAX_ITER}")
    print(f"  Class Weight: {config.BASELINE_CLASS_WEIGHT}")
    
    # Load data
    train_df, val_df = load_data()
    
    # Prepare data (uses config defaults)
    X_train_bow, X_val_bow, vectorizer = create_bag_of_words(
        train_df['text'], 
        val_df['text']
    )
    
    y_train = train_df['sentiment_choice']
    y_val = val_df['sentiment_choice']
    
    # Train model (uses config defaults)
    model = train_logistic_regression(X_train_bow, y_train, X_val_bow, y_val)
    
    # Save model
    save_model(model, vectorizer)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
