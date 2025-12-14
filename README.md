# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Ádám Márk Biró, SS0N7G
- **Aiming for +1 Mark**: No

### Solution Description

This project implements a Legal Text Decoder using machine learning and deep learning techniques for Hungarian legal text classification. Due to the absence of GPU resources, the solution focuses on computationally efficient models that can train in reasonable time on CPU.

**Two models are implemented:**

1. **Baseline Model**: Uses Bag-of-Words (BoW) representation with Logistic Regression for text classification. This provides a simple yet effective baseline for comparison.

2. **MLP Model**: Uses a frozen pre-trained Sentence Transformer (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) to generate contextual embeddings for the legal texts. A 2-layer Multi-Layer Perceptron (MLP) is trained on top of these embeddings for classification. The Sentence Transformer remains frozen to reduce computational cost and training time.
   - **Loss Function**: The model utilizes **Ordinal Cross Entropy Loss (CORAL)**. Since the target labels (sentiment scores 1-5) are ordinal, this loss function is more suitable than standard Cross Entropy as it accounts for the order and distance between classes.

**Note on Model Choice**: More complex architectures like LSTMs or full transformer fine-tuning were not feasible due to the long training times required on CPU without GPU acceleration.

The training pipeline includes:
- Data preprocessing and tokenization of legal texts
- Model training with configurable hyperparameters
- Comprehensive evaluation metrics: Deep evaluation including Confusion Matrix, Accuracy, Precision, Recall, and F1-score
- Inference capabilities for new legal documents

The solution follows a modular architecture with separate scripts for each stage of the machine learning pipeline, ensuring reproducibility and maintainability.

### Data Preparation

The data preparation process involves the following steps:

1. **Data Source**: The dataset is automatically downloaded from the provided SharePoint link when running the preprocessing script. The dataset contains Hungarian legal text documents with sentiment annotations.

2. **Data Split Strategy**: 
   - **Test Set**: Created exclusively from consensus texts, ensuring high-quality evaluation data
   - **Training and Validation Sets**: Created from regular annotated texts, with stratified splitting to maintain class distribution

3. **Preprocessing**: Run `src/data_processing_01.py` to:
   - Automatically download the dataset ZIP file
   - Extract and load JSON annotation files
   - Process consensus files for test set creation with majority voting
   - Process regular annotation files for training/validation
   - Remove duplicate texts between consensus and regular sets
   - Split regular data into training (85%) and validation (15%) sets with stratification
   - Save processed datasets in CSV format for the training pipeline

The preprocessing script handles all data acquisition and preparation automatically, requiring no manual data download or placement. The consensus-based test set ensures reliable evaluation metrics.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

**Note:** This project is configured to run on CPU only, as no GPU is available. This ensures compatibility and reproducibility across different systems without requiring GPU support.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command.

**Note:** The `data/` and `output/` directories will be created automatically if they don't exist. For better visibility of results, it's recommended to create them manually beforehand, especially when using mounted volumes.

**To capture the logs for submission (required), redirect the output to a file:**

**Option 1: With mounted directories (for persistent data and results)**

*Recommended: Create directories first (optional but helpful):*
```bash
mkdir -p data output log
```

*Windows PowerShell:*
```powershell
docker run -v "${PWD}\data:/app/data" -v "${PWD}\output:/app/output" dl-project > log/run.log 2>&1
```

*Windows CMD:*
```cmd
docker run -v "%cd%/data:/app/data" -v "%cd%/output:/app/output" dl-project > log/run.log 2>&1
```

*Linux/Mac:*
```bash
docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/output:/app/output" dl-project > log/run.log 2>&1
```

**Option 2: Without mounts (data/output created inside container)**

*All platforms:*
```bash
docker run dl-project > log/run.log 2>&1
```

*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).
*   If using Option 2, the data must be included in the Docker image or downloaded by the scripts.

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `data_processing_01.py`: Automatically downloads the EURLEX dataset, loads, cleans, tokenizes, and preprocesses legal text data, then splits into train/validation/test sets.
    - `baseline_model_train_02.py`: Trains the baseline model architecture on preprocessed legal text data with configured hyperparameters. The model uses Bag-of-Words with Logistic Regression.
    - `baseline_model_eval_03.py`: Evaluates the trained baseline model on test data and generates performance metrics and visualizations.
    - `mlp_train_04.py`: Trains an MLP (Multi-Layer Perceptron) model as an alternative architecture for legal text classification. The embedding for the MLP is a Sentence Transformers to get proper context for the texts. It implements Ordinal Cross Entropy (CORAL) loss for training.
    - `mlp_eval_05.py`: Evaluates the trained MLP model on test data and compares performance against the baseline model.
    - `inference_06.py`: Performs inference on new, unseen legal documents using trained models to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs, learning rate) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts, including logging configuration.
    - `run.sh`: Shell script that orchestrates the entire pipeline execution - runs data preprocessing, baseline training, baseline evaluation, MLP training, MLP evaluation, and inference sequentially. This is the main entry point used by the Dockerfile.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for analyzing the distribution of labels, examining the characteristics of legal text data corresponding to each label, and performing exploratory data analysis (EDA) with visualizations.

- **`log/`**: Contains log files.
    - `run.log`: Log file showing the output of training runs, including hyperparameters, model architecture, training metrics, validation results, and final evaluation.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.



## Eltérések (Deviations)

### Eltérés 1 (Deviation 1): CPU-only Environment

**Futtatáshoz szükséges módosítás (Required modification for execution):** The project runs exclusively on CPU without GPU, as I did not have access to GPU resources. No GPU-specific Docker settings are required (e.g., `--gpus all` flag). Use the standard `docker run` command as described in the README.