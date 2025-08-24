# StaticDynamic-BERT-AutoPyTorch

This repository contains the implementation of StaticBERT Auto-PyTorch and DynamicBERT Auto-PyTorch, novel approaches that integrate static and dynamically fine-tuned BERT embeddings into the Auto-PyTorch framework for binary and multi-class text classification tasks.

## Overview

This work introduces two novel approaches for enhancing AutoML in text classification tasks:

1. **StaticBERT Auto-PyTorch**: A static embedding approach that integrates pre-fine-tuned BERT embeddings into AutoML for binary classification, enhancing both performance and computational efficiency.

2. **DynamicBERT Auto-PyTorch**: A dynamic fine-tuning method designed for multi-class classification, which adjusts BERT embeddings based on the dataset to provide adaptive performance across diverse NLP tasks.

## Paper

This implementation is based on our paper "Static and dynamic contextual embedding for AutoML in text classification tasks" published at ICNLP. [(https://ieeexplore.ieee.org/abstract/document/11108687)]

## Features

- Integration of BERT embeddings with Auto-PyTorch
- Support for both binary and multi-class classification
- Efficient memory usage through optimized vector sizes
- Dynamic fine-tuning capabilities for multi-class tasks
- Pre-fine-tuned embeddings for binary classification
- Comprehensive evaluation on diverse datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/StaticDynamic-BERT-AutoPyTorch.git
cd StaticDynamic-BERT-AutoPyTorch

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Binary Classification with StaticBERT

To run StaticBERT Auto-PyTorch for binary classification:

```bash
python src/static_dynamic/StaticBERT_AutoPyTorch.py --data_path path/to/your/data.csv --model_path path/to/saved/models
```

The script will:
1. Process and prepare your data
2. Generate static BERT embeddings
3. Train and evaluate the model
4. Output performance metrics and results

### Multi-class Classification with DynamicBERT

To run DynamicBERT Auto-PyTorch for multi-class classification:

```bash
python src/static_dynamic/DynamicBERT_AutoPyTorch.py --data_path path/to/your/data.csv --model_path path/to/saved/models
```

The script will:
1. Process and prepare your data
2. Dynamically fine-tune BERT for your specific dataset
3. Train and evaluate the model
4. Output performance metrics and results

### Example Dataset Format

Your input CSV file should contain at least two columns:
- A text column containing the input text
- A label column containing the classification labels

Example:
```csv
text,label
"This is a positive review",1
"This is a negative review",0
```


## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- Auto-PyTorch
- NumPy
- scikit-learn

## Citation


@inproceedings{safikhani2025static,
  title={Static and Dynamic Contextual Embedding for AutoML in Text Classification Tasks},
  author={Safikhani, Parisa and Broneske, David},
  booktitle={2025 7th International Conference on Natural Language Processing (ICNLP)},
  pages={292--301},
  year={2025},
  organization={IEEE}
}

