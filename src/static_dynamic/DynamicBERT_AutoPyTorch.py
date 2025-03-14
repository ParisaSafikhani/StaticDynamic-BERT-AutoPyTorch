import os
import logging
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, auc
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
#from autoPyTorch.codes.functions import process_and_rename_columns, setup_model_directory, fine_tune_bert, text_embeddings, get_embeddings_path, prepare_embeddings_with_labels
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import CrossValTypes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoPyTorch.codes.functions import process_and_rename_columns, setup_model_directory, fine_tune_bert, text_embeddings, get_embeddings_path, prepare_embeddings_with_labels

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set a different temporary directory if needed
os.environ["TMPDIR"] = "/home/tmp"

# Ensure CUDA devices are correctly set if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change as per your requirement

# Setup logging to see detailed information
logging.basicConfig(level=logging.INFO)

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for AutoPyTorch: {device}")

# Function to monitor memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert to MB

# Paths to data and saved models
path_to_data = ".../Auto-PyTorch_autoNLP/data/old/Humor_test.csv"
path_to_saved_models = ".../Auto-PyTorch_autoNLP/Fine-tuned-models"

start_time = time.time()

data, num_labels = process_and_rename_columns(path_to_data)
model_directory = setup_model_directory(path_to_data=path_to_data, path_to_saved_models=path_to_saved_models)
fine_tune_bert(data, num_labels, model_directory, patience_limit=2, num_epochs=100)

embeddings, labels = text_embeddings(path_to_data, model_directory, 'text', max_length=512)
print(f"Embeddings type after function call: {type(embeddings)}")
print(f"Labels type after function call: {type(labels)}")

embeddings_path = get_embeddings_path(path_to_data)
df = prepare_embeddings_with_labels(embeddings, labels, embeddings_path)
print("Embeddings path:", embeddings_path)

filtered_data = df.groupby('label').filter(lambda x: len(x) > 1)
print(filtered_data)

# Split your data again
X = filtered_data.drop(['label'], axis=1)
y = filtered_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feat_types = ['numerical'] * X_train.shape[1]  # Assuming all features are numerical


api = TabularClassificationTask(resampling_strategy=CrossValTypes.k_fold_cross_validation)
# Run the AutoPyTorch search
search_start_time = time.time()
api.search(
    optimize_metric='accuracy',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feat_types=feat_types,
    total_walltime_limit=1000,
    func_eval_time_limit_secs=300,  # Increased evaluation time limit
    budget_type='epochs',
    min_budget=5,
    max_budget=50,
    enable_traditional_pipeline=True,
    memory_limit=81920,
    smac_scenario_args=None,
    get_smac_object_callback=None,
    all_supported_metrics=True,
    precision=32,
    disable_file_output=None,
    load_models=True,
    portfolio_selection=None,
    dataset_compression=False
)
search_end_time = time.time()

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)

end_time = time.time()

# Print results
print("Accuracy score", score)
print(f"Total execution time: {end_time - start_time:.2f} seconds")
print(f"AutoPyTorch search time: {search_end_time - search_start_time:.2f} seconds")
print(f"Memory usage: {memory_usage():.2f} MB")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall:.4f}')

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='macro')
print(f'F1 Score: {f1:.4f}')

micro_f1 = f1_score(y_test, y_pred, average='micro')

print("Micro F1 Score: {:.2f}".format(micro_f1))


# Calculate AUPRC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred, pos_label=1)
auprc = auc(recall_curve, precision_curve)
print(f'AUPRC: {auprc:.4f}')

# Plot Precision-Recall curve
plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='blue', label=f'AUPRC = {auprc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig('precision_recall_curve.png')
plt.close()

print(f"Total execution time: {end_time - start_time:.2f} seconds")
print(f"AutoPyTorch search time: {search_end_time - search_start_time:.2f} seconds")
print(f"Memory usage: {memory_usage():.2f} MB")



from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np

# Replace the existing prediction with probability estimates
if hasattr(api, 'predict_proba'):
    y_score = api.predict_proba(X_test)
else:
    raise AttributeError("The AutoPyTorch API does not support 'predict_proba'. Ensure that the model provides probability estimates.")
# Binarize the output labels for multiclass AUPRC
classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)
n_classes = y_test_binarized.shape[1]
# Compute Macro-average AUPRC
average_precision_macro = average_precision_score(y_test_binarized, y_score, average='macro')
print(f'Macro-average AUPRC: {average_precision_macro:.4f}')

# Compute Micro-average AUPRC
average_precision_micro = average_precision_score(y_test_binarized, y_score, average='micro')
print(f'Micro-average AUPRC: {average_precision_micro:.4f}')


