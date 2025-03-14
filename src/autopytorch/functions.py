import pandas as pd
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import copy
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import KFold

def rename_columns_based_on_content(csv_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
    
        # Check if the DataFrame has at least two columns
        if len(df.columns) < 2:
            raise ValueError("The DataFrame must have at least two columns.")
    
        # Assume: The column with the longest average text content is the text column
        avg_length_per_column = df.applymap(lambda x: len(str(x))).mean()
        text_column = avg_length_per_column.idxmax()
    
        # Ensure the text column was identified
        if text_column not in df.columns:
           raise KeyError(f"Text column '{text_column}' could not be identified.")
    
        # Assume: The other main column is the label column
        label_column = [col for col in df.columns if col != text_column][0]
    
        # Rename columns
        df.rename(columns={text_column: 'text', label_column: 'label'}, inplace=True)
    
        # Check if renaming was successful
        if 'text' not in df.columns or 'label' not in df.columns:
            raise KeyError("Error renaming columns. Check the column names in your CSV file.")
    

 



def process_and_rename_columns(csv_path):
    # CSV-Datei einlesen
    df = pd.read_csv(csv_path)
    print(df)
    # Überprüfen, ob der DataFrame mindestens zwei Spalten hat
    if len(df.columns) < 2:
        raise ValueError("The DataFrame must have at least two columns.")
    
    # Durchschnittliche Länge der Einträge pro Spalte berechnen
    avg_length_per_column = df.applymap(lambda x: len(str(x))).mean()
    text_column = avg_length_per_column.idxmax()
    
    # Überprüfen, ob die Textspalte korrekt identifiziert wurde
    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' could not be identified.")
    
    # Die andere Spalte als Label-Spalte identifizieren
    label_column = [col for col in df.columns if col != text_column][0]
    
    # Spalten umbenennen
    df.rename(columns={text_column: 'text', label_column: 'label'}, inplace=True)
    
    # Überprüfen, ob das Umbenennen erfolgreich war
    if 'text' not in df.columns or 'label' not in df.columns:
        raise KeyError("Error renaming columns. Check the column names in your CSV file.")
    
    # Zeilen mit NaN-Werten in den 'text'- oder 'label'-Spalten entfernen
    df.dropna(subset=['text', 'label'], inplace=True)
    
    # Sicherstellen, dass 'text' Spalte aus Strings besteht
    df['text'] = df['text'].astype(str)
    
    # Anzahl der Vorkommen jedes Labels zählen
    label_counts = df['label'].value_counts()
    print("Label counts before filtering:", label_counts)  # Debugging-Ausgabe
    
    # Labels mit weniger als 2 Vorkommen entfernen
    labels_to_keep = label_counts[label_counts >= 2].index
    df = df[df['label'].isin(labels_to_keep)]
    
    # Einzigartige Labels nach der Filterung identifizieren und ein Mapping erstellen
    unique_labels = sorted(df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_mapping)
    
    # Anzahl der eindeutigen Labels nach der Filterung
    num_labels = len(unique_labels)
    
    # Überprüfen, ob noch Labels übrig sind
    if num_labels == 0:
        raise ValueError("No labels with at least 2 occurrences found.")
    
    # DataFrame und Anzahl der eindeutigen Labels zurückgeben
    print("DataFrame after filtering:\n", df)  # Debugging-Ausgabe
    return df, num_labels


def set_cv_folds(n_samples, n_features=None, target_type='continuous', min_samples_per_fold=30):
    """
    Determine the number of cross-validation folds based on dataset characteristics.
    
    Args:
    n_samples (int): Number of samples in the dataset.
    n_features (int, optional): Number of features in the dataset.
    target_type (str): Type of target variable ('continuous' or 'categorical').
    min_samples_per_fold (int): Minimum number of samples required per fold.
    
    Returns:
    int: Recommended number of cross-validation folds.
    """
    # Base number of folds
    if n_samples < 100:
        base_folds = 3
    elif n_samples < 1000:
        base_folds = 5
    elif n_samples < 10000:
        base_folds = 10
    else:
        base_folds = 20
    
    # Adjust based on features (if provided)
    if n_features is not None:
        if n_features > n_samples / 10:
            base_folds = max(3, base_folds - 2)  # Reduce folds for high-dimensional data
    
    # Adjust based on target type
    if target_type == 'categorical':
        base_folds = min(base_folds, 5)  # Limit folds for classification tasks
    
    # Ensure minimum samples per fold
    max_folds = n_samples // min_samples_per_fold
    
    # Return the minimum of base_folds and max_folds
    return min(base_folds, max_folds)



def setup_model_directory(path_to_data, path_to_saved_models):
    """
    Set up the directory for the model based on the data file name, clearing existing contents if the directory exists,
    or creating a new directory if it does not exist.

    Args:
        path_to_data (str): The file path to the data CSV.
        path_to_saved_models (str): The base path where model directories are saved.

    Returns:
        str: The path to the model directory.
    """
    # Extract the data name from the path
    data_name = os.path.basename(path_to_data).split('.')[0]
    
    # Construct the full path to the model directory
    model_directory = os.path.join(path_to_saved_models, data_name)
    
    # Check if the model directory exists
    if os.path.exists(model_directory):
        # Delete the contents of the folder
        for filename in os.listdir(model_directory):
            file_path = os.path.join(model_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        # Create the directory if it does not exist
        os.makedirs(model_directory)
        print('model directory has been created')
    return model_directory



def fine_tune_bert(data, num_labels, model_directory, patience_limit=2, num_epochs=100):
    # Tokenizer and model setup
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate average token length and determine batch size and max length
    token_lengths = data['text'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
    average_length = token_lengths.mean()
    if average_length <= 64:
        batch_size, max_length = 32, 64
    elif average_length <= 128:
        batch_size, max_length = 24, 128
    elif average_length <= 256:
        batch_size, max_length = 16, 256
    else:
        batch_size, max_length = 8, 512
    
    # Automatically determine the number of folds using set_cv_folds
    y = data['label']  # Extract the labels
    n_samples = len(data)
    target_type = 'categorical' if num_labels > 1 else 'continuous'
    
    # Dynamically set the number of folds
    n_folds = set_cv_folds(n_samples=n_samples, target_type=target_type, min_samples_per_class_per_fold=5, y=y)

    # DataLoader creation function
    def create_data_loader(df, tokenizer, max_length, batch_size):
        inputs = tokenizer.batch_encode_plus(
            df['text'].tolist(), 
            max_length=max_length, 
            pad_to_max_length=True, 
            truncation=True, 
            return_tensors="pt"
        )
        labels = torch.tensor(df['label'].tolist())
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        return DataLoader(dataset, batch_size=batch_size)

    # Initialize KFold with the dynamically determined number of folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"Starting fold {fold + 1} of {n_folds}")
        
        # Create model for this fold
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.to(device)
        
        # Prepare Data Loaders for this fold
        train_loader = create_data_loader(data.iloc[train_idx], tokenizer, max_length, batch_size)
        val_loader = create_data_loader(data.iloc[val_idx], tokenizer, max_length, batch_size)

        # Setup the optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss().to(device)

        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        # Training and validation loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = tuple(item.to(device) for item in batch)
                b_input_ids, b_input_mask, b_labels = batch
                model.zero_grad()
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            total_eval_loss = 0
            for batch in val_loader:
                batch = tuple(item.to(device) for item in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_eval_loss += loss.item()

            avg_val_loss = total_eval_loss / len(val_loader)
            print(f'Fold {fold + 1}, Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f'Validation loss has not improved for {patience_limit} epochs. Stopping early.')
                    break

        fold_val_losses.append(best_val_loss)

        best_fold = np.argmin(fold_val_losses)
        best_model_path = model_directory
        print(f"The best model is from fold {best_fold + 1}, saved at {best_model_path}")
        
        # Save the best model for this fold
        model.load_state_dict(best_model_state)
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"Fold {fold + 1} complete. The best model has been saved.")

    # Print average validation loss across all folds
    print(f"Average validation loss across all folds: {np.mean(fold_val_losses):.4f}")
    print("Cross-validation complete.")    




def text_embeddings(input_data_path, model_name, feature_col, max_length=512, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data, num_labels = process_and_rename_columns(input_data_path)
    data['text'] = data['text'].astype(str)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained(model_name).to(device)
    
    all_embeddings = []
    
    for i in range(0, len(data), batch_size):
        batch_data = data.iloc[i:i+batch_size]
        tokens = tokenizer(batch_data[feature_col].tolist(), max_length=max_length, truncation=True, padding='max_length', return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            batch_embeddings = model(**tokens).last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        
        del tokens  # Explicitly delete to free memory
        torch.cuda.empty_cache()
    
    embeddings = np.vstack(all_embeddings)
    del model  # Delete the model to free memory
    torch.cuda.empty_cache()

    # Convert labels to a Series if it is a DataFrame
    labels = data['label']
    if isinstance(labels, pd.DataFrame):
        labels = labels.squeeze()  # Convert DataFrame to Series
    
    # Print to verify the format and content of the embeddings
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding: {embeddings[0]}")
    print(f"Embeddings type: {type(embeddings)}")
    print(f"Labels type: {type(labels)}")

    return embeddings, labels  # Return the embeddings and labels


  

def get_embeddings_path(path_to_data):
    """
    This function generates the path to embeddings based on the provided path_to_data.
    
    Parameters:
    path_to_data (str): The file path to the data CSV.
    
    Returns:
    str: The file path to the corresponding embeddings CSV.
    """
    # Base directory for embeddings
    base_embeddings_dir = "/home/safikhani/main_repository/Auto-PyTorch_autoNLP/embeddings"

    # Extract the base name of the data file without extension
    data_basename = os.path.splitext(os.path.basename(path_to_data))[0]

    # Construct the embeddings file name
    embeddings_file_name = f"{data_basename}_embedding.csv"

    # Construct the full path to the embeddings file
    embeddings_path = os.path.join(base_embeddings_dir, embeddings_file_name)
    
    return embeddings_path

def pari(embeddings,data,output_path):
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.columns = [f'embed_{i + 1}' for i in range(embeddings.shape[1])]
    df_embeddings['label'] = data['label']
    df_embeddings.to_csv(output_path)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def prepare_embeddings_with_labels(embeddings, labels, output_path):
    # Print to verify the format and content of the embeddings
    print(f"Received embeddings of type: {type(embeddings)}")
    print(f"Received labels of type: {type(labels)}")

    # Ensure embeddings are numerical
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings are not in expected format (NumPy array).")
    
    # Ensure labels are a pandas Series or numpy array
    if isinstance(labels, pd.DataFrame):
        labels = labels.squeeze()  # Convert DataFrame to Series if needed
    
    # Convert labels to numeric if they are not
    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError("Labels are not in expected numeric format.")
    
    # Determine the maximum length of the embeddings
    max_length = max(len(embed) for embed in embeddings)
    
    # Initialize list for padded embeddings
    padded_embeddings = []
    
    for embed in embeddings:
        if len(embed) < max_length:
            # Pad the embedding with zeros if it is shorter than the max length
            padded_embed = np.pad(embed, (0, max_length - len(embed)), 'constant')
        else:
            # Truncate the embedding if it is longer than the max length
            padded_embed = embed[:max_length]
        
        padded_embeddings.append(padded_embed)
    
    # Convert list of embeddings to a numpy array
    padded_embeddings = np.array(padded_embeddings)
    
    # Convert to DataFrame
    df_embeddings = pd.DataFrame(padded_embeddings)
    df_embeddings.columns = [f'embed_{i + 1}' for i in range(padded_embeddings.shape[1])]
    df_embeddings['label'] = labels

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_embeddings.to_csv(output_path, index=False)
    return df_embeddings

def get_one_sample_per_label(data):
    # Group by the 'label' column and use first() to get the first occurrence of each label
    unique_samples = data.groupby('label').first().reset_index()
    return unique_samples


def set_cv_folds(n_samples, n_features=None, target_type='continuous', min_samples_per_class_per_fold=5, y=None):
    """
    Determine the number of cross-validation folds based on dataset characteristics.
    
    Args:
        n_samples (int): Number of samples in the dataset.
        n_features (int, optional): Number of features in the dataset.
        target_type (str): Type of target variable ('continuous' or 'categorical').
        min_samples_per_class_per_fold (int): Minimum number of samples per class required per fold.
        y (array-like, optional): Target variable array for classification tasks.
    
    Returns:
        int: Recommended number of cross-validation folds.
    
    Raises:
        ValueError: If input parameters are invalid.
    """
    # Input validation
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if n_features is not None and (not isinstance(n_features, int) or n_features <= 0):
        raise ValueError("n_features must be a positive integer.")
    if not isinstance(min_samples_per_class_per_fold, int) or min_samples_per_class_per_fold <= 0:
        raise ValueError("min_samples_per_class_per_fold must be a positive integer.")
    if target_type not in ['continuous', 'categorical']:
        raise ValueError("target_type must be 'continuous' or 'categorical'.")
    if target_type == 'categorical':
        if y is None:
            raise ValueError("Target variable 'y' must be provided for categorical target type.")
        else:
            n_samples = len(y)
    
    # Base number of folds based on sample size
    if n_samples < 100:
        base_folds = 3
    elif n_samples < 1000:
        base_folds = 5
    elif n_samples < 10000:
        base_folds = 10
    else:
        base_folds = 20
    
    # Adjust based on features (if provided)
    if n_features is not None and n_features > n_samples / 10:
        base_folds = max(3, base_folds - 2)  # Reduce folds for high-dimensional data
    
    # Adjust number of folds dynamically for classification tasks
    if target_type == 'categorical':
        from collections import Counter
        class_counts = Counter(y)
        
        # Calculate max folds per class
        max_folds_per_class = {
            cls: count // min_samples_per_class_per_fold
            for cls, count in class_counts.items()
        }
        max_possible_folds = min(max_folds_per_class.values())
        
        if max_possible_folds < 2:
            raise ValueError("Not enough samples per class to perform cross-validation with the given min_samples_per_class_per_fold.")
        
        # Set number of folds dynamically
        base_folds = min(base_folds, max_possible_folds)
    
    # Ensure minimum total samples per fold
    max_folds_overall = n_samples // (min_samples_per_class_per_fold * len(class_counts)) if target_type == 'categorical' else n_samples // min_samples_per_class_per_fold
    
    if max_folds_overall < 2:
        raise ValueError("Not enough samples to perform cross-validation with the given minimum samples per fold.")
    
    # Final number of folds
    num_folds = min(base_folds, max_folds_overall)
    
    return num_folds
