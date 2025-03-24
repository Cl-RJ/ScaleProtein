import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ProteinFunctionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, label2id, max_samples=None):
        """
        csv_file: Path to your CSV with columns:
                  'structureId', 'chainId', 'sequence', and 'classification'
        tokenizer: ESM-2 tokenizer.
        label2id: Dictionary mapping classification labels to numeric IDs.
        max_samples: Maximum number of samples to include (for limiting dataset size)
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.label2id = label2id
        
        # Filter out rows with missing sequences or classification information
        self.df.dropna(subset=["sequence", "classification"], inplace=True)
        
        # Limit to max_samples if specified
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.head(max_samples)
            print(f"Dataset limited to first {max_samples} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["sequence"]).strip()
        label_str = str(row["classification"]).strip()
        
        # Tokenize the sequence
        inputs = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Map the classification to its numeric label
        label_id = self.label2id.get(label_str, 0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "text_label": label_str  # Store text label for reporting
        }

def evaluate_model(model_path, test_data_path, max_samples=1000):
    """
    Evaluate the fine-tuned model on a test dataset.
    
    Args:
        model_path: Path to the saved model
        test_data_path: Path to the test CSV file
        max_samples: Maximum number of samples to evaluate (default: 1000)
    """
    # Load the model configuration to get label mappings
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    # Extract label mappings from the model config
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Load test dataset with sample limit
    test_dataset = ProteinFunctionDataset(test_data_path, tokenizer, label2id, max_samples=max_samples)
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Collect predictions and true labels
    all_predictions = []
    all_true_labels = []
    all_text_labels = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            true_label = sample["labels"].item()
            text_label = sample["text_label"]
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
            all_predictions.append(predicted_class)
            all_true_labels.append(true_label)
            all_text_labels.append(text_label)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")
    
    # Calculate accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Get classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(all_true_labels, all_predictions, target_names=target_names)
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Print per-class accuracy
    class_counts = {}
    class_correct = {}
    
    for true, pred, text in zip(all_true_labels, all_predictions, all_text_labels):
        if text not in class_counts:
            class_counts[text] = 0
            class_correct[text] = 0
            
        class_counts[text] += 1
        if true == pred:
            class_correct[text] += 1
    
    print("\nPer-Class Accuracy:")
    for cls in sorted(class_counts.keys()):
        accuracy = class_correct[cls] / class_counts[cls]
        print(f"{cls}: {accuracy:.4f} ({class_correct[cls]}/{class_counts[cls]})")
    
    # Return metrics for further analysis if needed
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "predictions": all_predictions,
        "true_labels": all_true_labels
    }

if __name__ == "__main__":
    # Path to your fine-tuned model
    model_path = "./esm2-function-classification-final"
    
    # Path to your test data
    test_data_path = "processed_protein_data.csv"  # You can use your main CSV or a dedicated test set
    
    # Run evaluation with a limit of 1000 samples
    results = evaluate_model(model_path, test_data_path, max_samples=1000) 