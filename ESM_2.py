import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

class ProteinFunctionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, label2id):
        """
        csv_file: Path to your filtered CSV with columns:
                  'structureId', 'chainId', 'sequence', and 'classification'
        tokenizer: ESM-2 tokenizer.
        label2id: Dictionary mapping classification labels to numeric IDs.
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.label2id = label2id
        
        # Filter out rows with missing sequences or classification information.
        self.df.dropna(subset=["sequence", "classification"], inplace=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["sequence"]).strip()
        label_str = str(row["classification"]).strip()
        
        # Tokenize the sequence. Using add_special_tokens=True (if needed).
        inputs = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Map the classification to its numeric label. If not found, default to 0.
        label_id = self.label2id.get(label_str, 0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

def main():
    model_id = "facebook/esm2_t33_650M_UR50D"
    config = AutoConfig.from_pretrained(model_id)
    
    # Define your possible classifications (update this list to include all classes in your CSV)
    possible_labels = [
        "HYDROLASE", "TRANSFERASE", "DNA", "IMMUNE SYSTEM"
        # Add additional classification labels as needed.
    ]
    label2id = {lbl: i for i, lbl in enumerate(possible_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    
    # Update configuration with number of labels
    config.num_labels = len(label2id)
    config.id2label = id2label
    config.label2id = label2id
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)
    
    # Check GPU usage and move model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    
    # Load dataset from filtered CSV
    csv_file = "filtered_processed_protein_data.csv"
    dataset = ProteinFunctionDataset(csv_file, tokenizer, label2id)
    
    # Split dataset into training (90%) and validation (10%) sets.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Set up training arguments with reduced batch sizes and gradient accumulation to help with memory.
    training_args = TrainingArguments(
        output_dir="./esm2-function-classification",
        num_train_epochs=3,
        per_device_train_batch_size=1,   # Reduced batch size to limit GPU memory usage.
        per_device_eval_batch_size=1,      # Reduced evaluation batch size.
        gradient_accumulation_steps=8,     # Effective batch size = 8.
        learning_rate=1e-4,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        fp16=True,                        # Enable mixed precision training.
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Fine-tune the model.
    trainer.train()
    
    # Evaluate the model on the validation set.
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    # Save the fine-tuned model.
    trainer.save_model("./esm2-function-classification-final")
    
    # Print total parameter count.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params:,}")

if __name__ == "__main__":
    main()
