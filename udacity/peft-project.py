# parameter-efficient fine-tuning using the Hugging Face peft library
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load a pre-trained BERT model and tokenizer
# MODEL_NAME = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Load dataset (IMDB sentiment classification)
# dataset = load_dataset("imdb")
# train_data, test_data = dataset["train"], dataset["test"]
dataset = load_dataset("imdb")
print(dataset)