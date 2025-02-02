import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification

# Ensure we use the correct device (CPU or MPS for Apple Silicon)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# -------------------------------
# 1. Loading the Pretrained Model
# -------------------------------
MODEL_NAME = "bert-base-uncased"

# Load BERT model for binary classification (IMDb Sentiment Analysis)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset("imdb")

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_data = dataset["train"].map(tokenize_function, batched=True)
test_data = dataset["test"].map(tokenize_function, batched=True)
train_data = train_data.select(range(500))
test_data = test_data.select(range(200))

# -------------------------------
# 2. Evaluating the Model Before Fine-Tuning
# -------------------------------
sample_text = "This is a fantastic movie!"
inputs = tokenizer(sample_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs).logits
    print("Original Model Prediction:", outputs.argmax().item())  # 0 (Negative) or 1 (Positive)

# -------------------------------
# 3. Creating a PEFT Config
# -------------------------------
config = LoraConfig()

# -------------------------------
# 4. Creating a PEFT Model
# -------------------------------
lora_model = get_peft_model(model, config)
lora_model.to(device)

# Print trainable parameters
lora_model.print_trainable_parameters()

# -------------------------------
# 5. Training the Model
# -------------------------------
training_args = TrainingArguments(
    output_dir="./bert-lora-results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

trainer.train()

# -------------------------------
# 6. Saving the Fine-Tuned Model
# -------------------------------
lora_model.save_pretrained("bert-lora")

# -------------------------------
# 7. Performing Inference with the Fine-Tuned Model
# -------------------------------
# Reload the fine-tuned model
lora_model = AutoPeftModelForSequenceClassification.from_pretrained("bert-lora")
lora_model.to(device)

# Perform inference
with torch.no_grad():
    lora_outputs = lora_model(**inputs).logits
    print("Fine-Tuned Model Prediction:", lora_outputs.argmax().item())
