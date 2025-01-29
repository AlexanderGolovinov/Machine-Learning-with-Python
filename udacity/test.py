from peft import LoraConfig
config = LoraConfig()

MODEL_NAME = "bert-base-uncased"
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

from peft import get_peft_model
lora_model = get_peft_model(model, config)

lora_model.print_trainable_parameters()

lora_model.save_pretrained("bert-lora")

from peft import AutoPeftModelForSequenceClassification
lora_model = AutoPeftModelForSequenceClassification.from_pretrained("bert-lora")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs = tokenizer("This is a great movie!", return_tensors="pt")
outputs = model(**inputs).logits
print("Original Model Prediction:", outputs.argmax().item())

lora_outputs = lora_model(**inputs).logits
print("Fine-Tuned Model Prediction:", lora_outputs.argmax().item())
