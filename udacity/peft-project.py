# parameter-efficient fine-tuning using the Hugging Face peft library
from peft import LoraConfig
config = LoraConfig()

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")

from peft import get_peft_model
lora_model = get_peft_model(model, config)

lora_model.print_trainable_parameters()

lora_model.save_pretrained("gpt-lora")

from peft import AutoPeftModelForCausalLM
lora_model = AutoPeftModelForCausalLM.from_pretrained("gpt-lora")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello, my name is ", return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
print(tokenizer.batch_decode(outputs))