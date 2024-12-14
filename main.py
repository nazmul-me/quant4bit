import os
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force using GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_id = "Salesforce/codegen-350M-multi"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
org_model = AutoModelForCausalLM.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config).to(device)

def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')
print_size_of_model(org_model)
print_size_of_model(model)

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print_size_of_model(model)
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(model)
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    # target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config).to(device)
print_trainable_parameters(model)

from datasets import load_dataset
raw_datasets = {
    "train": load_dataset("json", data_files="data/code_segments_humaneval.json", split=f"train[:{80}%]"),
    "validation": load_dataset("json", data_files="data/code_segments_humaneval.json", split=f"train[{-20}%:]"),
}
print(raw_datasets)

train_dataset = raw_datasets["train"]
validation_dataset = raw_datasets["validation"]
train_dataset = train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
# Add labels for causal language modeling
train_dataset = train_dataset.map(
    lambda samples: {
        'labels': samples['input_ids'],
        'attention_mask': samples['attention_mask']
    }
)
validation_dataset = validation_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
print(train_dataset)
print(validation_dataset)


# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token
# Create the DataCollatorWithPadding
data_collator = transformers.DataCollatorWithPadding(tokenizer)

# Custom DataCollator with label handling
class CustomDataCollator:
    def __init__(self, data_collator, device):
        self.data_collator = data_collator
        self.device = device

    def __call__(self, batch):
        batch = self.data_collator(batch)
        # Ensure labels are included
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids']
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}

# Create the custom data collator
custom_data_collator = CustomDataCollator(data_collator, device)

trainer = transformers.Trainer(
    model=model.to(device),
    train_dataset=train_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        do_train=True,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        dataloader_pin_memory=False,
        output_dir="outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=custom_data_collator
    # data_collator=data_collator #transformers.default_data_collator
    # data_collator= transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
# Add device context
with torch.cuda.device(device):
    model.config.use_cache = False
    trainer.train()