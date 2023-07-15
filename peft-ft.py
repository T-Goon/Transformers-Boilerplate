from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import time
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import torch.nn as nn
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

time_start: float = time.perf_counter()

print('Loading Model')

# free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
# max_memory = f"{free_in_GB-2}GB"

# n_gpus = torch.cuda.device_count()
# max_memory = {i: max_memory for i in range(n_gpus)}

# model_name: str = "facebook/opt-350m"
model_name: str = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # max_memory=max_memory,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        # llm_int8_threshold=6.0,
        # llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    trust_remote_code=True,
    device_map='auto'
)


# for "tiiuae/falcon-7b-instruct"
tokenizer.pad_token = tokenizer.eos_token

print('post processing')
print(model)

# for param in model.parameters():
#     param.requires_grad = False  # freeze the model - train adapters later
#     if param.ndim == 1:
#         # cast the small parameters (e.g. layernorm) to fp32 for stability
#         param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
# model.model.decoder.project_in = lambda x: x.requires_grad_(True)


# class CastOutputToFloat(nn.Sequential):
#     def forward(self, x):
#         return super().forward(x).to(torch.float32)


# model.lm_head = CastOutputToFloat(model.lm_head)

model = prepare_model_for_kbit_training(model)

print('Configuring LORA')

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


config = LoraConfig(
    r=8,
    lora_alpha=32,
    # for "facebook/opt-350m"
    # target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # for "tiiuae/falcon-7b-instruct"
    target_modules=["query_key_value"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Verifying the datatypes.
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items():
    total += v
for k, v in dtypes.items():
    print(k, v, v / total)

print('Training model')

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        # larger batch size = more vram usage
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=20,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        # Paging for better memory management
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
)
# silence the warnings. Please re-enable for inference!
model.config.use_cache = False
trainer.train()

# Save the model
model.save_pretrained("peft-ft-model")

print('Doing Inference')
batch = tokenizer("Two things are infinite:",
                  return_tensors="pt",
                  # for "tiiuae/falcon-7b-instruct"
                  return_token_type_ids=False
                  )
batch = batch.to('cuda')

# model.eval()
# with torch.cuda.amp.autocast():
# output_tokens = model.generate(**batch, max_new_tokens=100)

output_tokens = model.generate(
    **batch,
    generation_config=GenerationConfig(
        do_sample=True,
        max_new_tokens=100,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False
    ))
print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))

time_end: float = time.perf_counter()
print(f'Run time: {time_end - time_start:0.4f} seconds')
