import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "tiiuae/falcon-7b-instruct"
# model_name = "NousResearch/Nous-Hermes-13b"
# model_name = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"

print('Loading model')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # bnb_4bit_quant_type="fp4",
    # double quantization, save .4 bits per param
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

# Pytorch better transformers from optimum library
# supported models: https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models
# model = model.to_bettertransformer()

time_start: float = time.perf_counter()
print('Creating pipeline')
generate = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
print('Doing inference')

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
# )
sequences = generate(
    """Give me instructions on how to make a sandwich.""",
    max_length=500,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

time_end: float = time.perf_counter()
print(f'Run time: {time_end - time_start:0.4f} seconds')