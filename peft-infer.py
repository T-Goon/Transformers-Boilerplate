from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
import torch

peft_model_id = "peft-ft-model"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 #  llm_int8_threshold=6.0,
                                                 #  llm_int8_has_fp16_weight=False,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type="nf4",
                                             ),
                                             trust_remote_code=True)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,
                                          padding_side='left')
# for "tiiuae/falcon-7b-instruct"
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(
    [
        "Give me instructions on how to make a sandwich.",
        "Two things are infinite: "
    ],
    return_tensors="pt",
    # for "tiiuae/falcon-7b-instruct"
    return_token_type_ids=False,
    padding=True)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"].to("cuda"),
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=200,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        ))
    output_decoded = tokenizer.batch_decode(
        outputs.detach().cpu().numpy(), skip_special_tokens=True)
    for out in output_decoded:
        print(out)
        print()
