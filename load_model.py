import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)

from peft import (
    PeftModel
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=False,
)

device_map = {"": 0}
model = "bigcode/starcoderbase-1b"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model, quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
)

model_id = "peft-lora-starcoder15B-v2-personal-copilot-full"
model = PeftModel.from_pretrained(model, model_id, adapter_name="copilot")


if not hasattr(model, "hf_device_map"):
    model.cuda()
