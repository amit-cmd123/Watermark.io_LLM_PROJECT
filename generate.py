"""Generation adapter; no fine tuning, no watermark during attacks."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from model_loading import load_pretrained


class Generator:
    def __init__(self, model_name, revision="main"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = load_pretrained(AutoTokenizer, model_name, revision)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = load_pretrained(
            AutoModelForCausalLM, model_name, revision, torch_dtype=dtype
        ).to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, seed, settings, processor=None):
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        if self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
            inputs = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        processors = LogitsProcessorList([] if processor is None else [processor])
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, logits_processor=processors, do_sample=True,
                temperature=settings.temperature, top_p=settings.top_p, top_k=0,
                max_new_tokens=settings.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id)
        ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        return text, ids
