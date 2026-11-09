"""Recursive paraphrase instruction and actual Marian back-translation."""
import torch
from transformers import MarianMTModel, MarianTokenizer


def paraphrase_prompt(text):
    return ("Rewrite the passage below using different wording and sentence structure. "
            "Preserve its meaning, facts, and approximately its length. "
            "Return only the rewritten passage without a preamble.\n\nPassage:\n" + text)


class Translator:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device).eval()
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        limit = getattr(self.model.config, "max_position_embeddings", 512)
        if inputs["input_ids"].shape[1] > limit:
            raise ValueError("Translation input exceeds context; reduce max-new-tokens. "
                             "No silent truncation was performed.")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, num_beams=4, do_sample=False,
                                         max_new_tokens=limit)
        if output[0, -1].item() != self.tokenizer.eos_token_id:
            raise ValueError("Translation reached output limit before EOS")
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
