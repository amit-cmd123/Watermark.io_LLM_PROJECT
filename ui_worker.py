"""Isolated dashboard jobs; delegates statistics and generation to the research code."""
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def execute(data):
    from detect import detect_ids, detect_text
    from watermark import WatermarkScheme, vocabulary_ids, make_processor
    import torch

    offline = data['mode'] == 'offline'
    settings = SimpleNamespace(**data)
    if offline:
        vocab = range(127)
        tokenizer = None
    else:
        from transformers import AutoTokenizer
        from model_loading import load_pretrained
        print('Loading tokenizer (first use may download files)…', flush=True)
        tokenizer = load_pretrained(AutoTokenizer, data['model'], data['revision'])
        vocab = vocabulary_ids(tokenizer)
    import engine
    key = data['key'].encode()
    name = data.get('scheme', 'v1')
    if name == 'v1':
        scheme = WatermarkScheme(vocab, key, data['gamma'])
        make_proc = lambda: make_processor(scheme, data['delta'])
        stats_of = lambda ids: detect_ids(ids, scheme, data['threshold'])
    else:
        scheme = engine.HashScheme(key, vocab, data['gamma'], context=1 if name == 'v2-h1' else 4,
                                   agg='min' if name == 'v2-min4' else 'chain')
        make_proc = lambda: engine.make_processor(scheme, data['delta'])
        stats_of = lambda ids: engine.detect(ids, scheme, data['threshold'])

    def statuses(ids):
        if name == 'v1':
            seen, out = set(), []
            for i, current in enumerate(ids):
                previous = ids[i - 1] if i else None
                pair = (previous, current)
                eligible = i > 0 and previous in scheme.eligible and current in scheme.eligible
                out.append('ignored' if not eligible or pair in seen else
                           'green' if current in scheme.green_set(previous) else 'ordinary')
                seen.add(pair)
            return out
        import numpy as np
        eligible, green, keys = scheme.flags_for_ids(ids)
        keep = eligible.copy()
        _, first = np.unique(keys[eligible], return_index=True)
        mask = np.zeros(int(eligible.sum()), bool)
        mask[first] = True
        keep[np.flatnonzero(eligible)] = mask
        return ['ignored'] * scheme.context + ['ignored' if not k else 'green' if g else 'ordinary'
                                               for k, g in zip(keep, green)]

    def score(ids, text):
        result = {'text': text, 'ids': ids, 'stats': stats_of(ids), 'scheme': name}
        result['tokens'] = [{'text': f'{c} ' if offline else tokenizer.decode([c]), 'status': s}
                            for c, s in zip(ids, statuses(ids))]
        if not offline:
            result['text_stats'] = stats_of(tokenizer.encode(text, add_special_tokens=False))
        return result

    action = data['action']
    if action == 'detect':
        text = data['text']
        if offline:
            ids = [int(x) for x in text.replace(',', ' ').split()]
            if any(x < 0 or x > 127 for x in ids):
                raise ValueError('Offline detection requires token IDs between 0 and 127 (127 is excluded padding).')
        else:
            ids = tokenizer.encode(text, add_special_tokens=False)
        return {'samples': {'Detected input': score(ids, text)}, 'offline': offline}

    if action == 'attack' and data['attack'] == 'translation':
        if offline:
            raise ValueError('Back-translation requires real translation models.')
        from attacks import Translator
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Loading English to French translation model…', flush=True)
        translator = Translator('Helsinki-NLP/opus-mt-en-fr', device)
        french = translator.translate(data['text'])
        del translator
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        print('Loading French to English translation model…', flush=True)
        translator = Translator('Helsinki-NLP/opus-mt-fr-en', device)
        text = translator.translate(french)
        return {'samples': {'Before attack': score(tokenizer.encode(data['text'], add_special_tokens=False), data['text']),
                            'Back-translated': score(tokenizer.encode(text, add_special_tokens=False), text)},
                'french': french, 'offline': False}

    print('Loading generation model…', flush=True)
    if offline:
        from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=128, n_positions=520, n_ctx=520,
            n_embd=32, n_layer=1, n_head=2, bos_token_id=None, eos_token_id=None, pad_token_id=127)).eval()

        def generate(prompt, seed, processor=None):
            torch.manual_seed(seed)
            with torch.inference_mode():
                ids = model.generate(torch.tensor([[1, 2]]), max_new_tokens=data['max_new_tokens'],
                    do_sample=True, top_k=0, top_p=data['top_p'], temperature=data['temperature'],
                    logits_processor=LogitsProcessorList([] if processor is None else [processor]),
                    pad_token_id=127)[0, 2:].tolist()
            return ' '.join(map(str, ids)), ids
    else:
        from generate import Generator
        generator = Generator(data['model'], data['revision'])
        generate = lambda prompt, seed, processor=None: generator.generate(prompt, seed, settings, processor)

    if action == 'generate':
        samples = {}
        for label, processor in [('Control', None), ('Watermarked', make_proc())]:
            print('Generating ' + label.lower() + '…', flush=True)
            text, ids = generate(data['prompt'], data['seed'], processor)
            samples[label] = score(ids, text)
        return {'samples': samples, 'offline': offline}
    if offline:
        raise ValueError('Recursive paraphrasing requires a pretrained language model.')
    from attacks import paraphrase_prompt
    text = data['text']
    samples = {'Before attack': score(tokenizer.encode(text, add_special_tokens=False), text)}
    for i in range(1, 4):
        print(f'Paraphrase round {i}…', flush=True)
        text, ids = generate(paraphrase_prompt(text), data['seed'] + i * 1000)
        samples[f'Paraphrase {i}'] = score(ids, text)
    return {'samples': samples, 'offline': False}


if __name__ == '__main__':
    request, output = map(Path, sys.argv[1:])
    output.write_text(json.dumps(execute(json.loads(request.read_text(encoding='utf-8')))), encoding='utf-8')
