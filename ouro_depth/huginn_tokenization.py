"""Native plain-completion encoding for the pinned Huginn tokenizer."""
from .train import encode_rows


def encode_native_rows(rows, tokenizer, max_length):
    """Use the official BOS postprocessor and retain the one-token target rule.

    Assert the exact pinned transformation; never silently add an EOS, chat
    template, answer prefix, truncation, or duplicate BOS.
    """
    if tokenizer.bos_token_id != 65504 or tokenizer.pad_token_id != 65509:
        raise ValueError('Unexpected pinned Huginn special-token IDs')
    encoded, answers = encode_rows(rows, tokenizer, max_length)
    for item in encoded:
        prompt = item['row']['prompt']
        native = tokenizer.encode(prompt, add_special_tokens=True)
        if native != [65504] + item['ids'] or native.count(65504) != 1:
            raise ValueError('Native encoding must prepend exactly one BOS')
        joint = tokenizer.encode(prompt + ' ' + item['row']['answer'], add_special_tokens=True)
        if joint != native + [item['target']]:
            raise ValueError('Native answer boundary changed')
        if len(native) > max_length:
            raise ValueError('Native prompt exceeds fixed width; refusing truncation')
        item['ids'] = native
    return encoded, answers
