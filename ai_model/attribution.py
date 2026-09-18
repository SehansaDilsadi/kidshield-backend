"""Model-derived explainability for KidShield's DistilBERT classifier.

Uses Integrated Gradients (Sundararajan et al., 2017) via Captum to attribute
the predicted class logit back to individual input tokens, so trigger words
are computed from the model's actual decision rather than a hardcoded list.
"""

import re
from typing import List, Tuple

import torch
from captum.attr import LayerIntegratedGradients

from .predictor import model, tokenizer

model.eval()

_SPECIAL_TOKENS = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}

# Function words that can pick up nonzero IG attribution without carrying
# any of the actual harmful signal a parent would want to see.
_STOPWORDS = {
    "you", "your", "yours", "so", "the", "a", "an", "is", "are", "was", "were",
    "am", "i", "to", "of", "and", "or", "it", "its", "this", "that", "in", "on",
    "for", "with", "at", "by", "be", "do", "does", "did", "my", "me", "we",
    "us", "he", "she", "they", "them", "his", "her", "their",
}


def _forward(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


_lig = LayerIntegratedGradients(_forward, model.distilbert.embeddings)


def _build_baseline(input_ids: torch.Tensor) -> torch.Tensor:
    """All-PAD baseline that keeps [CLS]/[SEP] in place — the standard IG
    reference point for BERT-family models."""
    baseline = input_ids.clone()
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}
    for i in range(baseline.shape[1]):
        if baseline[0, i].item() not in special_ids:
            baseline[0, i] = tokenizer.pad_token_id
    return baseline


def _merge_wordpieces(tokens: List[str], scores: List[float]) -> List[Tuple[str, float]]:
    """Collapse '##' continuation pieces back into whole words, summing scores."""
    words: List[Tuple[str, float]] = []
    for token, score in zip(tokens, scores):
        if token in _SPECIAL_TOKENS:
            continue
        if token.startswith("##") and words:
            prev_word, prev_score = words[-1]
            words[-1] = (prev_word + token[2:], prev_score + score)
        else:
            words.append((token, score))
    return words


def get_token_attributions(
    text: str,
    target_label_id: int,
    n_steps: int = 32,
    top_k: int = 5,
    min_score: float = 0.05,
) -> List[str]:
    """Return up to `top_k` words that most drove the model toward `target_label_id`.

    Returns an empty list if no token clears `min_score` — callers should treat
    that as "attribution had no strong signal" and fall back to another method.
    """
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    baseline_ids = _build_baseline(input_ids)

    attributions = _lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_label_id,
        n_steps=n_steps,
    )

    scores = attributions.sum(dim=-1).squeeze(0)
    norm = torch.norm(scores)
    if norm > 0:
        scores = scores / norm
    scores_list = scores.detach().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    words = _merge_wordpieces(tokens, scores_list)

    candidates = [
        (word, score) for word, score in words
        if score > min_score
        and re.search(r"[a-zA-Z]", word)
        and word.lower() not in _STOPWORDS
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)

    return [word for word, _ in candidates[:top_k]]
