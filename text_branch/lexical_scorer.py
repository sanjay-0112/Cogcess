"""
Cogcess lexical difficulty scorer (v5).

Replaces the learned Mendeley lexical model for *display* statistics.

Why: the old model was trained on `difficult_ug`, a sparse, context-dependent
label (83% zeros), with a "frequency" feature that was only a count inside the
3,921-word Mendeley corpus. It collapsed to ~5/100 for every text.

This scorer is deterministic and uses real English word frequency
(wordfreq Zipf scale, ~0 = unseen, ~7+ = "the"), plus word length, syllables,
and a small archaic-word penalty. Scores are 0-100.

    pip install wordfreq
"""

import re
import numpy as np


# A word at or above this score counts as "difficult".
DIFFICULT_THRESHOLD = 40.0

# Zipf >= EASY_ZIPF is treated as fully familiar; the frequency term grows
# linearly as Zipf falls toward 1.
EASY_ZIPF = 6.0
ZIPF_RANGE = 5.0

# Component weights (sum to 1.0)
W_FREQ = 0.65
W_LEN = 0.20
W_SYL = 0.15

ARCHAIC_BONUS = 25.0
ARCHAIC = {
    "thy", "thee", "thou", "thine", "thyself", "ye", "doth", "dost", "hath",
    "hast", "hadst", "wilt", "shalt", "wherefore", "whither", "whence",
    "thence", "ere", "nay", "yea", "oer", "neer", "twixt", "betwixt",
    "methinks", "prithee", "anon", "albeit",
}

_APOSTROPHE_LIKE = re.compile(r"(?<=[A-Za-z])[\"'\u2019\u2018`\u00b4](?=[A-Za-z])")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_CONTRACTION_SUFFIXES = {"s", "t", "re", "ll", "ve", "d", "m"}


def tokenize(text):
    """
    Lowercase word tokens with apostrophes handled properly.

    "world's", "world\u2019s" and the broken 'world"s' all become "world"
    (no stray "s" token). Negations like "don't" are dropped, since they are
    always very common function words.
    """
    text = _APOSTROPHE_LIKE.sub("'", str(text).replace("\n", " "))

    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if "'" in tok:
            base, _, suffix = tok.partition("'")
            if suffix in _CONTRACTION_SUFFIXES:
                if suffix == "t" and base.endswith("n"):
                    continue
                tok = base
            else:
                tok = tok.replace("'", "")   # o'er -> oer
        tokens.append(tok)

    return tokens


def count_syllables(word):
    """Cheap vowel-group syllable estimate (good enough for scoring)."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)

    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1

    return max(count, 1)


def _default_zipf(word):
    try:
        from wordfreq import zipf_frequency
    except ImportError as exc:
        raise ImportError(
            "lexical_scorer needs the 'wordfreq' package: pip install wordfreq"
        ) from exc

    return zipf_frequency(word, "en")


def word_difficulty(word, zipf):
    """Difficulty of one word on a 0-100 scale. zipf == 0 means unseen."""
    freq_term = float(np.clip((EASY_ZIPF - zipf) / ZIPF_RANGE, 0.0, 1.0))
    len_term = float(np.clip((len(word) - 4) / 8.0, 0.0, 1.0))
    syl_term = float(np.clip((count_syllables(word) - 1) / 4.0, 0.0, 1.0))

    score = 100.0 * (W_FREQ * freq_term + W_LEN * len_term + W_SYL * syl_term)

    if word in ARCHAIC:
        score += ARCHAIC_BONUS

    return min(score, 100.0)


def score_text(text, zipf_fn=None, top_n=5):
    """
    Score every word in `text`. Returns None if there are no words.

    zipf_fn: optional callable word -> Zipf frequency (for testing).
    """
    words = tokenize(text)

    if not words:
        return None

    zipf_fn = zipf_fn or _default_zipf

    zipfs = {w: float(zipf_fn(w)) for w in set(words)}
    scores = np.asarray(
        [word_difficulty(w, zipfs[w]) for w in words],
        dtype=np.float32,
    )

    # hardest distinct words
    best = {}
    for w, s in zip(words, scores):
        best[w] = max(best.get(w, 0.0), float(s))
    hardest = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    return {
        "words": words,
        "scores": scores,
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "p90": float(np.percentile(scores, 90)),
        "max": float(np.max(scores)),
        "difficult_ratio": float(np.mean(scores >= DIFFICULT_THRESHOLD)),
        "hardest": hardest,
        "unseen_words": int(sum(1 for w in set(words) if zipfs[w] <= 0.0)),
    }
