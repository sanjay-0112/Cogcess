"""
Sanity tests for lexical_scorer.

Run from repo root:  python text_branch/test_lexical_scorer.py

By default uses the real `wordfreq` package. If it is not installed, falls back
to a small hand-written Zipf table (approximate values) so the logic can still
be checked.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from lexical_scorer import score_text, tokenize, DIFFICULT_THRESHOLD

STUB_ZIPF = {
    # function words / very common
    "a": 7.6, "the": 7.7, "and": 7.5, "to": 7.5, "of": 7.4, "it": 7.2,
    "was": 7.0, "we": 7.1, "for": 7.1, "by": 6.8, "or": 6.9, "in": 7.4,
    "this": 7.1, "that": 7.2, "thou": 4.3, "thy": 4.4, "thee": 4.0,
    "is": 7.3, "be": 7.0, "are": 7.0, "from": 7.0, "now": 6.8, "art": 5.4,
    "own": 6.0, "with": 7.2, "within": 5.6, "where": 6.6, "might": 6.2,
    "only": 6.5, "too": 6.6, "sweet": 5.3, "self": 5.4, "world": 6.2,
    "s": 6.0, "bud": 4.1, "else": 5.7, "eat": 5.4, "due": 5.5, "grave": 4.7,
    "pity": 4.2, "thine": 3.4, "foe": 3.5, "cruel": 4.6, "lies": 5.3,
    "making": 5.9, "famine": 3.4, "abundance": 3.7, "feed": 5.2,
    "light": 6.0, "flame": 4.5, "fuel": 4.8, "substantial": 4.2,
    "self-substantial": 0.0, "fairest": 3.6, "creatures": 4.6, "desire": 5.0,
    "increase": 5.3, "thereby": 4.0, "beauty": 5.3, "rose": 5.0,
    "never": 6.5, "die": 5.7, "riper": 2.6, "time": 6.9, "decease": 3.0,
    "tender": 4.2, "heir": 4.0, "bear": 5.4, "memory": 5.1, "contracted": 3.3,
    "bright": 5.2, "eyes": 5.9, "fed": 4.9, "eyes": 5.9,
    "fresh": 5.2, "ornament": 3.5, "herald": 3.6, "gaudy": 3.0,
    "spring": 5.2, "buriest": 0.0, "content": 5.3, "churl": 2.3,
    "makest": 0.0, "waste": 4.9, "niggarding": 0.0, "glutton": 3.0,
    "although": 5.3, "raining": 3.9, "heavily": 4.3, "went": 5.7, "walk": 4.9,
    "ne": 3.0, "might": 6.2, "thereby": 4.0, "creatures": 4.6,
}

SONNET = (
    "From fairest creatures we desire increase, That thereby beauty\"s rose "
    "might never die, But as the riper should by time decease, His tender "
    "heir might bear his memory. Feed\"st thy light\"s flame with "
    "self-substantial fuel, Making a famine where abundance lies, Thyself "
    "thy foe, to thy sweet self too cruel. Thou that art now the world\"s "
    "fresh ornament And only herald to the gaudy spring, Within thine own "
    "bud buriest thy content And, tender churl, makest waste in niggarding. "
    "Pity the world, or else this glutton be, To eat the world\"s due, by "
    "the grave and thee."
)

EASY = "Although it was raining heavily, we went for a walk."


def main():
    try:
        import wordfreq  # noqa: F401
        zipf_fn = None
        print("Using real wordfreq.\n")
    except ImportError:
        zipf_fn = lambda w: STUB_ZIPF.get(w, 4.0)
        print("wordfreq not installed -> using approximate stub table.\n")

    # --- tokenizer: no stray "s" from broken / curly apostrophes ---
    assert "s" not in tokenize("the world\"s due"), tokenize("the world\"s due")
    assert "s" not in tokenize("the world\u2019s due")
    assert "s" not in tokenize("the world's due")
    assert tokenize("don't stop") == ["stop"]
    print("tokenizer: OK")

    easy = score_text(EASY, zipf_fn)
    hard = score_text(SONNET, zipf_fn)

    for name, r in (("EASY", easy), ("SONNET", hard)):
        print(f"\n{name}")
        print(f"  mean {r['mean']:.1f}  median {r['median']:.1f}  "
              f"p90 {r['p90']:.1f}  max {r['max']:.1f}")
        print(f"  difficult words (>= {DIFFICULT_THRESHOLD:.0f}): "
              f"{r['difficult_ratio'] * 100:.1f}%")
        print("  hardest:", [(w, round(s)) for w, s in r["hardest"]])

    assert hard["mean"] > easy["mean"] + 8, "hard text should score clearly higher"
    assert hard["difficult_ratio"] > easy["difficult_ratio"]
    assert hard["max"] > easy["max"]
    print("\nordering checks: OK")


if __name__ == "__main__":
    main()
