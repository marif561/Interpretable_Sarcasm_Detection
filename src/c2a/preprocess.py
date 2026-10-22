"""Text cleaning (paper Sect. 3.2).

English headlines: lower-case, remove web links and every non-alphanumeric character.
Hindi tweets: the same, but Devanagari letters and **emojis are kept**, because emojis
are strong sarcasm cues in the Twitter data.
"""
from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
EN_KEEP_RE = re.compile(r"[^a-z0-9\s]")
# Devanagari block + common emoji / symbol ranges
HI_KEEP_RE = re.compile(r"[^0-9a-z\sऀ-ॿ\U0001F000-\U0001FAFF☀-➿‍️]")


def clean_en(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text.lower())
    return WS_RE.sub(" ", EN_KEEP_RE.sub(" ", text)).strip()


def clean_hi(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text.lower())
    text = re.sub(r"@\w+|#", " ", text)
    return WS_RE.sub(" ", HI_KEEP_RE.sub(" ", text)).strip()


CLEANERS = {"en": clean_en, "hi": clean_hi}
