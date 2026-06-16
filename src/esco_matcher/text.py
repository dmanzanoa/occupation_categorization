from __future__ import annotations

import html
import re
import unicodedata


SPANISH_STOPWORDS = {
    "a",
    "al",
    "algo",
    "ante",
    "como",
    "con",
    "de",
    "del",
    "desde",
    "donde",
    "el",
    "ella",
    "en",
    "entre",
    "es",
    "esta",
    "este",
    "hacer",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mediante",
    "mi",
    "mis",
    "para",
    "pero",
    "persona",
    "personas",
    "por",
    "primera",
    "primero",
    "primer",
    "que",
    "se",
    "sin",
    "sobre",
    "soy",
    "su",
    "sus",
    "tambien",
    "un",
    "una",
    "uno",
    "y",
    "yo",
}

_WORD_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )


def stem_spanish_word(word: str) -> str:
    """Apply conservative suffix stripping for retrieval, not linguistic analysis."""
    word = strip_accents(word.lower())
    suffixes = (
        "amientos",
        "imientos",
        "aciones",
        "aduras",
        "adoras",
        "adores",
        "ancias",
        "antes",
        "amiento",
        "imiento",
        "acion",
        "adura",
        "ancia",
        "ante",
        "mente",
        "idades",
        "idad",
        "ando",
        "iendo",
        "ados",
        "adas",
        "idos",
        "idas",
        "ador",
        "adora",
        "istas",
        "ista",
        "es",
        "os",
        "as",
        "ar",
        "er",
        "ir",
        "en",
        "an",
        "e",
        "o",
        "a",
    )
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    normalized = strip_accents(str(text).lower())
    return [
        stem_spanish_word(token)
        for token in _WORD_RE.findall(normalized)
        if len(token) > 1 and token not in SPANISH_STOPWORDS
    ]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(str(text)) if part.strip()]


def highlighted_html(text: str, query_tokens: set[str]) -> str:
    """Escape text and mark complete words shared with the query."""
    pieces: list[str] = []
    cursor = 0
    for match in _WORD_RE.finditer(str(text)):
        pieces.append(html.escape(str(text)[cursor : match.start()]))
        word = match.group(0)
        escaped_word = html.escape(word)
        if stem_spanish_word(word) in query_tokens:
            pieces.append(f"<mark>{escaped_word}</mark>")
        else:
            pieces.append(escaped_word)
        cursor = match.end()
    pieces.append(html.escape(str(text)[cursor:]))
    return "".join(pieces)
