from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_esco_csv
from .text import highlighted_html, split_sentences, tokenize


@dataclass(frozen=True)
class MatchResult:
    code: str
    label: str
    description: str
    score: float
    lexical_score: float
    semantic_score: float | None
    evidence_html: str
    concept_uri: str
    hierarchy: tuple[str, ...]


class EscoMatcher:
    """Hybrid ESCO retriever with an offline lexical fallback."""

    def __init__(
        self,
        csv_path: str | Path,
        use_semantic: bool = False,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.frame = load_esco_csv(csv_path)
        self.documents = [
            " ".join(
                (
                    row.preferredLabel_esp,
                    row.altLabels_esp,
                    row.description_esp,
                )
            )
            for row in self.frame.itertuples()
        ]
        self.labels = [
            f"{row.preferredLabel_esp} {row.altLabels_esp}"
            for row in self.frame.itertuples()
        ]
        self.doc_tokens = [tokenize(document) for document in self.documents]
        self.alias_tokens: list[list[str]] = []
        self.alias_owners: list[int] = []
        for owner, row in enumerate(self.frame.itertuples()):
            aliases = [
                *str(row.preferredLabel_esp).split("/"),
                *str(row.altLabels_esp).splitlines(),
            ]
            for alias in aliases:
                tokens = tokenize(alias)
                if tokens:
                    self.alias_tokens.append(tokens)
                    self.alias_owners.append(owner)
        self._build_lexical_index()
        self.semantic_model: Any | None = None
        self.semantic_embeddings: np.ndarray | None = None
        self.semantic_error: str | None = None
        if use_semantic:
            self._build_semantic_index(model_name)

    @property
    def semantic_enabled(self) -> bool:
        return self.semantic_model is not None

    def _build_lexical_index(self) -> None:
        document_count = len(self.doc_tokens)
        frequencies: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            frequencies.update(set(tokens))
        self.idf = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = np.asarray([len(tokens) for tokens in self.doc_tokens])
        self.average_doc_length = float(self.doc_lengths.mean()) or 1.0
        self.label_term_frequencies = [Counter(tokens) for tokens in self.alias_tokens]
        self.label_lengths = np.asarray([len(tokens) for tokens in self.alias_tokens])
        self.average_label_length = float(self.label_lengths.mean()) or 1.0

    def _build_semantic_index(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.semantic_model = SentenceTransformer(model_name)
            self.semantic_embeddings = np.asarray(
                self.semantic_model.encode(
                    self.documents,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
        except Exception as error:
            self.semantic_model = None
            self.semantic_embeddings = None
            self.semantic_error = str(error)

    def _bm25_scores(
        self,
        query_tokens: list[str],
        term_frequencies: list[Counter[str]],
        lengths: np.ndarray,
        average_length: float,
    ) -> np.ndarray:
        scores = np.zeros(len(term_frequencies), dtype=float)
        k1, b = 1.5, 0.75
        for index, frequencies in enumerate(term_frequencies):
            length_factor = k1 * (
                1 - b + b * lengths[index] / average_length
            )
            score = 0.0
            for term in set(query_tokens):
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (k1 + 1) / (frequency + length_factor)
                    )
            scores[index] = score
        maximum = float(scores.max())
        return scores / maximum if maximum > 0 else scores

    def _lexical_scores(self, query_tokens: list[str]) -> np.ndarray:
        document_scores = self._bm25_scores(
            query_tokens,
            self.term_frequencies,
            self.doc_lengths,
            self.average_doc_length,
        )
        alias_scores = self._bm25_scores(
            query_tokens,
            self.label_term_frequencies,
            self.label_lengths,
            self.average_label_length,
        )
        label_scores = np.zeros(len(self.documents), dtype=float)
        for alias_index, owner in enumerate(self.alias_owners):
            label_scores[owner] = max(label_scores[owner], alias_scores[alias_index])
        query_terms = set(query_tokens)
        coverage_scores = np.asarray(
            [
                len(query_terms.intersection(frequencies)) / len(query_terms)
                for frequencies in self.term_frequencies
            ]
        )
        return (
            0.6 * document_scores
            + 0.25 * label_scores
            + 0.15 * coverage_scores
        )

    def _semantic_scores(self, query: str) -> np.ndarray | None:
        if self.semantic_model is None or self.semantic_embeddings is None:
            return None
        query_embedding = self.semantic_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        similarities = self.semantic_embeddings @ query_embedding
        return np.clip((similarities + 1) / 2, 0, 1)

    def _evidence(self, description: str, query_tokens: list[str]) -> str:
        query_set = set(query_tokens)
        sentences = split_sentences(description)
        if not sentences:
            return highlighted_html(description, query_set)

        def sentence_score(sentence: str) -> tuple[float, int]:
            terms = tokenize(sentence)
            overlap = sum(self.idf.get(term, 0.0) for term in set(terms) & query_set)
            return overlap, -len(terms)

        best = max(sentences, key=sentence_score)
        return highlighted_html(best, query_set)

    def search(self, query: str, top_k: int = 5) -> list[MatchResult]:
        query = query.strip()
        query_tokens = tokenize(query)
        if len(query_tokens) < 2:
            raise ValueError("Describe la ocupación con al menos dos palabras informativas.")
        top_k = max(1, min(int(top_k), len(self.frame)))

        lexical = self._lexical_scores(query_tokens)
        query_sentences = [
            tokenize(sentence)
            for sentence in split_sentences(query)
            if len(tokenize(sentence)) >= 2
        ]
        if len(query_sentences) > 1:
            sentence_scores = np.asarray(
                [self._lexical_scores(tokens) for tokens in query_sentences]
            )
            sentence_lengths = np.asarray(
                [len(set(tokens)) for tokens in query_sentences], dtype=float
            )
            sentence_weights = np.minimum(1.0, sentence_lengths / 8.0)
            sentence_scores *= sentence_weights[:, np.newaxis]
            lexical = 0.4 * lexical + 0.6 * sentence_scores.max(axis=0)
        semantic = self._semantic_scores(query)
        combined = lexical if semantic is None else (0.35 * lexical + 0.65 * semantic)
        ranking = np.argsort(combined)[::-1][:top_k]

        hierarchy_columns = [f"grupo{number}" for number in range(1, 9)]
        results: list[MatchResult] = []
        for index in ranking:
            row = self.frame.iloc[int(index)]
            hierarchy = tuple(
                str(row[column])
                for column in hierarchy_columns
                if column in row and str(row[column]).strip()
            )
            results.append(
                MatchResult(
                    code=row["code"],
                    label=row["preferredLabel_esp"],
                    description=row["description_esp"],
                    score=float(combined[index]),
                    lexical_score=float(lexical[index]),
                    semantic_score=(
                        float(semantic[index]) if semantic is not None else None
                    ),
                    evidence_html=self._evidence(
                        row["description_esp"], query_tokens
                    ),
                    concept_uri=str(row.get("conceptUri", "")),
                    hierarchy=hierarchy,
                )
            )
        return results
