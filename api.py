from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from esco_matcher import EscoMatcher  # noqa: E402


app = FastAPI(
    title="ESCO Occupation Matcher API",
    description=(
        "API para buscar ocupaciones ESCO a partir de una descripción libre en español."
    ),
    version="1.0.0",
)


class SearchRequest(BaseModel):
    description: str = Field(
        ...,
        description="Descripción libre de la ocupación en español.",
        min_length=3,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Número de resultados a devolver.",
    )
    semantic: bool = Field(
        default=False,
        description="Si es true, intenta usar embeddings semánticos para rerank.",
    )


class MatchResultResponse(BaseModel):
	code: str
	label: str
	description: str
	score: float
	lexical_score: float
	semantic_score: float | None
	evidence_html: str
	concept_uri: str
	hierarchy: list[str]
	ciuo08_code: str
	ciuo08_label: str
	ciuo08_cl_path: str
	joined_codes: str


class SearchResponse(BaseModel):
	semantic_requested: bool
	semantic_enabled: bool
	semantic_error: str | None
	count: int
	results: list[MatchResultResponse]


@lru_cache(maxsize=2)
def _get_matcher(use_semantic: bool) -> EscoMatcher:
    return EscoMatcher(ROOT / "data" / "codes_enriched.json", use_semantic=use_semantic)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    try:
        matcher = _get_matcher(payload.semantic)
        matches = matcher.search(payload.description, top_k=payload.top_k)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error interno: {error}") from error

    results = [
        MatchResultResponse(
            code=match.code,
            label=match.label,
            description=match.description,
            score=match.score,
            lexical_score=match.lexical_score,
            semantic_score=match.semantic_score,
            evidence_html=match.evidence_html,
            concept_uri=match.concept_uri,
            hierarchy=list(match.hierarchy),
            ciuo08_code=match.ciuo08_code,
            ciuo08_label=match.ciuo08_label,
            ciuo08_cl_path=match.ciuo08_cl_path,
            joined_codes=match.joined_codes,
        )
        for match in matches
    ]

    return SearchResponse(
        semantic_requested=payload.semantic,
        semantic_enabled=matcher.semantic_enabled,
        semantic_error=matcher.semantic_error,
        count=len(results),
        results=results,
    )


@app.get("/search", response_model=SearchResponse)
def search_get(
	description: str = Query(..., min_length=3),
	top_k: int = Query(5, ge=1, le=50),
	semantic: bool = Query(False),
) -> SearchResponse:
    return search(SearchRequest(description=description, top_k=top_k, semantic=semantic))
