"""
HTTP API for the search UI.

Run from repo root::

  pip install -r requirements-api.txt
  export PYTHONPATH=ai/src
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --app-dir ai/src
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure `retrieval` / `company_search` resolve when started via uvicorn.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from company_search import run_company_table_search  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="Built World Search API", version="0.1.0")

_raw_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()] if _raw_origins != "*" else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search string")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/search")
def post_search(body: SearchRequest) -> dict:
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must be non-empty")
    try:
        return run_company_table_search(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
