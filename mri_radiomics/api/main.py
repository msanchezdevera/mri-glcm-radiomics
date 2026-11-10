"""FastAPI service.

  GET  /                  minimal web UI
  GET  /health            liveness + model metrics
  POST /predict           one-shot classification of an uploaded image
  POST /cases             upload an image, store it as a case for the agent
  GET  /cases             list cases (long-term memory)
  GET  /cases/{case_id}   one case
  POST /agent/chat        talk to the agent; returns the answer and the step trace
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.harness import Agent
from ..agent.llm import get_llm
from ..agent.memory import Store
from ..model import RadiomicsModel, get_model

MAX_BYTES = 10 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png"}
STATIC = Path(__file__).parent / "static"
SAMPLES = Path(__file__).resolve().parents[2] / "assets" / "samples"

app = FastAPI(title="MRI Radiomics Agent", version="1.0.0",
              description="GLCM texture model + tool-using agent. Research demo, not a medical device.")


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(os.getenv("RADIOMICS_DB", "var/agent.db"))


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    return Agent(llm=get_llm(), store=get_store(), model=get_model())


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)
    case_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[str]
    stopped: str
    trace: list[dict]


async def _featurize(file: UploadFile, model: RadiomicsModel) -> dict:
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"unsupported type {file.content_type}; send JPEG or PNG")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            return model.featurize(Path(tmp.name))
        except Exception as e:
            raise HTTPException(422, f"could not read image: {e}") from e


if SAMPLES.is_dir():
    app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(model: RadiomicsModel = Depends(get_model)):
    return {"status": "ok", "llm": get_llm().name, "model_metrics": model.metrics}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: RadiomicsModel = Depends(get_model)):
    feats = await _featurize(file, model)
    return {**model.predict(feats).as_dict(), "drivers": model.explain(feats, 5)}


@app.post("/cases", status_code=201)
async def create_case(file: UploadFile = File(...), model: RadiomicsModel = Depends(get_model),
                      store: Store = Depends(get_store)):
    feats = await _featurize(file, model)
    pred = model.predict(feats).as_dict()
    case_id = store.add_case(file.filename or "upload", feats, pred)
    return {"case_id": case_id, "prediction": pred}


@app.get("/cases")
def list_cases(label: str | None = None, limit: int = 20, store: Store = Depends(get_store)):
    return store.list_cases(label, limit)


@app.get("/cases/{case_id}")
def get_case(case_id: str, store: Store = Depends(get_store)):
    case = store.get_case(case_id)
    if case is None:
        raise HTTPException(404, "case not found")
    return case


@app.post("/agent/chat", response_model=ChatResponse)
def chat(req: ChatRequest, agent: Agent = Depends(get_agent)):
    result = agent.run(req.session_id, req.message, req.case_id)
    return ChatResponse(reply=result.reply, tool_calls=result.tool_calls,
                        stopped=result.stopped, trace=result.trace())
