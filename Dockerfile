FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mri_radiomics/ mri_radiomics/
COPY models/ models/
COPY skills/ skills/
COPY assets/samples/ assets/samples/

ENV LLM_PROVIDER=mock RADIOMICS_DB=/data/agent.db
VOLUME /data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "mri_radiomics.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
