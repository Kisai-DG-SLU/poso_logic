# Dockerfile pour le POC Triage Médical CHSA
FROM python:3.10-slim

WORKDIR /app

RUN pip install fastapi uvicorn pydantic

COPY scripts/06_api.py /app/

EXPOSE 8000

CMD ["python", "06_api.py"]