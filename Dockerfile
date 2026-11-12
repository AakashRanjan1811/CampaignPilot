FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000
# Default: dashboard. For the API run:  docker run -p 8000:8000 campaignpilot api
ENTRYPOINT ["sh", "-c", "if [ \"$1\" = api ]; then exec uvicorn api.main:app --host 0.0.0.0 --port 8000; else exec streamlit run app.py --server.address 0.0.0.0 --server.port 8501; fi", "--"]
