# Generic container: serves the API by default; override CMD to serve the Streamlit app.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 LOG_PREDICTIONS=0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY churnlib ./churnlib
COPY app ./app
COPY models ./models
COPY data/processed/customers_clean.csv data/processed/test.csv ./data/processed/
COPY reports ./reports
COPY docs ./docs
COPY streamlit_app.py .
COPY .streamlit ./.streamlit
RUN mkdir -p logs
EXPOSE 8000
# API:        docker run -p 8000:8000 churn
# Dashboard:  docker run -p 8501:8501 churn streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
