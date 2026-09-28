FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/dronevision-follow
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /opt/dronevision-follow/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /opt/dronevision-follow/backend
COPY frontend /opt/dronevision-follow/frontend
COPY config /opt/dronevision-follow/config
RUN mkdir -p /opt/dronevision-follow/sessions
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
