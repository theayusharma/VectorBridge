FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv sync
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0.tar.gz

COPY . .

ENV PORT=8080
ENV ENVIRONMENT="production"
EXPOSE $PORT

CMD ["uv", "run", "src/app.py"]
