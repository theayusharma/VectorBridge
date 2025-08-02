FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .

RUN uv sync

COPY . .

ENV PORT=8080
EXPOSE $PORT

CMD ["uv", "src/main.py", "--host", "0.0.0.0", "--port", "${PORT}", "--proxy-headers"]
