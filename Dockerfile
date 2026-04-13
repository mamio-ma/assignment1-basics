FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

CMD ["uv", "run", "pytest", "tests/test_tokenizer.py", "-v"]

## build docker: docker build -t assignment1 .
## run test
## docker run assignment1 uv run pytest tests/test_tokenizer.py::test_encode_memory_usage -v
