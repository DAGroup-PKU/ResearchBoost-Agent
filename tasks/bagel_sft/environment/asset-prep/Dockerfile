# linux/amd64 manifest of python:3.11-slim-bookworm
FROM python:3.11-slim-bookworm@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199
ARG PIP_INDEX_URL
RUN pip install --no-cache-dir huggingface_hub==0.29.1
WORKDIR /prepare
COPY scripts/ scripts/
CMD ["python", "scripts/prepare_bagel_assets.py", "--assets-dir", "/cache"]
