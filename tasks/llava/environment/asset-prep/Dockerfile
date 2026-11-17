# linux/amd64 manifest of python:3.11-slim-bookworm
FROM python:3.11-slim-bookworm@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199
WORKDIR /prepare
COPY scripts/ scripts/
COPY assets/ assets/
CMD ["python", "scripts/prepare_assets.py", "--assets-dir", "/cache"]
