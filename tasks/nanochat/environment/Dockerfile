# linux/amd64 manifest of nvidia/cuda:12.8.1-devel-ubuntu22.04 (Docker Hub): the CUDA 12.8 toolkit for Inductor/Triton
# and for any extension the agent builds.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl build-essential ninja-build ca-certificates util-linux procps \
    && rm -rf /var/lib/apt/lists/*
ARG PIP_INDEX_URL
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

# ---- The training environment (Python 3.10 from Ubuntu): root-owned and read-only for the agent. The judge runs with
# it, and so does every reference run. requirements.txt is upstream's uv.lock (karpathy/nanochat @ 92d63d4) exported
# for the gpu extra: torch 2.9.1+cu128, triton 3.5.1, kernels 0.11.7, numpy 1.26.4, pyarrow 21.0.0, rustbpe 0.1.0,
# tiktoken 0.11.0, wandb 0.21.3, ...
COPY requirements.txt /opt/researchboost/requirements.txt
RUN python3 -m venv /opt/nanochat-env \
    && /opt/nanochat-env/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -r /opt/researchboost/requirements.txt \
    && /opt/nanochat-env/bin/pip install ninja \
    && rm -rf /root/.cache/pip

# ---- FlashAttention-3, as upstream loads it: kernels.get_kernel("varunneal/flash-attention-3") (revision main) from the
# Hugging Face cache. The cache is pinned to commit de87b9b5 (its build for torch 2.9 / CUDA 12.8 / x86_64) and read
# offline by every run (HF_HOME=/opt/researchboost/hf, HF_HUB_OFFLINE=1). A download by commit hash writes no refs/,
# so refs/main is written here.
ENV HF_HOME=/opt/researchboost/hf
# Optional Hugging Face endpoint (a mirror) for hosts where huggingface.co is slow or unreachable; the download is pinned
# by commit hash either way.
ARG HF_ENDPOINT
RUN /opt/nanochat-env/bin/python -c "from kernels.utils import install_kernel, build_variant; \
print(build_variant(), install_kernel('varunneal/flash-attention-3', revision='de87b9b5af06dd9984df595bef90b2eba44b181a'))" \
    && mkdir -p /opt/researchboost/hf/hub/models--varunneal--flash-attention-3/refs \
    && printf de87b9b5af06dd9984df595bef90b2eba44b181a > /opt/researchboost/hf/hub/models--varunneal--flash-attention-3/refs/main \
    && test -d /opt/researchboost/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-x86_64-linux \
    && chmod -R a+rX /opt/researchboost/hf

# ---- The reference: karpathy/nanochat at 92d63d4.
RUN git clone -q https://github.com/karpathy/nanochat.git /tmp/nanochat \
    && git -C /tmp/nanochat checkout -q 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd \
    && mkdir -p /opt/researchboost/ws/targets && cp -a /tmp/nanochat /opt/researchboost/ws/targets/reference \
    && rm -rf /tmp/nanochat \
    && test "$(git -C /opt/researchboost/ws/targets/reference rev-parse HEAD)" = 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
RUN /opt/nanochat-env/bin/python -c "import torch, triton, kernels, numpy, rustbpe, tiktoken, wandb, pyarrow; print(torch.__version__, triton.__version__)" \
    && /opt/nanochat-env/bin/pip freeze > /opt/researchboost/installed-packages.txt
COPY workspace/ /opt/researchboost/ws/
COPY bootstrap.py entrypoint.sh no_credentials.py /opt/researchboost/
# The candidate environment: owned by the agent, which may install anything into it (torch included). The judge runs
# the tree under test with its interpreter. A .pth file adds the training environment's site-packages after its own:
# packages the agent installs here take precedence, everything else comes from the training environment.
RUN useradd -m -s /bin/bash agent && mkdir -p /submission/nanochat /opt/researchboost/assets \
    && cp -a /opt/researchboost/ws/targets/reference/. /submission/nanochat/ \
    && git -C /submission/nanochat checkout -q -b opt \
    && printf '{"environment": {}}\n' > /submission/switches.json \
    && /opt/nanochat-env/bin/python -m venv /opt/candidate-env \
    && base=$(/opt/nanochat-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && own=$(/opt/candidate-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && echo "import site; site.addsitedir('$base')" > "$own/researchboost-training-env.pth" \
    && /opt/candidate-env/bin/python -c 'import torch; assert torch.__file__.startswith("/opt/nanochat-env/"), torch.__file__' \
    && chown -R agent:agent /submission /opt/candidate-env \
    && chmod +x /opt/researchboost/entrypoint.sh
ENV RESEARCHBOOST_GPU_ARCH=9 \
    ASSETS=/opt/researchboost/assets CUDA_HOME=/usr/local/cuda \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PATH=/opt/candidate-env/bin:/opt/nanochat-env/bin:$PATH PYTHONNOUSERSITE=1
WORKDIR /submission/nanochat
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
