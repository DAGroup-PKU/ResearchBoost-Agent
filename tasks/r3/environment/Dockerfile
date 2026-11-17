# linux/amd64 manifest of nvidia/cuda:12.8.1-devel-ubuntu22.04 (Docker Hub): the CUDA 12.8 toolkit builds flash-attn
# for the training environment and mmcv-full's ops for the reward service.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl build-essential ninja-build ca-certificates util-linux procps \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ARG PIP_INDEX_URL
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG REWARD_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# ---- The training environment (Python 3.10 from Ubuntu): root-owned and read-only for the agent. The judge runs with
# it, and so does every reference run. Upstream R3 (sen-ye/R3 @ 761b0b0) requirements.txt pins torch 2.9.1 /
# torchvision 0.24.1 and flash_attn 2.5.8; opencv-python is replaced by the headless build of the same version (no
# libGL in the image).
COPY constraints.txt /opt/researchboost/constraints.txt
RUN python3 -m venv /opt/r3-env \
    && /opt/r3-env/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -c /opt/researchboost/constraints.txt \
        torch torchvision \
    && /opt/r3-env/bin/pip install -c /opt/researchboost/constraints.txt packaging wheel ninja psutil pytz \
    && rm -rf /root/.cache/pip
# flash-attn 2.5.8 has no prebuilt wheel for torch 2.9: built from source for Hopper (sm_90) against CUDA 12.8.
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=96 \
        /opt/r3-env/bin/pip install -c /opt/researchboost/constraints.txt --no-build-isolation --no-deps flash_attn==2.5.8 \
    && /opt/r3-env/bin/python -c "import flash_attn, flash_attn_2_cuda; print(flash_attn.__version__)" \
    && rm -rf /root/.cache/pip
RUN /opt/r3-env/bin/pip install -c /opt/researchboost/constraints.txt \
        decord einops huggingface_hub matplotlib numpy opencv-python-headless pyarrow PyYAML Requests safetensors \
        scipy sentencepiece transformers accelerate wandb openai \
    && rm -rf /root/.cache/pip

# ---- The GenEval reward service (yifan123/reward-server @ 8245af2): its own environment, as upstream instructs
# (Python 3.10, torch 2.1.2+cu121, mmcv-full 1.7.2 with ops for sm_90, mmdetection 2.x, open_clip 2.31.0). It is part
# of the judge: the same service, configuration and weights for the reference and every candidate.
RUN python3 -m venv /opt/reward-env \
    && /opt/reward-env/bin/pip install --extra-index-url "$REWARD_TORCH_INDEX_URL" \
        torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    && /opt/reward-env/bin/pip install gunicorn==23.0.0 openmim==0.3.9 open-clip-torch==2.31.0 numpy==1.26.0 \
        opencv-python-headless==4.11.0.86 clip-benchmark==1.6.1 flask==3.1.0 wheel ninja \
    && rm -rf /root/.cache/pip
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=96 \
        /opt/reward-env/bin/pip install --no-build-isolation mmcv-full==1.7.2 \
    && /opt/reward-env/bin/python -c "import mmcv, mmcv.ops; print(mmcv.__version__)" \
    && rm -rf /root/.cache/pip
# Optional mirror of github.com for hosts whose route to GitHub drops connections (git clones only; every commit
# below is pinned by hash, so the mirror cannot change what is built). Removed again after the last clone.
ARG GITHUB_MIRROR
RUN if [ -n "$GITHUB_MIRROR" ]; then git config --global url."$GITHUB_MIRROR/".insteadOf https://github.com/; fi
RUN git clone -q --branch v2.28.2 --depth 1 https://github.com/open-mmlab/mmdetection.git /opt/mmdetection \
    && sed -i "s/^mmcv_maximum_version = .*/mmcv_maximum_version = '2.3.0'/" /opt/mmdetection/mmdet/__init__.py \
    && /opt/reward-env/bin/pip install --no-build-isolation -e /opt/mmdetection \
    && git clone -q https://github.com/yifan123/reward-server.git /opt/reward-server \
    && git -C /opt/reward-server checkout -q 8245af22b75fe640950a880ad31fe702a5d9a014 \
    && /opt/reward-env/bin/python -c "import mmdet, open_clip, flask, gunicorn" \
    && rm -rf /root/.cache/pip

# ---- The reference: sen-ye/R3 at 761b0b0.
RUN git clone -q https://github.com/sen-ye/R3.git /tmp/r3 && git -C /tmp/r3 checkout -q 761b0b0713352a52128cbd39a004dc383bd082bd \
    && mkdir -p /opt/researchboost/ws/targets && cp -a /tmp/r3 /opt/researchboost/ws/targets/reference \
    && rm -rf /tmp/r3 \
    && test "$(git -C /opt/researchboost/ws/targets/reference rev-parse HEAD)" = 761b0b0713352a52128cbd39a004dc383bd082bd
RUN git config --global --remove-section url."$GITHUB_MIRROR/" 2>/dev/null; test -z "$(git config --global --get-regexp '^url\.' || true)"
RUN /opt/r3-env/bin/python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)" \
    && (cd /opt/researchboost/ws/targets/reference && PYTHONDONTWRITEBYTECODE=1 /opt/r3-env/bin/python -c \
        "import train.online_rl, flash_attn") \
    && /opt/r3-env/bin/pip freeze > /opt/researchboost/installed-packages.txt \
    && /opt/reward-env/bin/pip freeze > /opt/researchboost/installed-packages-reward.txt
COPY workspace/ /opt/researchboost/ws/
# The judge's frozen copies of the data every run reads: upstream's GenEval training prompts and the evaluation fixture
# (the first eight GenEval test prompts, one per rank), taken from the reference checkout.
RUN mkdir -p /opt/researchboost/ws/bench/data /opt/researchboost/ws/bench/records \
    && cp /opt/researchboost/ws/targets/reference/data/rl_train/train_geneval.jsonl /opt/researchboost/ws/bench/data/ \
    && head -n 8 /opt/researchboost/ws/targets/reference/data/rl_test/test_geneval.jsonl \
        > /opt/researchboost/ws/bench/data/test_geneval_first8.jsonl \
    && test "$(wc -l < /opt/researchboost/ws/bench/data/test_geneval_first8.jsonl)" = 8
COPY bootstrap.py entrypoint.sh no_credentials.py /opt/researchboost/
# The candidate environment: owned by the agent, which may install anything into it (torch included). The judge runs
# the tree under test with its interpreter. A .pth file adds the training environment's site-packages after its own:
# packages the agent installs here take precedence, everything else comes from the training environment.
RUN useradd -m -s /bin/bash agent && mkdir -p /submission/r3 /opt/researchboost/assets \
    && cp -a /opt/researchboost/ws/targets/reference/. /submission/r3/ \
    && git -C /submission/r3 checkout -q -b opt \
    && printf '{"environment": {}}\n' > /submission/switches.json \
    && /opt/r3-env/bin/python -m venv /opt/candidate-env \
    && base=$(/opt/r3-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && own=$(/opt/candidate-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && echo "import site; site.addsitedir('$base')" > "$own/researchboost-training-env.pth" \
    && /opt/candidate-env/bin/python -c 'import torch; assert torch.__file__.startswith("/opt/r3-env/"), torch.__file__' \
    && chown -R agent:agent /submission /opt/candidate-env \
    && chmod +x /opt/researchboost/entrypoint.sh
ENV RESEARCHBOOST_GPU_ARCH=9 \
    ASSETS=/opt/researchboost/assets CUDA_HOME=/usr/local/cuda \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PATH=/opt/candidate-env/bin:/opt/r3-env/bin:$PATH PYTHONNOUSERSITE=1
WORKDIR /submission/r3
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
