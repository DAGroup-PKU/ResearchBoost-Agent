# BAGEL-7B-MoT supervised fine-tuning, upstream ByteDance-Seed/Bagel @ a2fa77d, installed as its README says:
# Python 3.10, `pip install -r requirements.txt`, `pip install flash_attn==2.5.8 --no-build-isolation`.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl build-essential ninja-build ca-certificates util-linux procps \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ARG PIP_INDEX_URL
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG BUILD_JOBS=32
COPY constraints.txt /opt/researchboost/constraints.txt
RUN python3 -m venv /opt/bagel-env \
    && /opt/bagel-env/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -c /opt/researchboost/constraints.txt \
        torch torchvision \
    && /opt/bagel-env/bin/pip install -c /opt/researchboost/constraints.txt packaging wheel ninja psutil \
    && rm -rf /root/.cache/pip
# flash-attn 2.5.8 has no prebuilt wheel for torch 2.5: built from source, for Hopper (sm_90) only.
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=$BUILD_JOBS \
        /opt/bagel-env/bin/pip install -c /opt/researchboost/constraints.txt --no-build-isolation --no-deps \
        flash_attn==2.5.8 \
    && /opt/bagel-env/bin/python -c "import flash_attn, flash_attn_2_cuda; print(flash_attn.__version__)" \
    && rm -rf /root/.cache/pip
# Optional mirror of github.com for hosts whose route to GitHub drops connections (git clones only; every commit
# below is pinned by hash, so the mirror cannot change what is built). Removed again after the last clone.
ARG GITHUB_MIRROR
RUN if [ -n "$GITHUB_MIRROR" ]; then git config --global url."$GITHUB_MIRROR/".insteadOf https://github.com/; fi
RUN git clone -q https://github.com/ByteDance-Seed/Bagel.git /tmp/bagel \
    && git -C /tmp/bagel checkout -q a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f \
    && mkdir -p /opt/researchboost/ws/targets && cp -a /tmp/bagel /opt/researchboost/ws/targets/reference \
    && rm -rf /tmp/bagel \
    && test "$(git -C /opt/researchboost/ws/targets/reference rev-parse HEAD)" = a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f
RUN git config --global --remove-section url."$GITHUB_MIRROR/" 2>/dev/null; test -z "$(git config --global --get-regexp '^url\.' || true)"
# Upstream's requirements.txt (torch and flash_attn are already installed; opencv_python -> headless, as above).
RUN sed -e 's/^opencv_python==/opencv-python-headless==/' \
        /opt/researchboost/ws/targets/reference/requirements.txt > /tmp/requirements.txt \
    && /opt/bagel-env/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -c /opt/researchboost/constraints.txt \
        -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /tmp/requirements.txt
RUN /opt/bagel-env/bin/python -c "import torch, torchvision, numpy, flash_attn, decord; print(torch.__version__, torchvision.__version__, numpy.__version__, flash_attn.__version__)" \
    && (cd /opt/researchboost/ws/targets/reference && PYTHONDONTWRITEBYTECODE=1 /opt/bagel-env/bin/python -c \
        "import data.dataset_base, modeling.bagel, train.fsdp_utils") \
    && /opt/bagel-env/bin/pip check \
    && /opt/bagel-env/bin/pip freeze > /opt/researchboost/installed-packages.txt
COPY workspace/ /opt/researchboost/ws/
# The judge's frozen copy of the data configuration every run reads: upstream's data/configs/example.yaml.
RUN mkdir -p /opt/researchboost/ws/bench/records \
    && cp /opt/researchboost/ws/targets/reference/data/configs/example.yaml /opt/researchboost/ws/bench/data_config.yaml
COPY bootstrap.py entrypoint.sh no_credentials.py /opt/researchboost/
# The candidate environment: owned by the agent, which may install anything into it (torch included). The judge runs
# the tree under test with its interpreter. A .pth file adds the training environment's site-packages after its own:
# packages the agent installs here take precedence, everything else comes from the training environment.
RUN useradd -m -s /bin/bash agent && mkdir -p /submission/bagel_sft /opt/researchboost/assets \
    && cp -a /opt/researchboost/ws/targets/reference/. /submission/bagel_sft/ \
    && git -C /submission/bagel_sft checkout -q -b opt \
    && printf '{"environment": {}}\n' > /submission/switches.json \
    && /opt/bagel-env/bin/python -m venv /opt/candidate-env \
    && base=$(/opt/bagel-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && own=$(/opt/candidate-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && echo "import site; site.addsitedir('$base')" > "$own/researchboost-training-env.pth" \
    && /opt/candidate-env/bin/python -c 'import torch; assert torch.__file__.startswith("/opt/bagel-env/"), torch.__file__' \
    && chown -R agent:agent /submission /opt/candidate-env \
    && chmod +x /opt/researchboost/entrypoint.sh
ENV RESEARCHBOOST_GPU_ARCH=9 \
    ASSETS=/opt/researchboost/assets CUDA_HOME=/usr/local/cuda \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline \
    PATH=/opt/candidate-env/bin:/opt/bagel-env/bin:$PATH PYTHONNOUSERSITE=1
WORKDIR /submission/bagel_sft
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
