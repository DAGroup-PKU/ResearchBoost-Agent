# linux/amd64 manifest of nvidia/cuda:12.1.1-devel-ubuntu22.04 (same digest on Docker Hub and nvcr.io)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl build-essential ca-certificates util-linux procps \
    && rm -rf /var/lib/apt/lists/*
# The training environment (Python 3.10 from Ubuntu, as upstream's README): root-owned and read-only for the agent.
# The judge runs with it, and so does every reference run.
RUN useradd -m -s /bin/bash agent && python3 -m venv /opt/llava-env
# Optional package indexes for hosts where pypi.org or download.pytorch.org is slow or unreachable (mirrors).
ARG PIP_INDEX_URL
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# Pinned: CUDA (the base image; nvcc 12.1 matches torch's CUDA for DeepSpeed and extension builds) and the
# torch 2.1.2+cu121 / torchvision 0.16.2+cu121 wheels the judge was formulated on. constraints.txt names only these
# two and every install below uses it, so nothing can replace them. `pip install --upgrade pip` and `wheel` are
# upstream's README step and the build tool its --no-build-isolation installs need.
COPY constraints.txt /opt/researchboost/constraints.txt
RUN /opt/llava-env/bin/pip install -c /opt/researchboost/constraints.txt --upgrade pip wheel \
    && /opt/llava-env/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -c /opt/researchboost/constraints.txt \
        torch torchvision \
    && rm -rf /root/.cache/pip
# Optional mirror of github.com for hosts whose route to GitHub drops connections (git clones only; every commit
# below is pinned by hash, so the mirror cannot change what is built). Removed again after the last clone.
ARG GITHUB_MIRROR
RUN if [ -n "$GITHUB_MIRROR" ]; then git config --global url."$GITHUB_MIRROR/".insteadOf https://github.com/; fi
# The reference is upstream at the pinned commit (scripts/zero3.json is tracked there).
RUN git init -q /opt/researchboost/ws/targets/reference \
    && git -C /opt/researchboost/ws/targets/reference fetch -q --depth 1 https://github.com/haotian-liu/LLaVA.git \
       c121f0432da27facab705978f83c4ada465e46fd \
    && git -C /opt/researchboost/ws/targets/reference checkout -q FETCH_HEAD \
    && git config --system --add safe.directory /opt/researchboost/ws/targets/reference
RUN git config --global --remove-section url."$GITHUB_MIRROR/" 2>/dev/null; test -z "$(git config --global --get-regexp '^url\.' || true)"
# The rest is upstream's own install (README): `pip install -e ".[train]"` (pyproject.toml's pins: transformers
# 4.37.2, tokenizers 0.15.1, accelerate 0.21.0, deepspeed 0.12.6, ...; the unpinned packages resolve at build time,
# so their versions may differ between builds), then `pip install flash-attn --no-build-isolation`. Differences:
#   - the editable install runs on a scratch copy and the llava package itself is uninstalled afterwards: the judge
#     imports llava from each tree under test, never from site-packages;
#   - deepspeed (upstream's pin 0.12.6) is installed first without build isolation, because its setup.py imports
#     torch;
#   - extra pin peft==0.10.0 (the version the judge was formulated with): pyproject.toml leaves peft unpinned, and
#     current peft imports accelerate.utils.memory.clear_device_cache, which upstream's accelerate 0.21.0 lacks, so
#     `from transformers import Trainer` fails and the reference cannot start;
#   - extra pin flash-attn==2.5.8: the README names no version, and current flash-attn has no prebuilt wheel for
#     torch 2.1 (it would compile from source, if at all). 2.5.8 is the version the judge was formulated with. Its
#     setup.py would download the matching release wheel (cu122, torch 2.1, pre-C++11 ABI) from GitHub in one
#     attempt; the build fetches that same wheel with retries and checks its sha256 instead.
# The build fails if the training entrypoint's modules do not import.
ARG FLASH_ATTN_WHEEL=flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# FLASH_ATTN_BASE_URL: where the wheel is fetched (a local copy where GitHub is slow); the sha256 below is checked anyway.
ARG FLASH_ATTN_BASE_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8
RUN curl -fsSL --retry 5 -o /tmp/$FLASH_ATTN_WHEEL \
        $FLASH_ATTN_BASE_URL/$FLASH_ATTN_WHEEL \
    && echo "1e782463ba32d2193924771c940805d74c365435d4142df18a261fc5a2fdff82  /tmp/$FLASH_ATTN_WHEEL" | sha256sum -c - \
    && /opt/llava-env/bin/pip install -c /opt/researchboost/constraints.txt --no-build-isolation deepspeed==0.12.6 \
    && cp -a /opt/researchboost/ws/targets/reference /tmp/llava-install \
    && (cd /tmp/llava-install \
        && /opt/llava-env/bin/pip install -c /opt/researchboost/constraints.txt -e ".[train]" peft==0.10.0) \
    && /opt/llava-env/bin/pip uninstall -y llava \
    && rm -rf /tmp/llava-install \
    && /opt/llava-env/bin/pip install -c /opt/researchboost/constraints.txt /tmp/$FLASH_ATTN_WHEEL \
    && rm /tmp/$FLASH_ATTN_WHEEL \
    && /opt/llava-env/bin/pip check \
    && /opt/llava-env/bin/python -c "import torch, torchvision, flash_attn, flash_attn_2_cuda, deepspeed, transformers; \
v = (torch.__version__, torchvision.__version__); assert v == ('2.1.2+cu121', '0.16.2+cu121'), v" \
    && (cd /opt/researchboost/ws/targets/reference \
        && PYTHONDONTWRITEBYTECODE=1 /opt/llava-env/bin/python -c "import llava.train.train") \
    && /opt/llava-env/bin/pip freeze > /opt/researchboost/installed-packages.txt \
    && rm -rf /root/.cache/pip
COPY workspace/ /opt/researchboost/ws/
COPY prep/ /opt/researchboost/prep/
COPY bootstrap.py entrypoint.sh no_credentials.py prep_pack.py /opt/researchboost/
# The candidate environment: owned by the agent, which may install anything into it (torch included). The judge
# runs the tree under test with its interpreter. A venv made from another venv sees the base interpreter's packages,
# not the training environment's, so a .pth file adds the training environment's site-packages after its own:
# packages the agent installs here take precedence, everything else comes from the training environment.
RUN mkdir -p /submission/llava /opt/researchboost/assets \
    && cp -a /opt/researchboost/ws/targets/reference/. /submission/llava/ \
    && git -C /submission/llava checkout -q -b opt \
    && printf '{"environment": {}}\n' > /submission/switches.json \
    && /opt/llava-env/bin/python -m venv /opt/candidate-env \
    && base=$(/opt/llava-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && own=$(/opt/candidate-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && echo "import site; site.addsitedir('$base')" > "$own/researchboost-training-env.pth" \
    && /opt/candidate-env/bin/python -c 'import torch; assert torch.__file__.startswith("/opt/llava-env/"), torch.__file__' \
    && chown -R agent:agent /submission /opt/candidate-env \
    && chmod +x /opt/researchboost/entrypoint.sh
# Compute capability major this image supports; preparation refuses other GPUs.
ENV RESEARCHBOOST_GPU_ARCH=9 \
    ASSETS=/opt/researchboost/assets \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PATH=/opt/candidate-env/bin:/opt/llava-env/bin:$PATH PYTHONNOUSERSITE=1
WORKDIR /submission/llava
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
