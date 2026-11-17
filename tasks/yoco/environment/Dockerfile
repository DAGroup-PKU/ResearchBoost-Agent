# linux/amd64 manifest of nvidia/cuda:12.1.1-devel-ubuntu22.04 (Docker Hub)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl build-essential ninja-build ca-certificates util-linux procps \
    && rm -rf /var/lib/apt/lists/*
# The training environment (Python 3.10 from Ubuntu): root-owned and read-only for the agent. The judge runs with
# it, and so does every reference run.
RUN python3 -m venv /opt/yoco-env
# Optional package indexes for hosts where pypi.org or download.pytorch.org is slow or unreachable (mirrors).
ARG PIP_INDEX_URL
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# Pinned: CUDA 12.1 (the base image; apex is built against its toolkit), torch 2.3.1+cu121 / torchvision 0.18.1 and
# numpy 1.23.0, the stack the judge was formulated on (constraints.txt, the constraints file of every install).
# pip 24.0: fairseq pins omegaconf<2.1, whose metadata newer pip versions reject.
COPY constraints.txt /opt/researchboost/constraints.txt
RUN /opt/yoco-env/bin/pip install pip==24.0 \
    && /opt/yoco-env/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -c /opt/researchboost/constraints.txt \
        torch torchvision \
    && /opt/yoco-env/bin/pip install -c /opt/researchboost/constraints.txt numpy cython wheel \
    && rm -rf /root/.cache/pip
# Optional mirror of github.com for hosts whose route to GitHub drops connections (git clones only; every commit
# below is pinned by hash, so the mirror cannot change what is built). Removed again after the last clone.
ARG GITHUB_MIRROR
RUN if [ -n "$GITHUB_MIRROR" ]; then git config --global url."$GITHUB_MIRROR/".insteadOf https://github.com/; fi
# Upstream's requirements.txt (microsoft/unilm YOCO @ f4f7b27a) at the commits the judge was formulated with: the
# fairseq fork it names (sunyt32/fairseq, branch zero_fs, at e872403f, with its Megatron submodule), infinibatch and
# torchscale from git, fairscale 0.4.0; tiktoken, sentencepiece, ninja, boto3 and iopath unpinned (they resolve at
# build time). Built without build isolation: fairseq and fairscale compile against the installed torch.
RUN /opt/yoco-env/bin/pip install -c /opt/researchboost/constraints.txt --no-build-isolation \
        fairscale==0.4.0 tiktoken sentencepiece ninja boto3 iopath \
        "git+https://github.com/sunyt32/fairseq.git@e872403f976bdeb4555fdf7e45fa3c582afe98df" \
        "git+https://github.com/shumingma/infinibatch.git@cf9a5cb9c7503298ce95f1fee66ad627c647246f" \
        "git+https://github.com/microsoft/torchscale.git@4d1e0e82e5adf86dd424f1463192635b73fc8efc" \
    && rm -rf /root/.cache/pip
# NVIDIA apex at 70018365 with its C++ and CUDA extensions for Hopper (sm_90), against the CUDA 12.1 toolkit of the
# base image: the fairseq fork's fused Adam and layer norm come from it. apex's setup.py reads --cpp_ext/--cuda_ext
# from its own command line (pip's --config-settings do not reach it), so the wheel is built with setup.py directly.
RUN git clone https://github.com/NVIDIA/apex /tmp/apex \
    && git -C /tmp/apex checkout 70018365c8add3e46574b897149db5d6dd21ef5c \
    && cd /tmp/apex && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=48 /opt/yoco-env/bin/python setup.py -q bdist_wheel \
        --cpp_ext --cuda_ext \
    && /opt/yoco-env/bin/pip install -c /opt/researchboost/constraints.txt --no-deps dist/apex-*.whl \
    && /opt/yoco-env/bin/python -c "import torch, amp_C, fused_layer_norm_cuda" \
    && cd / && rm -rf /tmp/apex /root/.cache/pip
# flash-attn 2.6.3: the official prebuilt wheel for torch 2.3, CUDA 12, C++11 ABI off (the torch wheels' ABI), cp310.
# FLASH_ATTN_WHEEL_URL: optional local copy of that same wheel for hosts whose GitHub downloads break (pip checks the
# wheel; the file name pins the build).
ARG FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN /opt/yoco-env/bin/pip install -c /opt/researchboost/constraints.txt --no-deps \
        "$FLASH_ATTN_WHEEL_URL" \
    && /opt/yoco-env/bin/pip install -c /opt/researchboost/constraints.txt einops \
    && rm -rf /root/.cache/pip
# The reference: microsoft/unilm at f4f7b27a, its YOCO/ directory as a repository of its own (one commit with fixed
# author and date, so the commit hash below is reproducible).
RUN git clone --filter=blob:none --no-checkout https://github.com/microsoft/unilm.git /tmp/unilm \
    && git -C /tmp/unilm sparse-checkout set YOCO \
    && git -C /tmp/unilm checkout f4f7b27aa08932ade248ab7a33aba7d444d3319e \
    && mkdir -p /opt/researchboost/ws/targets && cp -a /tmp/unilm/YOCO /opt/researchboost/ws/targets/reference \
    && rm -rf /tmp/unilm && cd /opt/researchboost/ws/targets/reference && git init -q && git add -A \
    && GIT_AUTHOR_NAME="microsoft/unilm" GIT_AUTHOR_EMAIL="unilm@users.noreply.github.com" \
       GIT_AUTHOR_DATE="2024-08-26T14:28:19+00:00" GIT_COMMITTER_NAME="microsoft/unilm" \
       GIT_COMMITTER_EMAIL="unilm@users.noreply.github.com" GIT_COMMITTER_DATE="2024-08-26T14:28:19+00:00" \
       git -c commit.gpgsign=false commit -q -m "microsoft/unilm@f4f7b27aa08932ade248ab7a33aba7d444d3319e YOCO/" \
    && test "$(git rev-parse HEAD)" = d1ed474155db05546d953c4d2aece86bd01c0e64
RUN git config --global --remove-section url."$GITHUB_MIRROR/" 2>/dev/null; test -z "$(git config --global --get-regexp '^url\.' || true)"
# The build fails if the training entrypoint's modules do not import or a pinned version moved.
RUN /opt/yoco-env/bin/pip check \
    && /opt/yoco-env/bin/python -c "import torch, torchvision, numpy; v = (torch.__version__, torchvision.__version__, \
numpy.__version__); assert v == ('2.3.1+cu121', '0.18.1+cu121', '1.23.0'), v" \
    && (cd /opt/researchboost/ws/targets/reference/yoco && PYTHONDONTWRITEBYTECODE=1 /opt/yoco-env/bin/python -c \
        "import models, tasks, criterions; from fairseq_cli.train import cli_main; import apex, flash_attn, amp_C, fused_layer_norm_cuda") \
    && /opt/yoco-env/bin/pip freeze > /opt/researchboost/installed-packages.txt
COPY workspace/ /opt/researchboost/ws/
COPY prep/ /opt/researchboost/prep/
COPY bootstrap.py entrypoint.sh no_credentials.py prep_pack.py /opt/researchboost/
# The candidate environment: owned by the agent, which may install anything into it (torch included). The judge
# runs the tree under test with its interpreter. A venv made from another venv sees the base interpreter's packages,
# not the training environment's, so a .pth file adds the training environment's site-packages after its own:
# packages the agent installs here take precedence, everything else comes from the training environment.
RUN useradd -m -s /bin/bash agent && mkdir -p /submission/yoco /opt/researchboost/assets \
    && cp -a /opt/researchboost/ws/targets/reference/. /submission/yoco/ \
    && git -C /submission/yoco checkout -q -b opt \
    && printf '{"environment": {}}\n' > /submission/switches.json \
    && /opt/yoco-env/bin/python -m venv /opt/candidate-env \
    && base=$(/opt/yoco-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && own=$(/opt/candidate-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && echo "import site; site.addsitedir('$base')" > "$own/researchboost-training-env.pth" \
    && /opt/candidate-env/bin/python -c 'import torch; assert torch.__file__.startswith("/opt/yoco-env/"), torch.__file__' \
    && chown -R agent:agent /submission /opt/candidate-env \
    && chmod +x /opt/researchboost/entrypoint.sh
# Compute capability major this image supports; preparation refuses other GPUs. The tokenizer cache
# ($ASSETS/tiktoken_cache) is set by the judge for every run it launches; it is not in the image's environment, whose
# variable names the startup credential check scans (a name containing TOKEN is refused).
ENV RESEARCHBOOST_GPU_ARCH=9 \
    ASSETS=/opt/researchboost/assets CUDA_HOME=/usr/local/cuda \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PATH=/opt/candidate-env/bin:/opt/yoco-env/bin:$PATH PYTHONNOUSERSITE=1
WORKDIR /submission/yoco
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
