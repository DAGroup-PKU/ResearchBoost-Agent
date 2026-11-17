# slime's official image of 2026-10-27 (slimerl/slime:latest then, linux/amd64), pinned by digest: sglang v0.5.21 with
# slime's v0.5.21 patches (/sgl-workspace/sglang), Megatron-LM 4b4acac9 with slime's patches (/root/Megatron-LM),
# torch 2.13 + CUDA 13 (cuda-compat-13-0 for older data-center drivers), Transformer Engine 2.18, apex,
# flash-linear-attention 0.5.2, Ray; the Python environment is /opt/sglang.
# BASE_IMAGE: the same image under another name where the registry is not reachable (a host-level build argument,
# e.g. a local copy loaded from the registry blobs); the next step checks it is that build.
ARG BASE_IMAGE=slimerl/slime@sha256:227723fe627d0dbc6f05d1c5ca14c28dc21d069150cdc38b27db7c8db7ec6b38
FROM ${BASE_IMAGE}
RUN test "$SGLANG_BUILD_COMMIT" = e00930c5489053f26d86b179cee0d087f846acbb \
    && test "$(git -C /root/Megatron-LM rev-parse HEAD)" = 4b4acac9a1d28ea6829c8d4f566d75698a21249d \
    && echo "c8c43285f580e6e93af6c665082b2088c53d79a28f85bcc61611cc11744cc1f3  /opt/slime/constraints.txt" | sha256sum -c -
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
# GitHub mirror for hosts where github.com is slow or unreachable; the clone is pinned by commit either way.
ARG GITHUB_MIRROR=https://github.com

# ---- The reference: THUDM/slime at 540ea919 (the commit the image was built from), and the image's patched sglang
# and Megatron-LM sources snapshotted as git repositories (they are the reference's rollout and training stacks).
RUN git clone -q "$GITHUB_MIRROR/THUDM/slime.git" /tmp/slime \
    && git -C /tmp/slime checkout -q 540ea919e45b8b929198abce008a103be9cfc779 \
    && test "$(git -C /tmp/slime rev-parse HEAD)" = 540ea919e45b8b929198abce008a103be9cfc779 \
    && mkdir -p /opt/researchboost/ws/targets && cp -a /tmp/slime /opt/researchboost/ws/targets/reference \
    && rm -rf /tmp/slime
RUN set -e; for pair in reference-sglang:/sgl-workspace/sglang reference-Megatron-LM:/root/Megatron-LM; do \
      name=${pair%%:*}; src=${pair#*:}; ref=/opt/researchboost/ws/targets/$name; \
      mkdir -p "$ref"; (cd "$src" && git ls-files -co --exclude-standard -z 2>/dev/null || find . -type f -print0) \
        | (cd "$src" && tar --null -T - -cf - --exclude=.git) | tar -xf - -C "$ref"; \
      find "$ref" \( -name __pycache__ -o -name '*.egg-info' \) -prune -exec rm -rf {} +; \
      git -C "$ref" init -q; git -C "$ref" add -A; \
      git -C "$ref" -c user.name=reference -c user.email=reference@localhost commit -qm "reference snapshot of $src"; \
    done
RUN LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH /opt/sglang/bin/python3 -c "import torch, sglang, megatron.core, transformer_engine, fla, ray; print(torch.__version__, torch.version.cuda, sglang.__version__, megatron.core.__version__, transformer_engine.__version__, fla.__version__, ray.__version__)" \
    && /opt/sglang/bin/python3 -m pip freeze > /opt/researchboost/installed-packages.txt
COPY workspace/ /opt/researchboost/ws/
COPY bootstrap.py entrypoint.sh no_credentials.py /opt/researchboost/
# The candidate trees (git, branch opt) and the candidate environment: owned by the agent, which may install anything
# into it. The judge runs the trees under test with its interpreter. A .pth file adds the training environment's
# site-packages after its own: packages the agent installs here take precedence, everything else comes from the
# training environment. /root is made traversable, not listable: the image's editable installs of slime and
# Megatron-LM point into it, and the agent user's processes import through them for a tree not on PYTHONPATH.
RUN useradd -m -s /bin/bash agent && mkdir -p /submission/slime_grpo /opt/researchboost/assets /opt/researchboost/prep \
    && for pair in slime:reference sglang:reference-sglang Megatron-LM:reference-Megatron-LM; do \
         git clone -q --no-local "/opt/researchboost/ws/targets/${pair#*:}" "/submission/slime_grpo/${pair%%:*}" \
         && git -C "/submission/slime_grpo/${pair%%:*}" checkout -q -b opt; done \
    && printf '{"environment": {}}\n' > /submission/switches.json \
    && /opt/sglang/bin/python3 -m venv /opt/candidate-env \
    && base=$(/opt/sglang/bin/python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && own=$(/opt/candidate-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && echo "import site; site.addsitedir('$base')" > "$own/researchboost-training-env.pth" \
    && /opt/candidate-env/bin/python -c 'import torch; assert torch.__file__.startswith("/opt/sglang/"), torch.__file__' \
    && chown -R agent:agent /submission /opt/candidate-env \
    && chmod 711 /root \
    && chmod +x /opt/researchboost/entrypoint.sh /opt/researchboost/ws/bench/launch.sh \
    && chmod -R a+rX /opt/researchboost
ENV ASSETS=/opt/researchboost/assets CUDA_HOME=/usr/local/cuda PIP_CONSTRAINT= \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PATH=/opt/candidate-env/bin:/opt/sglang/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONNOUSERSITE=1
WORKDIR /submission/slime_grpo
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
