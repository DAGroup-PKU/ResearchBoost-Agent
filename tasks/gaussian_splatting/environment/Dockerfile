# linux/amd64 manifest of nvidia/cuda:12.8.1-devel-ubuntu22.04 (same digest on Docker Hub and nvcr.io)
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl build-essential ninja-build libgl1 libglib2.0-0 \
    ca-certificates util-linux procps && rm -rf /var/lib/apt/lists/*
# The training environment (Python 3.10 from Ubuntu): root-owned and read-only for the agent. The judge runs with
# it, and so does every reference run.
RUN useradd -m -s /bin/bash agent && python3 -m venv /opt/gs-env
ARG PIP_INDEX_URL
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# Pinned: CUDA (the base image) and the torch 2.8.0+cu128 / torchvision 0.23.0+cu128 wheels, nothing else.
COPY constraints.txt /opt/researchboost/constraints.txt
RUN /opt/gs-env/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -c /opt/researchboost/constraints.txt \
        torch torchvision \
    && rm -rf /root/.cache/pip
# Upstream 3D Gaussian Splatting at the commit the judge was formulated on, with its three CUDA submodules at the
# commits that commit pins, except the rasterizer: upstream's accelerated rasterizer (README "Training speed
# acceleration": branch 3dgs_accel, the Taming-3DGS drop-in) replaces the default dr_aa build. The reference tree is
# one plain git repository: the submodules' sources are files of the tree (no nested repositories), so a change to
# any CUDA source is a change of the tree.
ARG GS_COMMIT=54c035f7834b564019656c3e3fcc3646292f727d
ARG RASTERIZER_COMMIT=26ce026ae9d3cfa56a103279b863a9f320c3e555
ARG SIMPLE_KNN_COMMIT=86710c2d4b46680c02301765dd79e465819c8f19
ARG FUSED_SSIM_COMMIT=1272e21a282342e89537159e4bad508b19b34157
ARG GLM_COMMIT=5c46b9c07008ae65cb81ab79cd677ecc1934b903
# Optional mirror of github.com for hosts whose route to GitHub drops connections (git clones only; every commit
# below is pinned by hash, so the mirror cannot change what is built). Removed again after the last clone.
ARG GITHUB_MIRROR
RUN if [ -n "$GITHUB_MIRROR" ]; then git config --global url."$GITHUB_MIRROR/".insteadOf https://github.com/; fi
RUN set -e; R=/opt/researchboost/ws/targets/reference; \
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git $R; git -C $R checkout -q $GS_COMMIT; \
    rm -rf $R/.git $R/.gitmodules $R/SIBR_viewers $R/submodules/*; \
    fetch() { git clone -q "$1" "$2" && git -C "$2" checkout -q "$3" && rm -rf "$2/.git"; }; \
    fetch https://github.com/graphdeco-inria/diff-gaussian-rasterization.git $R/submodules/diff-gaussian-rasterization $RASTERIZER_COMMIT; \
    rm -rf $R/submodules/diff-gaussian-rasterization/third_party/glm $R/submodules/diff-gaussian-rasterization/.gitmodules; \
    fetch https://github.com/g-truc/glm.git $R/submodules/diff-gaussian-rasterization/third_party/glm $GLM_COMMIT; \
    fetch https://gitlab.inria.fr/bkerbl/simple-knn.git $R/submodules/simple-knn $SIMPLE_KNN_COMMIT; \
    fetch https://github.com/rahul-goel/fused-ssim.git $R/submodules/fused-ssim $FUSED_SSIM_COMMIT; \
    git -C $R init -q -b main; git -C $R add -A; \
    GIT_AUTHOR_DATE=2024-10-30T12:39:22Z GIT_COMMITTER_DATE=2024-10-30T12:39:22Z \
    git -C $R -c user.name=researchboost -c user.email=researchboost@localhost commit -q \
        -m "reference: graphdeco-inria/gaussian-splatting $GS_COMMIT, rasterizer 3dgs_accel $RASTERIZER_COMMIT, simple-knn $SIMPLE_KNN_COMMIT, fused-ssim $FUSED_SSIM_COMMIT"
RUN git config --global --remove-section url."$GITHUB_MIRROR/" 2>/dev/null; test -z "$(git config --global --get-regexp '^url\.' || true)"
# Upstream's Python dependencies (environment.yml: plyfile, tqdm, opencv-python, joblib), unpinned except torch; its
# three CUDA extensions are NOT installed into the environment: they are built for Hopper (sm_90) from the reference
# tree into /opt/researchboost/ext/reference, the way the judge builds every tree's extensions (bench/bench.py
# build_extensions: pip install --no-deps --no-build-isolation --target), and runs put that directory first on
# sys.path. The build fails if the training script does not import.
RUN /opt/gs-env/bin/pip install -c /opt/researchboost/constraints.txt plyfile tqdm opencv-python-headless joblib \
        setuptools wheel \
    && R=/opt/researchboost/ws/targets/reference && E=/opt/researchboost/ext/reference && B=$(mktemp -d) \
    && for m in diff-gaussian-rasterization simple-knn fused-ssim; do \
        cp -a $R/submodules/$m $B/$m && (cd $B && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 CUDA_HOME=/usr/local/cuda \
            /opt/gs-env/bin/pip install --no-deps --no-build-isolation --no-cache-dir --target $E ./$m) || exit 1; \
        done \
    && rm -rf $B && touch $E/.complete \
    && /opt/gs-env/bin/pip check \
    && /opt/gs-env/bin/python -c "import torch, torchvision; v = (torch.__version__, torchvision.__version__); \
assert v == ('2.8.0+cu128', '0.23.0+cu128'), v" \
    && (cd $R && PYTHONPATH=$E PYTHONDONTWRITEBYTECODE=1 /opt/gs-env/bin/python -c \
        "import train, lpipsPyTorch; from diff_gaussian_rasterization import SparseGaussianAdam; import simple_knn._C, fused_ssim") \
    && test -z "$(git -C $R status --porcelain)" \
    && /opt/gs-env/bin/pip freeze > /opt/researchboost/installed-packages.txt \
    && rm -rf /root/.cache/pip
COPY workspace/ /opt/researchboost/ws/
COPY bootstrap.py entrypoint.sh no_credentials.py /opt/researchboost/
# The tree the agent works on: a copy of the reference (CUDA submodules included as plain files) on branch opt. The
# candidate environment: owned by the agent, which may install anything into it (torch included). The judge runs the
# tree under test and builds its extensions with its interpreter. A venv made from another venv sees the base
# interpreter's packages, not the training environment's, so a .pth file adds the training environment's
# site-packages after its own: packages the agent installs here take precedence, everything else comes from the
# training environment.
RUN mkdir -p /submission/gaussian-splatting /opt/researchboost/assets \
    && cp -a /opt/researchboost/ws/targets/reference/. /submission/gaussian-splatting/ \
    && git -C /submission/gaussian-splatting checkout -q -b opt \
    && printf '{"environment": {}}\n' > /submission/switches.json \
    && /opt/gs-env/bin/python -m venv /opt/candidate-env \
    && base=$(/opt/gs-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && own=$(/opt/candidate-env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && echo "import site; site.addsitedir('$base')" > "$own/researchboost-training-env.pth" \
    && /opt/candidate-env/bin/python -c 'import torch; assert torch.__file__.startswith("/opt/gs-env/"), torch.__file__' \
    && chown -R agent:agent /submission /opt/candidate-env \
    && chmod +x /opt/researchboost/entrypoint.sh
# Compute capability major this image supports; preparation refuses other GPUs.
ENV RESEARCHBOOST_GPU_ARCH=9 TORCH_HOME=/opt/researchboost/assets/torch CUDA_HOME=/usr/local/cuda \
    PATH=/opt/candidate-env/bin:/opt/gs-env/bin:$PATH PYTHONNOUSERSITE=1
WORKDIR /submission/gaussian-splatting
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
CMD ["sleep", "infinity"]
