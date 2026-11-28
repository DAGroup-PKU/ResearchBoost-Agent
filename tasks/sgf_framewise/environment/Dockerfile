FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates git libgl1 libglib2.0-0 python3 python3-venv util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash agent \
    && useradd --system --no-create-home --shell /usr/sbin/nologin judge-worker

COPY requirements.txt bootstrap.py no_credentials.py entrypoint.sh /opt/researchboost/
COPY workspace/ /opt/researchboost/ws/

RUN python3 -m venv /opt/reference-env \
    && python3 -m venv /opt/candidate-env \
    && /opt/candidate-env/bin/python -m pip install --no-cache-dir -r /opt/researchboost/requirements.txt \
    && mkdir -p /submission /opt/researchboost/ws/targets/reference \
    && cd /opt/researchboost/ws/targets/reference \
    && commit=$(python3 -c 'import json; print(json.load(open("../../bench/bench.json"))["reference_commit"])') \
    && git init -q \
    && git remote add origin https://github.com/zhuang2002/Self_Gradient_Forcing.git \
    && git fetch --depth=1 origin "$commit" \
    && git checkout --detach "$commit" \
    && cp -a . /submission/sgf_framewise \
    && printf '{}\n' > /submission/switches.json \
    && chown -R agent:agent /submission/sgf_framewise /submission/switches.json /opt/candidate-env \
    && chmod 755 /opt/researchboost/entrypoint.sh

ENV PATH="/opt/candidate-env/bin:${PATH}"
WORKDIR /submission/sgf_framewise
ENTRYPOINT ["/opt/researchboost/entrypoint.sh"]
CMD ["/usr/bin/sleep", "infinity"]
