FROM python:3.12-slim
RUN useradd -m agent && mkdir -p /opt/researchboost/ws/targets /submission /var/log/researchboost /opt/candidate-env/bin
COPY bootstrap.py no_credentials.py entrypoint.sh /opt/researchboost/
COPY workspace/ /opt/researchboost/ws/
COPY reference/ /opt/researchboost/ws/targets/reference/
COPY reference/ /submission/toytask/
RUN echo '{}' > /submission/switches.json && chown -R agent:agent /submission /opt/candidate-env \
    && chmod -R a+rX /opt/researchboost
# as in real tasks: the agent's own environment first on PATH (the judge must not inherit it; protocol supplement C10)
ENV PATH=/opt/candidate-env/bin:$PATH
ENTRYPOINT ["/bin/bash", "/opt/researchboost/entrypoint.sh"]
CMD ["sleep", "infinity"]
