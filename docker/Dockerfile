FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m piptools sync requirements.txt

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user data_utils.py /opt/app/
COPY --chown=user:user model-a.pt /opt/app/
COPY --chown=user:user model-b.pt /opt/app/
COPY --chown=user:user model-c.pt /opt/app/
COPY --chown=user:user model-d.pt /opt/app/
COPY --chown=user:user model-e.pt /opt/app/
COPY --chown=user:user model-f.pt /opt/app/
COPY --chown=user:user model-g.pt /opt/app/
COPY --chown=user:user model-h.pt /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]