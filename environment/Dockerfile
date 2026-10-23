FROM --platform=linux/amd64 python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements-lock.txt wheels-sha256.txt /tmp/
COPY wheels/ /tmp/wheels/
RUN cd /tmp/wheels && sha256sum -c /tmp/wheels-sha256.txt && python -m pip install --no-index --no-deps --find-links /tmp/wheels -r /tmp/requirements-lock.txt && python -m pip check && rm -rf /tmp/wheels
# Build a separate dependency-only filesystem once. No tests, solutions or
# agent artifacts enter it. Hardlinks avoid copying large libraries per worker.
RUN mkdir -p /opt/candidate-root/etc /opt/candidate-root/tmp \
        /opt/candidate-root/dev /opt/candidate-root/app \
    && cp -al /usr /opt/candidate-root/usr \
    && cp -a /bin /lib /lib64 /opt/candidate-root/ \
    && cp /etc/ld.so.cache /opt/candidate-root/etc/ld.so.cache \
    && chmod 0755 /opt/candidate-root /opt/candidate-root/tmp
COPY src/ /app/src/
ENV PYTHONPATH=/app/src
