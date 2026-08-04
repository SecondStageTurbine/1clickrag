# SPDX-License-Identifier: MPL-2.0
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY app /srv/app

EXPOSE 8404

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=40 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8404/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "app.server"]
