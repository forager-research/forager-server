FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies.
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3.7-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential python3.7-dev

# Run the web service on container startup.
COPY . ./
CMD exec uvicorn --host "0.0.0.0" --port $PORT --workers $NPROC --log-level warning handler:app
