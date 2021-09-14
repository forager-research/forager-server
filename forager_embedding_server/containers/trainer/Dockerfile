FROM nvidia/cuda:10.0-runtime-ubuntu18.04

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies.
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.8 python3.8-distutils build-essential python3.8-dev wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential python3.8-dev wget

# Run script on container startup.
COPY . ./
STOPSIGNAL SIGTERM
ENV FLASK_APP=main.py
CMD exec flask run --host=0.0.0.0 --port=$PORT
