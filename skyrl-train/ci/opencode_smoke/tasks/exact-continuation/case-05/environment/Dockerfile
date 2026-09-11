FROM node:22-bookworm-slim

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bash ca-certificates coreutils curl git \
    && npm install --global opencode-ai@1.18.2 \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
