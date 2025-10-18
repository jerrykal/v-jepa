# syntax=docker/dockerfile:1.7
FROM jerrykal/minedojo-base:latest AS common

USER root

# Configure apt in non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Create mount path
RUN mkdir -p /mount/nfs
ENV MOUNT_PATH=/mount/nfs

# Install extra toolkits
RUN apt-get update \
 && apt-get install -y --no-install-recommends jq curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

WORKDIR /workspace

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project

# Place executables in the environment at the front of the path
ENV PATH="/workspace/.venv/bin:$PATH"

# Copy project source
COPY --chown=user:user . .

# Patch MineDojo config (change player mode to Adventure)
# TEMP: This patch is temporary and is subject to change.
RUN sed -i -E 's/<AgentSection mode="\w+">/<AgentSection mode="Adventure">/' \
    /workspace/.venv/lib/python3.10/site-packages/minedojo/sim/mc_meta/minedojo_mission.xml.j2

# -------- base target --------
FROM common AS base
CMD ["bash"]

# -------- ovx target (base + one extra file) --------
FROM common AS ovx

# Option A: copy from local context
# COPY --chown=user:user run.sh /workspace/run.sh
# RUN chmod +x /workspace/run.sh

# Option B: download from GitHub raw
ARG RUN_FILE_URL
ARG RUN_FILE_PATH=/run.sh
RUN if [ -n "$RUN_FILE_URL" ]; then \
      curl -L "$RUN_FILE_URL" -o "$RUN_FILE_PATH" && chmod +x "$RUN_FILE_PATH"; \
    else \
      echo "RUN_FILE_URL not provided, skipping extra file"; \
    fi