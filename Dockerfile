# syntax=docker/dockerfile:1.7
FROM ychangc/minedojo_pytorch:conda-cuda12.1-pytorch2.4.1 AS common

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

# Install Python dependencies (copy early to improve caching)
COPY --chown=user:user requirements.txt /home/user/requirements.txt
RUN pip install --exists-action=i -r /home/user/requirements.txt

# Copy project source
COPY --chown=user:user .. /workspace
WORKDIR /workspace

# Patch MineDojo config (change player mode to Adventure)
RUN sed -i -E 's/<AgentSection mode="\w+">/<AgentSection mode="Adventure">/' \
    /home/user/MineDojo/minedojo/sim/mc_meta/minedojo_mission.xml.j2


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