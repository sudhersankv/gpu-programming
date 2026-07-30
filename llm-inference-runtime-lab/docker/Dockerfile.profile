FROM llm-lab-vllm:phase2

RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg wget ca-certificates && \
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64 /" \
      > /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys \
      http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-cli && \
    rm -rf /var/lib/apt/lists/*