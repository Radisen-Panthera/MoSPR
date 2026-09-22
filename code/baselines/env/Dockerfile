ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG WITH_MAMBA=0
ARG WITH_CELL2LOCATION=0
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONHASHSEED=2021

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git build-essential ninja-build \
        openslide-tools libgl1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 lightning==2.6.5 timm==1.0.28 einops==0.8.2 h5py==3.16.0 \
        numpy==2.1.3 pandas==2.3.3 scipy==1.17.1 scikit-learn==1.9.0 pyyaml==6.0.3 \
        tqdm openslide-python

RUN if [ "$WITH_MAMBA" = "1" ]; then \
        pip install --no-cache-dir causal-conv1d==1.1.1 && \
        pip install --no-cache-dir "mamba-ssm @ git+https://github.com/isyangshu/MambaMIL.git" ; \
    fi

RUN if [ "$WITH_CELL2LOCATION" = "1" ]; then \
        pip install --no-cache-dir cell2location==0.1.5 scvi-tools==1.4.2 \
            anndata==0.12.19 scanpy==1.11.5 ; \
    fi

WORKDIR /workspace
ENV MOSPR_ROOT=/workspace MOSPR_DATASET_ROOT=/data MOSPR_CPNN_REPO=/workspace/external/CPNN
CMD ["bash"]
