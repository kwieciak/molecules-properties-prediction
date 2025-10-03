FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.3.0

#to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

#keep all base settings and do not remove ROCm/PyTorch dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

COPY . /app/

CMD ["python3", "main.py"]