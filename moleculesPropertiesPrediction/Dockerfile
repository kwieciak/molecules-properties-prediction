FROM rocm/pytorch:latest

#to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

#kep all base settings and do not remove ROCm/PyTorch dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

COPY . /app/

CMD ["python3", "main.py"]