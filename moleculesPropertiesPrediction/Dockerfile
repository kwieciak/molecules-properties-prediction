FROM rocm/pytorch:latest

# Zachowaj wszystkie ustawienia bazowe i nie usuwaj zależności ROCm/PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalacja dodatkowych bibliotek Python
RUN pip install --no-cache-dir numpy pandas scikit-learn

# Instalacja torch_geometric i zależności
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric

# Workdir i komenda startowa
WORKDIR /app

# Skopiowanie kodu projektu do obrazu
COPY . /app

CMD ["python3", "main.py"]
