# ── Imagem base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadados ─────────────────────────────────────────────────────────────────
LABEL maintainer="Veritas Team — CESAR School"
LABEL description="Veritas: detecção de imagens reais vs geradas por IA"

# ── Variáveis de ambiente ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Dependências do sistema ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Diretório de trabalho ─────────────────────────────────────────────────────
WORKDIR /app

# ── Copia requirements e instala dependências Python ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copia o código e os modelos ───────────────────────────────────────────────
COPY v3/          ./v3/
COPY models/      ./models/
COPY pipeline.py  .

# ── Cria pasta para os resultados do MLflow ───────────────────────────────────
RUN mkdir -p mlruns

# ── Porta do MLflow UI ────────────────────────────────────────────────────────
EXPOSE 5000

# ── Comando padrão ────────────────────────────────────────────────────────────
CMD ["sh", "-c", "python pipeline.py && mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns"]
