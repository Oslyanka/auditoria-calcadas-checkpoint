"""
Baixa as fontes brutas do dataset do CPZAO (e e idempotente, ou seja: se ja baixou
antes, nao baixa de novo).

Sao duas fontes publicas:
  1) dataset de buracos (pothole) do Hugging Face -> vem com labels YOLO prontos;
  2) metadados do dataset de obstaculos (ROD), tambem do Hugging Face -> so os CSVs
     por enquanto; as imagens em si o build_dataset.py baixa sob demanda.

Tudo cai em _scratch/raw/ (que fica fora do git, e so cache local).
"""
import os, sys, json, urllib.request, csv, io

# raiz do projeto e pasta de cache das fontes brutas
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "_scratch", "raw")
os.makedirs(RAW, exist_ok=True)  # garante que a pasta de cache existe


def get(url, timeout=180):
    """Baixa o conteudo de uma URL e devolve os bytes crus."""
    # manda um User-Agent de navegador porque alguns servidores bloqueiam requisicao "sem cara"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 cpzao-dataset"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(url, dest, timeout=180):
    """Baixa a URL pro arquivo dest, mas pula se o arquivo ja existe e nao ta vazio."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest  # ja tenho esse arquivo, nao baixa de novo (idempotente)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(get(url, timeout))
    return dest


if __name__ == "__main__":
    # o snapshot_download baixa o repositorio inteiro do dataset de uma vez
    from huggingface_hub import snapshot_download

    print("== [1/2] pothole dataset (Ryukijano) ==")
    snapshot_download(
        repo_id="Ryukijano/Pothole-detection-Yolov8",
        repo_type="dataset",
        local_dir=os.path.join(RAW, "pothole"),
        max_workers=8,  # baixa varios arquivos em paralelo pra ser mais rapido
    )
    print("pothole ok")

    print("== [2/2] ROD obstacle metadata (Abtinzandi) ==")
    # aqui so pegamos os CSVs de metadados dos 3 splits; a imagem de cada linha
    # o build_dataset baixa depois, so das que a gente for realmente usar
    base = "https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO/resolve/main"
    for split in ("train", "valid", "test"):
        p = download(f"{base}/{split}/metadata.csv", os.path.join(RAW, "rod", f"{split}_metadata.csv"))
        print("  ", p, os.path.getsize(p))
