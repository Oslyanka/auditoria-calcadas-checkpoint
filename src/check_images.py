"""
Confere se as imagens abrem de verdade (sem JPEG quebrado/truncado).

A ideia aqui e simples: pega um monte de imagens (das fontes brutas ou do dataset
final), tenta abrir e decodificar cada uma, e lista as que estao corrompidas. Serve
pra gente nao descobrir imagem quebrada so na hora do treino.
"""
import os, sys, glob
from PIL import Image

# raiz do projeto = a pasta acima de src/ (pra montar caminhos absolutos)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(pattern):
    """Recebe um padrao glob, tenta abrir cada imagem e devolve (todas, as_ruins)."""
    bad = []
    files = glob.glob(pattern, recursive=True)  # acha todos os arquivos que batem no padrao
    for p in files:
        try:
            with Image.open(p) as im:
                im.load()              # forca a leitura dos pixels (so abrir nao basta)
                _ = im.convert("RGB")  # garante que da pra converter pra RGB (o que o YOLO usa)
        except Exception as e:  # noqa  -> se estourar qualquer erro, a imagem ta ruim
            bad.append((os.path.relpath(p, ROOT), str(e)))
    return files, bad


if __name__ == "__main__":
    # se passar padroes na linha de comando usa eles; senao, usa esta lista padrao
    # (as 3 fontes brutas em _scratch/raw + o dataset final em data/dataset)
    patterns = sys.argv[1:] or [
        os.path.join(ROOT, "_scratch", "raw", "pothole", "**", "*.jpg"),
        os.path.join(ROOT, "_scratch", "raw", "rod", "files", "**", "*.jpg"),
        os.path.join(ROOT, "_scratch", "raw", "blindroad", "**", "*.jpg"),
        os.path.join(ROOT, "data", "dataset", "images", "**", "*.jpg"),
    ]
    # roda a checagem em cada padrao e imprime um resumo (quantas achou / quantas ruins)
    for pat in patterns:
        files, bad = check(pat)
        print(f"{len(files):5d} arquivos | {len(bad):3d} corrompidos | {pat}")
        for p, e in bad:
            print("     !", p, "->", e)  # mostra o arquivo problematico e o erro
