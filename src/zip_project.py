"""
Monta o .zip do projeto pra entrega, deixando de fora o que nao precisa ir
(cache de dados brutos, saidas de treino, node_modules, etc).
"""
import os, zipfile, sys

# raiz do projeto; o zip sai, por padrao, uma pasta acima (ao lado do projeto)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(ROOT), "cp_checkpoint.zip")

# pastas que NAO entram no zip (peso morto ou coisa que se regenera)
EXCLUDE_DIRS = {"_scratch", ".git", "node_modules", "__pycache__", ".ipynb_checkpoints", "runs"}
# extensoes que NAO entram (nao faz sentido zipar um zip dentro do zip)
EXCLUDE_EXTS = {".zip"}


def main():
    n, size = 0, 0  # contadores: quantos arquivos e quantos bytes no total
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # os.walk anda recursivamente por todas as pastas do projeto
        for base, dirs, files in os.walk(ROOT):
            # poda as pastas excluidas NA HORA (mexer em dirs[:] faz o walk nem entrar nelas)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if os.path.splitext(f)[1].lower() in EXCLUDE_EXTS:
                    continue  # pula arquivos com extensao excluida
                p = os.path.join(base, f)
                rel = os.path.relpath(p, ROOT)  # caminho relativo (pra manter a estrutura dentro do zip)
                z.write(p, rel)
                n += 1
                size += os.path.getsize(p)
    # resumo: numero de arquivos, tamanho original e tamanho ja comprimido
    print(f"{OUT}\n{n} arquivos | {size/1e6:.1f} MB -> {os.path.getsize(OUT)/1e6:.1f} MB zip")


if __name__ == "__main__":
    main()
