"""
Gera os 3 notebooks do checkpoint (arquivos .ipynb validos).

Em vez de editar notebook na mao (que e chato e quebra facil), a gente escreve o conteudo
aqui como texto e esse script monta o JSON do .ipynb certinho. Os 3 notebooks sao:
  01 -> coleta e curadoria dos dados
  02 -> treino e validacao do YOLOv8n
  03 -> exportacao e deploy no celular
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks")

# "molde" de um notebook vazio: metadados + kernel Python 3 + config de GPU no Colab
NB = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(text):
    """Cria uma celula de markdown a partir de um texto (quebra em linhas)."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")}


def code(text):
    """Cria uma celula de codigo a partir de um texto (quebra em linhas)."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").split("\n")}


def build(cells):
    """Monta o notebook final: copia o molde e ajeita as quebras de linha de cada celula."""
    nb = json.loads(json.dumps(NB))  # copia profunda do molde (pra nao compartilhar o mesmo dict)
    nb["cells"] = cells
    # no formato .ipynb, toda linha menos a ultima precisa terminar com \n
    for c in nb["cells"]:
        c["source"] = [l + "\n" for l in c["source"][:-1]] + [c["source"][-1]]
    return nb


# ---------------------------------------------------------------- 01
# Notebook 1: explica e executa a coleta/curadoria chamando os scripts de src/
nb01 = build([
    md("""
# 01 — Coleta e Curadoria de Dados

**CPZÃO · Auditoria de Calçadas · 3 classes**

Este notebook obtém as imagens rotuladas a partir de três bases públicas, aplica a curadoria
(deduplicação, conversão máscara→caixa, unificação de taxonomia) e monta o dataset YOLO com
split `train/val/test`.

| Classe | Fonte | Anotação original |
|:--|:--|:--|
| `buraco_trinca` | HF `Ryukijano/Pothole-detection-Yolov8` | caixas YOLO |
| `piso_tatil` | GitHub `qweawq/Blind-road-and-crosswalk-dataset` | máscara por pixel |
| `obstaculo` | HF `Abtinzandi/Obstacle-Detection-Dataset-YOLO` | caixas YOLO (12 classes → 1) |

> No Colab, faça upload da pasta do projeto ou clone o repositório antes de rodar.
"""),
    code("""
# Ambiente (Colab) — pule esta célula se estiver rodando localmente
import os, sys
if 'google.colab' in sys.modules:
    # instala o minimo pra rodar a coleta no Colab
    !pip -q install ultralytics huggingface_hub opencv-python-headless
    # opções: subir o zip do projeto pelo painel lateral ou clonar o repositório
    # !git clone <URL_DO_SEU_REPOSITORIO>.git projeto
    # os.chdir('projeto')
# garante que estamos na raiz do projeto (senao os caminhos relativos quebram)
assert os.path.exists('requirements.txt'), 'Rode a partir da raiz do repositório'
!pip -q install -r requirements.txt
"""),
    code("""
# 1) baixa as 3 bases públicas (cache local em _scratch/raw)
%run src/download_sources.py
"""),
    code("""
# 2) curadoria + split + dataset.yaml (o grosso do trabalho mora aqui)
%run src/build_dataset.py
"""),
    code("""
# 3) auditoria de integridade das imagens do dataset final (pega JPEG quebrado)
%run src/check_images.py data/dataset/images/*/*.jpg
"""),
    md("### Estatísticas do dataset"), code("""
# le o resumo gerado pelo build_dataset e mostra numeros + tabela por classe
import json, pandas as pd
s = json.load(open('docs/dataset_stats.json'))['stats']
print('total de imagens :', s['total_imagens'])
print('total instâncias :', s['total_instancias'])
print('por split        :', s['splits'])
pd.DataFrame(s['classes']).T
"""),
    md("### Amostra visual com as caixas (QA)"), code("""
# mostra os mosaicos de conferencia (imagens com as caixas desenhadas) de cada split
from IPython.display import Image, display
for split in ('train', 'val', 'test'):
    display(Image(filename=f'docs/figures/qa_{split}.jpg', width=900))
"""),
    md("""
### Decisões de curadoria
* **máscara → caixa**: fechamento morfológico + componentes conectados (área mín. 0,15%);
* **unificação**: 12 classes de mobiliário urbano do ROD → `obstaculo`;
* **deduplicação** por *average hash* (Hamming ≤ 8): frames quase idênticos ficam no **mesmo** split;
* **split estratificado** 70/20/10 respeitando os grupos de duplicatas;
* **integridade**: JPEG truncado é descartado na coleta.
"""),
])

# ---------------------------------------------------------------- 02
# Notebook 2: treina o YOLOv8n e analisa as metricas
nb02 = build([
    md("""
# 02 — Treino e Validação (YOLOv8n)

**CPZÃO · Auditoria de Calçadas**

* Dataset: 100 imagens (70 train / 20 val / 10 test), 3 classes
* Modelo: **YOLOv8n** (leve, compatível com exportação mobile)
* Métricas: mAP@50, mAP@50-95, Precision, Recall, matriz de confusão e curvas de perda

> No Colab: **Runtime → Change runtime type → GPU (T4)** antes de rodar.
"""),
    code("""
# checa ambiente e se o dataset ja existe (tem que ter rodado o notebook 01 antes)
import sys, subprocess, glob, os
if 'google.colab' in sys.modules:
    subprocess.run('pip -q install ultralytics'.split(), check=True)
    # cloude/suba o projeto e entre na raiz do repositório
    # !git clone <URL_DO_SEU_REPOSITORIO>.git projeto && %cd projeto
assert os.path.exists('data/dataset/dataset.yaml'), 'dataset não encontrado — rode o notebook 01'
import torch, ultralytics
print('device  :', 'CUDA' if torch.cuda.is_available() else 'CPU')
print('ultralytics', ultralytics.__version__)
"""),
    code("""
# treino propriamente dito: parte do YOLOv8n pre-treinado e ajusta pras nossas 3 classes
from ultralytics import YOLO
data = 'data/dataset/dataset.yaml'
model = YOLO('yolov8n.pt')          # pesos pré-treinados (COCO)
results = model.train(
    data=data, imgsz=320, epochs=80, batch=16,
    patience=30, cos_lr=True, lr0=0.01,   # para cedo se estagnar; lr em cosseno
    project='runs', name='calcada_v1', exist_ok=True,
    device=0 if torch.cuda.is_available() else 'cpu',  # usa GPU se tiver, senao CPU
    workers=0 if 'google.colab' not in sys.modules else 2,
    seed=0, plots=True,                   # seed fixa p/ reprodutibilidade; plots=True gera as curvas
)
print('melhores pesos:', results.save_dir + '/weights/best.pt')
"""),
    md("### Curvas de treino (perdas, mAP e matriz de confusão)"), code("""
# exibe as figuras que o treino gerou: curvas de perda/mAP, matriz de confusao e um lote de predicoes
from IPython.display import Image, display, clear_output
d = 'runs/calcada_v1'
display(Image(f'{d}/results.png', width=950))
display(Image(f'{d}/confusion_matrix_normalized.png', width=520))
display(Image(f'{d}/val_batch0_pred.jpg', width=900))
"""),
    md("### Avaliação nos splits `val` e `test`"), code("""
# roda a validacao no melhor peso e imprime as metricas por split (geral e por classe)
model = YOLO('runs/calcada_v1/weights/best.pt')
rows = {}
for split in ('val', 'test'):
    r = model.val(data=data, split=split, imgsz=320, plots=True, name=split)
    rows[split] = dict(mAP50=r.box.map50, mAP50_95=r.box.map,
                       precision=r.box.mp, recall=r.box.mr)
    per = {name: round(float(v), 4) for name, v in zip(r.names.values(), r.box.map50_class)}
    print(split, '-> mAP50 =', round(r.box.map50, 4), '| por classe:', per)
"""),
    md("### Avaliação na resolução de execução no celular (192 px)"), code("""
# mede a performance na mesma resolucao em que o modelo vai rodar no smartphone (192)
r = model.val(data=data, split='test', imgsz=192, plots=False, name='test192')
print('test@192 -> mAP50 =', round(r.box.map50, 4), '| mAP50-95 =', round(r.box.map, 4))
"""),
    md("""
### Análise das métricas
* **mAP@50** resume a qualidade geral da detecção (IoU ≥ 0,5);
* **mAP@50-95** penaliza caixas imprecisas — o número que revela a qualidade da anotação;
* **Precision × Recall** indicam o equilíbrio entre falsos positivos e falsos negativos (um
  modelo com Recall baixo "deixa passar" defeitos na auditoria);
* a comparação `test@320` × `test@192` quantifica a perda de rodar no smartphone.
"""),
])

# ---------------------------------------------------------------- 03
# Notebook 3: exporta o modelo e prepara o app PWA pro celular
nb03 = build([
    md("""
# 03 — Exportação e Deploy Mobile

Exporta os pesos treinados para os formatos usados no celular e prepara o PWA
`yoloinferApp` (ONNX Runtime Web) com o modelo da calçada.
"""),
    code("""
# exporta o best.pt pra ONNX (192 e 320) e tenta TFLite (opcional); guarda tudo em models/
from ultralytics import YOLO
import os, shutil, glob
os.makedirs('models', exist_ok=True)
model = YOLO('runs/calcada_v1/weights/best.pt')

exports = {}
exports['onnx_192'] = model.export(format='onnx', imgsz=192, simplify=True, opset=13)  # esse vai pro app
exports['onnx_320'] = model.export(format='onnx', imgsz=320, simplify=True, opset=13)  # versao maior
try:
    exports['tflite'] = model.export(format='tflite', imgsz=320)
except Exception as e:
    exports['tflite'] = f'opcional, não gerado: {e}'  # app usa ONNX, entao TFLite nao e obrigatorio
exports['pt'] = shutil.copy('runs/calcada_v1/weights/best.pt', 'models/best.pt')
for k, v in exports.items():
    print(f'{k:10s} -> {v}')
"""),
    md("### Verificação: ONNX carrega e inferência funciona"), code("""
# testa o ONNX exportado rodando uma inferencia de verdade nas imagens de teste
import numpy as np
from ultralytics import YOLO
m = YOLO('models/calcada_192.onnx')
r = m.predict(source='data/dataset/images/test', imgsz=192, conf=0.25, save=True, project='runs', name='onnx_check')
print('classes:', r[0].names)
print('detectões na 1ª imagem:', len(r[0].boxes))
"""),
    md("""
### Colocar o modelo no app

O app carrega `/models/calcada_192.onnx` como modelo padrão (entrada `1×3×192×192`,
mesmo tamanho do `INPUT_W/H` de `src/inference.js`).
"""),
    code("""
# copia o modelo pro diretorio publico do app, pra ele ja abrir com o modelo da calcada
app_models = 'app/yoloinferApp/Code/public/models'
os.makedirs(app_models, exist_ok=True)
shutil.copy('models/calcada_192.onnx', f'{app_models}/calcada_192.onnx')
print('copiado para', app_models)
print(os.listdir(app_models))
"""),
    md("""
### Rodar no celular

```bash
cd app/yoloinferApp/Code
npm install
npm run dev     # desenvolvimento (HTTPS automático)
npm run build   # gera dist/ para publicar
```

1. publique o `dist/` em **HTTPS** (Netlify / GitHub Pages / túnel);
2. no celular: abrir no navegador → **Add to Home Screen** (PWA);
3. **Start camera** → a câmera traseira (`environment`) inicia e as caixas aparecem em tempo real;
4. use os sliders de *Confidence* e *IoU* e observe o medidor de FPS.

Sem build, clique em **Model** e selecione `models/calcada_192.onnx`.
"""),
])


def main():
    os.makedirs(OUT, exist_ok=True)
    # escreve cada notebook e, logo em seguida, tenta reabrir pra garantir que o JSON ficou valido
    for name, nb in (("01_coleta_curadoria.ipynb", nb01),
                     ("02_treinamento_yolo.ipynb", nb02),
                     ("03_deploy_mobile.ipynb", nb03)):
        p = os.path.join(OUT, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        # valida (se o json estiver quebrado, estoura aqui na hora de abrir)
        json.load(open(p, encoding="utf-8"))
        print("ok", p)


if __name__ == "__main__":
    main()
