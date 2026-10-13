# Auditoria de Calçadas — CPZÃO (CP2 + CP3)

> **Checkpoint integrado de Visão Computacional** — detecção de defeitos, obstáculos e
> elementos de acessibilidade em calçadas com **YOLOv8n**, do dataset até a inferência em
> tempo real no smartphone (*edge computing*).

| | |
|:--|:--|
| **Disciplina** | Visão Computacional / Inteligência Artificial |
| **Entrega e apresentação** | 13/out (apresentação presencial em aula) |
| **Modelo** | YOLOv8n (Ultralytics 8.x) — exportado em **ONNX @192×192** (TFLite é opcional e não foi gerado; o app usa ONNX) |
| **App** | PWA adaptado do [`yoloinferApp`](https://github.com/Mjrovai/yoloinferApp) (ONNX Runtime Web) |

---

## 1. Integrantes

| # | Nome | RM |
|:-:|:--|:--|
| 1 | Aksel Viktor Caminha Rae | RM99011 |
| 2 | Ian Xavier Kuraoka | RM98860 |
| 3 | Rony Ken Nagai | RM551549 |
| 4 | Tomáz Versolato Carballo | RM551417 |

---

## 2. Problema

Calçadas mal conservadas e sem acessibilidade são um risco permanente para pedestres —
sobretudo para pessoas com deficiência visual, idosos e cadeirantes. A auditoria dessas
condições hoje é manual, lenta e subjetiva.

Este projeto constrói um **pipeline completo de visão computacional** capaz de detectar,
em tempo real e direto na borda (celular), três elementos centrais de uma auditoria urbana:

1. **Defeitos do piso** (`buraco_trinca`) — buracos, trincas e falhas que derrubam ou
   desestabilizam quem passa;
2. **Acessibilidade** (`piso_tatil`) — o piso tátil/faixa guia que orienta pessoas com
   deficiência visual;
3. **Obstáculos** (`obstaculo`) — mobiliário urbano e objetos que obstruem o livre
   trânsito (postes, lixeiras, bancos, cones, bicicletários…).

---

## 3. Classes anotadas

| ID | Classe | O que representa | Exemplos anotados |
|:--:|:--|:--|:--|
| 0 | `buraco_trinca` | defeito do piso | buraco, trinca, destacamento, falha de pavimento |
| 1 | `piso_tatil` | elemento de acessibilidade | faixa guia tátil (piso de potes/barras) |
| 2 | `obstaculo` | obstáculo no passeio | poste, lixeira, banco, cone, hydrante, caixa elétrica, bicicletário |

---

## 4. Dataset

**100 imagens rotuladas · 3 classes · 174 instâncias** — subamostra proposital das bases
públicas (o notebook de aquisição do professor traz +5.000 imagens; aqui o recorte atende ao
mínimo exigido de 60, com **70 imagens de treino**).

| Split | Imagens | `buraco_trinca` | `piso_tatil` | `obstaculo` |
|:--|:--:|:--:|:--:|:--:|
| **train** | **70** | 24 | 23 | 23 |
| val | 20 | 7 | 7 | 6 |
| test | 10 | 3 | 3 | 4 |
| **total** | **100** | 34 | 33 | 33 |

**Instâncias por classe:** `buraco_trinca` 90 · `piso_tatil` 43 · `obstaculo` 41.

### 4.1 Fontes e licenças

| Classe | Base | Formato original | Licença |
|:--|:--|:--|:--|
| `buraco_trinca` | [`Ryukijano/Pothole-detection-Yolov8`](https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8) (Roboflow Universe) | caixas YOLO (`.txt`) | CC BY 4.0 |
| `piso_tatil` | [`qweawq/Blind-road-and-crosswalk-dataset`](https://github.com/qweawq/Blind-road-and-crosswalk-dataset) | **máscara** por pixel → convertida em caixa | domínio público / uso acadêmico |
| `obstaculo` | [`Abtinzandi/Obstacle-Detection-Dataset-YOLO`](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) (ROD, 24k imgs / 25 classes) | caixas YOLO, 12 classes de mobiliário urbano **unificadas** em `obstaculo` | MIT |

### 4.2 Curadoria (reprodutível)

Tudo é gerado por script — nada foi feito "na mão", então o dataset é 100% reproduzível:

```bash
python src/download_sources.py   # baixa as 3 bases (cache em _scratch/raw)
python src/build_dataset.py      # converte, deduplica, faz o split e gera o dataset.yaml
python src/check_images.py       # auditoria de integridade das imagens
```

Decisões de curadoria aplicadas em `src/build_dataset.py`:

* **máscara → caixa**: a máscara do piso tátil passa por fechamento morfológico e
  componentes conectados (área mínima de 0,15% da imagem) para virar caixa YOLO;
* **unificação de taxonomia**: 12 classes de mobiliário do ROD (poste, lixeira, banco,
  cone, hidrante…) viram uma única classe `obstaculo`, alinhada ao objetivo do projeto;
* **deduplicação**: agrupamento por *average hash* (Hamming ≤ 8) para que frames quase
  idênticos do vídeo de origem **não vazem entre treino e teste** (senão a métrica infla);
* **split estratificado por classe** 70/20/10, respeitando os grupos de duplicatas;
* **verificação de integridade**: JPEGs truncados são descartados na coleta.

Figuras de conferência geradas: `docs/figures/qc_*.jpg` (caixas por classe) e
`docs/figures/qa_*.jpg` (amostra por split).

---

## 5. Treino e métricas

**Configuração:** YOLOv8n · `imgsz=320` · `epochs=100` · `batch=8` · `optimizer=auto` com
`cos_lr` · treino em CPU (fp32, numericamente estável) via `python src/train.py --device cpu`
· seed fixa para reprodutibilidade. O notebook `notebooks/02_treinamento_yolo.ipynb` roda o
mesmo fluxo na **GPU do Colab**.

| Split | imgsz | mAP@50 | mAP@50-95 | Precision | Recall |
|:--|:--:|:--:|:--:|:--:|:--:|
| val | 320 | `0.4216` | `0.2018` | `0.4295` | `0.5406` |
| **test** | **320** | `0.2842` | `0.1580` | `0.7854` | `0.2240` |
| test | 192 (app) | `0.3390` | `0.1591` | `0.4945` | `0.3387` |

**mAP@50 por classe (teste):**

| Classe | mAP@50 | AP@50-95 |
|:--|:--:|:--:|
| `buraco_trinca` | `0.3473` | `0.2437` |
| `piso_tatil` | `0.3102` | `0.1133` |
| `obstaculo` | `0.1950` | `0.1170` |

> Valores brutos, curvas de perda e matriz de confusão: **`docs/metrics.md`**,
> **`docs/metrics.json`** e `docs/figures/test_confusion_matrix*.png`.
> Figuras: `docs/figures/results.png` (curvas de perda/mAP), `test_confusion_matrix_normalized.png`.

**Análise.** O modelo foi validado em três frentes: (i) `val` durante o treino para o
early-stopping; (ii) `test` nunca visto, para a métrica oficial; e (iii) `test` **a 192 px**,
que é a resolução real de execução no celular — a diferença entre as duas linhas mede
honestamente o custo de rodar na borda.

---

## 6. Deploy mobile (tempo real no celular)

O [`yoloinferApp`](https://github.com/Mjrovai/yoloinferApp) foi adaptado ao projeto:

* `src/main.js` → `CLASS_NAMES = ['buraco_trinca','piso_tatil','obstaculo']` e modelo
  padrão `/models/calcada_192.onnx`;
* `index.html` → título/identidade do projeto e apenas o modo de detecção;
* modelos COCO de exemplo removidos — o app roda **só** com o modelo da calçada;
* **câmera traseira** (`facingMode: 'environment'`), caixas em tempo real, sliders de
  confiança/IoU, medição de FPS na tela.

### 6.1 Como rodar

```bash
cd app/yoloinferApp/Code
npm install
npm run dev          # desenvolvimento (https automático — a câmera exige HTTPS)
npm run build        # gera dist/ pronto para hospedar
```

No celular:

1. publique/aceda o `dist/` por **HTTPS** (Netlify, GitHub Pages ou `npx serve` + túnel);
2. abra no navegador → **Add to Home Screen** (vira PWA);
3. toque em **Start camera** (a permissão da câmera só aparece em HTTPS/localhost);
4. aponte para a calçada: `buraco_trinca`, `piso_tatil` e `obstaculo` aparecem com caixas,
   rótulo e confiança, com o FPS medido em tempo real.

Alternativa sem build: use o botão **Model** e carregue `models/calcada_192.onnx`.

### 6.2 Formatos exportados

| Arquivo | Formato | Uso |
|:--|:--|:--|
| `models/calcada_192.onnx` | ONNX, entrada `1×3×192×192` | **app (ONNX Runtime Web)** |
| `models/calcada_320.onnx` | ONNX, entrada `1×3×320×320` | versão com mais precisão |
| `models/best.pt` | pesos Ultralytics | retreino / reexportação |

> TFLite não foi gerado (requer TensorFlow instalado). Não é necessário: o app roda em
> ONNX Runtime Web. Para gerar, instale `tensorflow` e rode `src/train.py` novamente.

---

## 7. Estrutura do repositório

```
cp/
├── README.md                     ← este arquivo
├── atividade.md                  ← enunciado do checkpoint
├── requirements.txt
├── data/dataset/                 ← dataset YOLO pronto (100 imgs + labels + dataset.yaml)
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── dataset.yaml
├── notebooks/
│   ├── 01_coleta_curadoria.ipynb     ← coleta/curadoria (reprodutível)
│   ├── 02_treinamento_yolo.ipynb     ← treino no Colab (GPU) + métricas
│   └── 03_deploy_mobile.ipynb        ← exportação e empacotamento do app
├── src/
│   ├── download_sources.py       ← baixa as 3 bases públicas
│   ├── build_dataset.py          ← conversão, dedup, split, dataset.yaml
│   ├── check_images.py           ← auditoria de integridade
│   ├── train.py                  ← treino + val + test + exportação
│   └── zip_project.py            ← gera o .zip da entrega
├── models/                       ← pesos exportados (pt/onnx/tflite)
├── app/yoloinferApp/             ← app adaptado (PWA + ONNX Runtime Web)
├── docs/
│   ├── metrics.md / metrics.json ← métricas completas
│   ├── dataset_stats.json        ← estatísticas do dataset
│   └── figures/                  ← curvas, matriz de confusão, amostras
└── runs/calcada_v1/              ← saída bruta do treino (ultralytics)
```

---

## 8. Como reproduzir do zero

```bash
pip install -r requirements.txt

# 1. dados
python src/download_sources.py
python src/build_dataset.py

# 2. treino + métricas + exportação (rode da raiz do repositório)
python src/train.py --imgsz 320 --epochs 100 --batch 8 --device cpu   # use --device 0 p/ GPU

# 3. app
cd app/yoloinferApp/Code && npm install && npm run dev
```

No Colab, abra `notebooks/02_treinamento_yolo.ipynb` e rode célula a célula (GPU T4).

---

## 9. Checklist da rubrica (20 pontos)

| Etapa | Critério | Pts | Onde está |
|:--|:--|:--:|:--|
| **Dataset & Rotulagem** | ≥60 fotos, 3 classes, split train/val/test | 4,0 | §4, `data/dataset/`, `docs/dataset_stats.json` |
| **Treino & Métricas** | pipeline, análise de mAP/perdas, exportação | 6,0 | §5, `docs/metrics.*`, `notebooks/02`, `models/` |
| **Deploy Mobile** | `yoloinferApp` integrado, inferência fluida | 6,0 | §6, `app/yoloinferApp/` |
| **Organização & GitHub** | README, notebooks, identificação do grupo | 4,0 | §1, §7, §8, notebooks/ |

---

## 10. Referências

1. Metodologia de referência: <https://falzoni.com.br/auditoriacalcadas/>
2. Notebook de aquisição de dados (Colab do professor)
3. Base do app mobile: <https://github.com/Mjrovai/yoloinferApp>
4. Ultralytics YOLOv8 — <https://docs.ultralytics.com>
5. ONNX Runtime Web — <https://onnxruntime.ai/docs/tutorials/web/>

**Atribuição de dados:** Roboflow Universe (CC BY 4.0), ROD Dataset (MIT) e
Blind-road-and-crosswalk-dataset — usados exclusivamente com fins acadêmicos.
