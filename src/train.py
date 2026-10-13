"""
Treino + validacao + exportacao do detector de calcadas (YOLOv8n).

Fluxo completo num arquivo so:
  1) treina o YOLOv8n no dataset;
  2) valida nos splits val e test (na resolucao de treino E na de 192, que e a do celular);
  3) exporta o modelo pra ONNX (192 pro app e na resolucao de treino) e tenta TFLite;
  4) escreve as metricas em docs/metrics.json + metrics.md e copia as figuras pra docs/.

Uso:
    python src/train.py                # treino local (CPU por padrao)
    python src/train.py --imgsz 640 --epochs 100 --device 0   # na GPU

Obs: nesta maquina a GPU se mostrou instavel pra treino (loss explode/NaN em 640),
entao o padrao seguro aqui e CPU com amp=False. Na GPU, rode por sua conta e risco.

Saidas:
    runs/calcada_v1/           -> pesos, curvas, matriz de confusao
    docs/figures/              -> figuras para o README
    docs/metrics.md|json       -> metricas formatadas
    models/                    -> best.onnx (192 p/ app), best_320.onnx, best.tflite
"""
import argparse, glob, json, os, shutil, sys

# caminhos base do projeto (tudo relativo a raiz, que e a pasta acima de src/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset", "dataset.yaml")
MODELS = os.path.join(ROOT, "models")
FIGS = os.path.join(ROOT, "docs", "figures")


def parse_args():
    # argumentos de linha de comando (todos tem default, entao da pra rodar sem passar nada)
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")   # peso base pre-treinado (COCO)
    p.add_argument("--imgsz", type=int, default=320)  # resolucao de treino
    p.add_argument("--epochs", type=int, default=80)  # quantas epocas
    p.add_argument("--batch", type=int, default=16)   # tamanho do batch
    p.add_argument("--name", default="calcada_v1")    # nome da pasta de saida em runs/
    p.add_argument("--device", default="cpu")         # 'cpu' ou '0' pra primeira GPU
    p.add_argument("--skip-train", action="store_true")  # so valida/exporta, sem treinar
    p.add_argument("--app-imgsz", type=int, default=192)  # resolucao do modelo que vai pro celular
    return p.parse_args()


def main():
    a = parse_args()
    from ultralytics import YOLO  # importa aqui dentro pra --help ser rapido mesmo sem o pacote

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    runs_dir = os.path.join(ROOT, "runs", a.name)
    best = None

    if not a.skip_train:
        # apaga um treino antigo com o mesmo nome pra ficar reprodutivo (comeca do zero)
        if os.path.exists(runs_dir):
            shutil.rmtree(runs_dir, ignore_errors=True)
        model = YOLO(a.model)
        results = model.train(
            data=DATA, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
            device=a.device, workers=0, project=os.path.join(ROOT, "runs"),
            name=a.name, patience=30, plots=True, verbose=True,
            # amp=False -> treino em fp32 (mais estavel numericamente; evita NaN da GPU aqui).
            # cos_lr -> learning rate em cosseno; lr0 -> taxa inicial.
            cache=False, amp=False, lr0=0.01, cos_lr=True,
        )
        best = os.path.join(str(results.save_dir), "weights", "best.pt")
    else:
        # modo --skip-train: pega o melhor peso que ja existe
        best = os.path.join(runs_dir, "weights", "best.pt")

    if not os.path.exists(best):
        sys.exit("pesos não encontrados: " + str(best))

    model = YOLO(best)  # recarrega o melhor peso pra validar e exportar
    # dicionario que vai virar o docs/metrics.json
    metrics = {"imgsz_train": a.imgsz, "epochs": a.epochs, "weights": os.path.relpath(best, ROOT)}

    # ---- validacao em val e test (na resolucao de treino) ------------
    for split in ("val", "test"):
        r = model.val(data=DATA, imgsz=a.imgsz, split=split, device=a.device,
                      workers=0, plots=True, verbose=True)
        metrics[f"{split}@{a.imgsz}"] = {
            "mAP50": round(float(r.box.map50), 4),    # mAP com IoU >= 0.5
            "mAP50-95": round(float(r.box.map), 4),   # mAP medio de 0.5 a 0.95 (mais exigente)
            "precision": round(float(r.box.mp), 4),   # precisao media
            "recall": round(float(r.box.mr), 4),      # revocacao media
            # mAP50 quebrado por classe (nome -> valor)
            "per_class_map50": {k: round(float(v), 4) for k, v in
                                zip(r.names.values(), r.box.ap50)},
            "per_class_ap50_95": {k: round(float(v), 4) for k, v in
                                  zip(r.names.values(), r.box.ap)},
        }
        # copia as figuras que o ultralytics gera (matriz de confusao, curvas, results)
        # pra docs/, prefixando com o split, pra usar no README
        src_dir = str(r.save_dir)
        for pat in ("confusion_matrix_normalized*", "confusion_matrix*",
                    "*curve", "results*"):
            for f in glob.glob(os.path.join(src_dir, pat)):
                dst = os.path.join(FIGS, split + "_" + os.path.basename(f))
                shutil.copy2(f, dst)

    # ---- validacao do modelo que vai pro celular (resolucao menor, 192) ----
    # mede quanto a gente perde de qualidade ao rodar no smartphone em 192px
    if a.app_imgsz != a.imgsz:
        for split in ("val", "test"):
            r = model.val(data=DATA, imgsz=a.app_imgsz, split=split, device=a.device,
                          workers=0, plots=False, verbose=True)
            metrics[f"{split}@{a.app_imgsz}"] = {
                "mAP50": round(float(r.box.map50), 4),
                "mAP50-95": round(float(r.box.map), 4),
                "precision": round(float(r.box.mp), 4),
                "recall": round(float(r.box.mr), 4),
                "per_class_map50": {k: round(float(v), 4) for k, v in
                                    zip(r.names.values(), r.box.ap50)},
            }

    # ---- exportacao dos modelos -------------------------------------
    exports = {}
    # 1) ONNX em 192: e esse que o app yoloinferApp carrega (entrada 1x3x192x192)
    try:
        p192 = model.export(format="onnx", imgsz=a.app_imgsz, simplify=True, opset=13)
        shutil.copy2(p192, os.path.join(MODELS, "calcada_192.onnx"))
        exports["onnx_192"] = os.path.relpath(os.path.join(MODELS, "calcada_192.onnx"), ROOT)
    except Exception as e:  # noqa  -> se falhar, registra o erro em vez de quebrar tudo
        exports["onnx_192"] = f"FALHOU: {e}"

    # 2) ONNX na resolucao de treino (pra quem quiser rodar em qualidade maior)
    try:
        p320 = model.export(format="onnx", imgsz=a.imgsz, simplify=True, opset=13)
        shutil.copy2(p320, os.path.join(MODELS, f"calcada_{a.imgsz}.onnx"))
        exports[f"onnx_{a.imgsz}"] = os.path.relpath(os.path.join(MODELS, f"calcada_{a.imgsz}.onnx"), ROOT)
    except Exception as e:  # noqa
        exports[f"onnx_{a.imgsz}"] = f"FALHOU: {e}"

    # 3) TFLite e opcional (precisa do tensorflow instalado); o app usa ONNX, entao
    #    se falhar tudo bem, so anota que nao gerou
    try:
        ptf = model.export(format="tflite", imgsz=a.imgsz)
        shutil.copy2(ptf, os.path.join(MODELS, "calcada.tflite"))
        exports["tflite"] = os.path.relpath(os.path.join(MODELS, "calcada.tflite"), ROOT)
    except Exception as e:  # noqa
        exports["tflite"] = f"não exportado ({type(e).__name__}: {e}) — app usa ONNX Runtime Web"

    # guarda tambem o .pt (peso PyTorch) em models/ pra ficar versionado junto
    shutil.copy2(best, os.path.join(MODELS, "best.pt"))

    # ---- grava metricas em JSON e em Markdown ------------------------
    metrics["exports"] = exports
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    write_markdown(metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def write_markdown(m):
    """Monta um docs/metrics.md bonitinho (tabela) a partir do dicionario de metricas."""
    lines = ["# Métricas do modelo (`docs/metrics.json`)", ""]
    lines.append(f"* Modelo base: YOLOv8n | imgsz de treino: **{m['imgsz_train']}** | épocas: **{m['epochs']}**")
    lines.append("")
    lines.append("| Split | imgsz | mAP@50 | mAP@50-95 | Precision | Recall |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|")
    # varre as chaves tipo "val@320"/"test@192" e monta uma linha de tabela pra cada
    for k, v in m.items():
        if k.startswith(("val@", "test@")):
            split, size = k.split("@")
            lines.append(f"| {split} | {size} | {v['mAP50']:.4f} | {v['mAP50-95']:.4f} "
                         f"| {v['precision']:.4f} | {v['recall']:.4f} |")
    lines.append("")
    # tabela de mAP por classe no teste (na resolucao de treino)
    key = "test@" + str(m["imgsz_train"])
    if key in m:
        lines += ["### mAP@50 por classe (test)", "", "| Classe | mAP@50 | AP@50-95 |", "|:--|:--:|:--:|"]
        for c, v in m[key]["per_class_map50"].items():
            ap = m[key].get("per_class_ap50_95", {}).get(c, float("nan"))
            lines.append(f"| `{c}` | {v:.4f} | {ap:.4f} |")
        lines.append("")
    # lista os arquivos exportados
    lines.append("### Exportações")
    for k, v in m["exports"].items():
        lines.append(f"* `{k}`: `{v}`")
    with open(os.path.join(ROOT, "docs", "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
