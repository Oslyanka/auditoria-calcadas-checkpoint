"""
Monta o dataset do CPZAO (Auditoria de Calcadas) no formato YOLO.

A sacada aqui e: em vez de sair fotografando calcada na rua, a gente aproveita 3 bases
publicas que JA sao rotuladas, pega so o que interessa de cada uma e junta tudo numa
taxonomia unica de 3 classes:

  A) buraco_trinca  -> HF Ryukijano/Pothole-detection-Yolov8      (ja vem em caixa YOLO)
  B) piso_tatil     -> Blind road & crosswalk dataset (GitHub)     (vem em mascara -> viro caixa)
  C) obstaculo      -> HF Abtinzandi/Obstacle-Detection-Dataset-YOLO (25 classes -> junto em 1)

Saida: data/dataset/{images,labels}/{train,val,test} + dataset.yaml + estatisticas em docs/.
"""
import os, csv, json, glob, shutil, random, hashlib, io, sys
import urllib.request
from collections import Counter, defaultdict

import cv2
import numpy as np

# caminhos base
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "_scratch", "raw")   # fontes brutas baixadas (fora do git)
OUT = os.path.join(ROOT, "data", "dataset")    # dataset final que o treino consome
DOCS = os.path.join(ROOT, "docs")              # estatisticas + figuras de QA

CLASSES = ["buraco_trinca", "piso_tatil", "obstaculo"]  # ordem = indice da classe no label
SPLIT = {"train": 70, "val": 20, "test": 10}            # quantas imagens por split (total 100)
TARGET = {"buraco_trinca": 34, "piso_tatil": 33, "obstaculo": 33}  # quantas imagens por classe (~equilibrado)

ROD_BASE = "https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO/resolve/main"
# das 25 classes do ROD, essas sao as que fazem sentido como "obstaculo de calcada"
# (poste, lixeira, banco, cone, hidrante, placa, etc) -> todas viram a classe 2
ROD_OBSTACLE_IDS = {5, 6, 9, 13, 17, 18, 19, 20, 21, 22, 23, 24}
ROD_NAMES = ["Bike", "Building", "Car", "Person", "Stairs", "Traffic sign", "Electrical Pole",
             "Road", "Motorcycle", "Dustbin", "Dog", "Manhole", "Tree", "Guard rail",
             "Pedestrian crosswalk", "Truck", "Bus", "Bench", "Traffic Cone", "Fire hydrant",
             "Teraffic Barrel", "Plant Pot", "Electrical Box", "Chair", "Bicycle Rack"]

# ----------------------------------------------------------------------------
# helpers (funcoes auxiliares usadas por todo o resto)
# ----------------------------------------------------------------------------

def ahash(path, size=8):
    """
    'average hash' da imagem: uma assinatura de 64 bits (8x8) que resume o visual.
    Imagens parecidas tem hash parecido -> uso isso pra achar quase-duplicatas.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)  # encolhe pra 8x8
    return (small > small.mean()).flatten().astype(np.uint8)  # cada pixel: acima ou abaixo da media


def hamming(a, b):
    """Distancia de Hamming: em quantos bits os dois hashes diferem (quanto menor, mais parecido)."""
    return int(np.count_nonzero(a != b))


def valid_image(path):
    """Checa se a imagem decodifica inteira (evita JPEG truncado entrar no dataset)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa
        return False


def download(url, dest):
    """Baixa url -> dest, pulando se o arquivo ja existe (idempotente, igual ao download_sources)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 cpzao-dataset"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)  # copia em streaming (nao carrega tudo na memoria)
    return dest


def read_yolo(path, allowed=None):
    """
    Le um arquivo de label YOLO e devolve lista de (classe, xc, yc, w, h).
    Se 'allowed' for passado, so mantem as linhas cujas classes estao nesse conjunto.
    """
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue  # linha quebrada, ignora
            cls = int(float(p[0]))
            if allowed is not None and cls not in allowed:
                continue  # classe que nao interessa
            out.append((cls, *[float(x) for x in p[1:5]]))
    return out


# ----------------------------------------------------------------------------
# fonte A: buraco / trinca  (pothole, ja vem em caixa YOLO)
# ----------------------------------------------------------------------------

def collect_potholes(target):
    """Pega 'target' imagens de buraco que tenham caixa (classe 0) e abram direito."""
    base = os.path.join(RAW, "pothole")
    items = []
    for split in ("train", "valid", "test"):  # varre os 3 splits originais da fonte
        idir, ldir = os.path.join(base, split, "images"), os.path.join(base, split, "labels")
        if not os.path.isdir(idir):
            continue
        for img in sorted(glob.glob(os.path.join(idir, "*"))):
            stem = os.path.splitext(os.path.basename(img))[0]  # nome sem extensao
            boxes = read_yolo(os.path.join(ldir, stem + ".txt"), allowed={0})  # so a classe 0 (buraco)
            if boxes and valid_image(img):
                items.append({"img": img, "boxes": boxes, "src": "Ryukijano/pothole"})
    random.seed(11)           # semente fixa -> sempre pega as mesmas imagens (reprodutivel)
    random.shuffle(items)
    return items[:target]     # corta no alvo


# ----------------------------------------------------------------------------
# fonte B: piso tatil (a anotacao vem como mascara de pixels -> converto pra caixa)
# ----------------------------------------------------------------------------

def mask_to_boxes(mask_path, min_area_ratio=0.0015, close_k=15):
    """
    Converte a mascara de segmentacao do piso tatil em caixas YOLO.
    Passos: isola a cor do piso tatil -> fecha buracos -> acha blobs -> vira caixa.
    """
    m = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # le a mascara em BGR
    if m is None:
        return []
    # na base chinesa, piso tatil (盲道) e pintado de RGB (128,128,0), que em BGR fica (0,128,128)
    blind = (m[:, :, 0] == 0) & (m[:, :, 1] == 128) & (m[:, :, 2] == 128)
    if blind.sum() == 0:
        return []  # nao tem piso tatil nessa mascara
    binm = (blind * 255).astype(np.uint8)  # mascara binaria (0/255)
    k = np.ones((close_k, close_k), np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)  # fecha falhas/buraquinhos na mascara
    # acha os componentes conectados (cada blob separado = um pedaco de piso tatil)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    H, W = binm.shape
    boxes = []
    for i in range(1, n):  # i=0 e o fundo, por isso comeca em 1
        x, y, w, h, area = stats[i]
        if area < min_area_ratio * H * W:
            continue  # blob pequeno demais -> provavelmente ruido
        if w < 8 or h < 8:
            continue  # caixa minuscula -> descarta
        # converte (x,y,w,h) em pixels pra formato YOLO normalizado (centro + tamanho em 0..1)
        boxes.append((1, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
    return boxes


def collect_tactile(target):
    """Coleta imagens de piso tatil, converte mascara->caixa e tira quase-duplicatas (frames de video)."""
    base = os.path.join(RAW, "blindroad", "Blind road and crosswalk dataset")
    items = []
    for split in ("train", "val", "test"):
        for img in sorted(glob.glob(os.path.join(base, split, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            mpath = os.path.join(base, split + "_labels", stem + ".png")  # mascara correspondente
            if not os.path.exists(mpath):
                continue
            if not valid_image(img):
                continue
            boxes = mask_to_boxes(mpath)
            if boxes:
                items.append({"img": img, "boxes": boxes, "src": "qweawq/blind-road"})
    # essa base e feita de frames de video, entao tem muita imagem quase igual.
    # aqui mantenho so 1 por grupo de visual parecido (hamming <= 8 no average hash)
    kept, seen = [], []
    for it in sorted(items, key=lambda x: x["img"]):
        h = ahash(it["img"])
        if h is None:
            continue
        if any(hamming(h, s) <= 8 for s in seen):
            continue  # ja tenho uma imagem quase identica a essa
        seen.append(h)
        kept.append(it)
    random.seed(12)
    random.shuffle(kept)
    return kept[:target]


# ----------------------------------------------------------------------------
# fonte C: obstaculo (mobiliario urbano do dataset ROD)
# ----------------------------------------------------------------------------

def collect_obstacles(target):
    """Coleta imagens com mobiliario urbano, baixando so as que a gente vai usar."""
    rows = []
    # 1) le os CSVs de metadados dos 3 splits (cada linha descreve uma imagem e suas classes)
    for split in ("train", "valid", "test"):
        p = os.path.join(RAW, "rod", f"{split}_metadata.csv")
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r["_split"] = split
                rows.append(r)

    def classes(r):
        # conjunto de ids de classe presentes naquela imagem
        return {int(x) for x in r["class_ids"].split(",") if x.strip()}

    # 'pure' = imagens que SO tem obstaculo (cena limpa de mobiliario)
    pure = [r for r in rows if classes(r) and classes(r) <= ROD_OBSTACLE_IDS]
    pure_names = {r["file_name"] for r in pure}
    # 'mixed' = imagens que tem obstaculo mas tambem outras coisas (cena de rua mais realista)
    mixed = [r for r in rows if classes(r) & ROD_OBSTACLE_IDS and r["file_name"] not in pure_names]
    random.seed(13)
    random.shuffle(pure)
    random.shuffle(mixed)
    # intercala puras e mistas pra ter variedade; se nao houver puras, usa so mistas
    if not pure:
        picked = mixed[: target * 3]
    else:
        picked = [x for pair in zip(pure, mixed + [None] * len(pure)) for x in pair if x is not None]
        picked = picked[: target * 3]  # pega com folga (3x) porque algumas podem falhar no download

    items = []
    for r in picked:
        sp = r["_split"]
        img_rel, lbl_rel = r["file_name"], r["label_file"]
        img_url = f"{ROD_BASE}/{sp}/{img_rel}"
        lbl_url = f"{ROD_BASE}/{sp}/{lbl_rel}"
        cache_img = os.path.join(RAW, "rod", "files", sp, img_rel)
        cache_lbl = os.path.join(RAW, "rod", "files", sp, lbl_rel)
        try:
            # so agora baixa de fato a imagem e o label dessa linha
            download(img_url, cache_img)
            download(lbl_url, cache_lbl)
        except Exception as e:  # noqa
            print("   ! falha ao baixar", img_rel, e)
            continue
        # fica so com as caixas das classes de obstaculo, e marca todas como classe 2
        boxes = []
        for cls, xc, yc, w, h in read_yolo(cache_lbl, allowed=ROD_OBSTACLE_IDS):
            boxes.append((2, xc, yc, w, h))
        if boxes and valid_image(cache_img):
            items.append({"img": cache_img, "boxes": boxes, "src": "Abtinzandi/ROD"})
        if len(items) >= target:
            break  # ja juntei o suficiente, para de baixar
    return items[:target]


# ----------------------------------------------------------------------------
# split (divide em train/val/test de forma estratificada, sem vazar duplicata)
# ----------------------------------------------------------------------------

def make_splits(per_class):
    """
    Divide as imagens em train/val/test POR classe (estratificado), com uma regra importante:
    quase-duplicatas tem que cair TODAS no mesmo split, senao o modelo 'cola' (ve no treino algo
    quase igual ao que esta no teste) e a metrica fica mentirosa.
    Devolve {classe: {'train': [...], 'val': [...], 'test': [...]}}.
    """
    # quantas imagens de cada classe vao pra cada split (soma = alvo de cada classe)
    plan = {"buraco_trinca": {"train": 24, "val": 7, "test": 3},
            "piso_tatil":    {"train": 23, "val": 7, "test": 3},
            "obstaculo":     {"train": 23, "val": 6, "test": 4}}
    out = {}
    for cls, items in per_class.items():
        # 1) agrupa as imagens parecidas em 'grupos de duplicata'
        groups, hashes = [], []
        for it in items:
            h = ahash(it["img"])
            gid = None
            for i, gh in enumerate(groups):
                if h is not None and gh["hash"] is not None and hamming(h, gh["hash"]) <= 8:
                    gid = i  # achou um grupo parecido, entra nele
                    break
            if gid is None:
                groups.append({"hash": h, "items": [it]})  # grupo novo
            else:
                groups[gid]["items"].append(it)
        random.seed(21)
        random.shuffle(groups)

        # 2) distribui os GRUPOS inteiros pelos splits respeitando as cotas do plano
        alloc = plan[cls]
        splits = {"train": [], "val": [], "test": []}
        counts = {k: 0 for k in splits}
        order = ["train", "val", "test"]
        for g in groups:
            placed = False
            for s in order:
                # coloca o grupo inteiro no primeiro split que ainda tem vaga
                if counts[s] + len(g["items"]) <= alloc[s]:
                    splits[s].extend(g["items"])
                    counts[s] += len(g["items"])
                    placed = True
                    break
            if not placed:
                # se nao coube em lugar nenhum pela cota, joga no split com mais vaga sobrando
                s = max(order, key=lambda k: alloc[k] - counts[k])
                splits[s].extend(g["items"])
                counts[s] += len(g["items"])
        out[cls] = splits
    return out


# ----------------------------------------------------------------------------
# main (orquestra tudo: coleta -> split -> grava imagens/labels/yaml/stats/QA)
# ----------------------------------------------------------------------------

def main():
    random.seed(0)
    # zera a pasta de saida pra o build ser 100% reprodutivo (sempre do zero)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    for s in SPLIT:
        os.makedirs(os.path.join(OUT, "images", s), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", s), exist_ok=True)

    # coleta cada classe da sua fonte
    print("-> coletando buraco_trinca ...")
    buraco = collect_potholes(TARGET["buraco_trinca"])
    print("   ", len(buraco), "imagens")
    print("-> coletando piso_tatil ...")
    piso = collect_tactile(TARGET["piso_tatil"])
    print("   ", len(piso), "imagens")
    print("-> coletando obstaculo ...")
    obst = collect_obstacles(TARGET["obstaculo"])
    print("   ", len(obst), "imagens")

    per_class = {"buraco_trinca": buraco, "piso_tatil": piso, "obstaculo": obst}
    splits = make_splits(per_class)  # divide em train/val/test

    index = []  # lista com cada arquivo gerado (pra rastreabilidade)
    stats = {"classes": {}, "splits": {}, "sources": {}}
    prefix = {"buraco_trinca": "buraco", "piso_tatil": "piso", "obstaculo": "obst"}

    # grava as imagens e os labels renomeados e numerados (buraco_001.jpg, piso_001.jpg, ...)
    for cls, cls_splits in splits.items():
        cid = CLASSES.index(cls)
        n = 0
        for split, items in cls_splits.items():
            for it in items:
                n += 1
                name = f"{prefix[cls]}_{n:03d}"  # nome padronizado com 3 digitos
                dst_img = os.path.join(OUT, "images", split, name + ".jpg")
                shutil.copy2(it["img"], dst_img)  # copia a imagem pro dataset final
                # normaliza as caixas (garante 0..1) e forca a classe certa
                boxes = []
                for c, xc, yc, w, h in it["boxes"]:
                    c = cid if cls != "obstaculo" else 2
                    xc, yc, w, h = [min(max(v, 0.0), 1.0) for v in (xc, yc, w, h)]  # clipa pra [0,1]
                    boxes.append((c, xc, yc, w, h))
                # escreve o arquivo de label YOLO correspondente
                with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
                    for b in boxes:
                        f.write("%d %.6f %.6f %.6f %.6f\n" % b)
                index.append({"file": f"images/{split}/{name}.jpg", "class": cls,
                              "split": split, "boxes": len(boxes), "source": it["src"]})
            stats["splits"][split] = stats["splits"].get(split, 0) + len(items)
        # estatisticas por classe: quantas imagens, quantas caixas e distribuicao por split
        stats["classes"][cls] = {"imagens": n,
                                 "instancias": sum(len(i["boxes"]) for i in sum(splits[cls].values(), [])),
                                 "por_split": {k: len(v) for k, v in cls_splits.items()}}

    # conta de qual fonte veio cada imagem
    for e in index:
        stats["sources"][e["source"]] = stats["sources"].get(e["source"], 0) + 1

    # escreve o dataset.yaml (o arquivo que o ultralytics le pra saber caminhos e classes)
    yaml_path = os.path.join(OUT, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Dataset CPZÃO - Auditoria de Calçadas (gerado por src/build_dataset.py)\n")
        f.write("# IMPORTANTE: rode os comandos a partir da raiz do repositório (path é relativo ao cwd)\n")
        f.write("path: data/dataset\n")
        for s in ("train", "val", "test"):
            f.write(f"{s}: images/{s}\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    # salva as estatisticas completas + o indice em docs/dataset_stats.json
    stats["total_imagens"] = len(index)
    stats["total_instancias"] = sum(e["boxes"] for e in index)
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "index": index}, f, indent=2, ensure_ascii=False)

    # gera uma amostra visual (QA): 6 imagens de cada split com as caixas desenhadas,
    # pra conferir no olho se a anotacao ficou certa
    figs = os.path.join(DOCS, "figures")
    os.makedirs(figs, exist_ok=True)
    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0)]  # cor por classe em BGR (vermelho/azul/verde)
    for split in ("train", "val", "test"):
        imgs = sorted(glob.glob(os.path.join(OUT, "images", split, "*.jpg")))[:6]
        tiles = []
        for p in imgs:
            im = cv2.imread(p)
            H, W = im.shape[:2]
            # acha o label correspondente trocando .../images/... por .../labels/... e .jpg por .txt
            lbl = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace(".jpg", ".txt")
            for c, xc, yc, w, h in read_yolo(lbl):
                # desconverte YOLO (centro+tamanho normalizado) pra cantos em pixel
                x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
                x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
                cv2.rectangle(im, (x1, y1), (x2, y2), colors[c], 2)  # desenha a caixa
                cv2.putText(im, CLASSES[c], (x1, max(14, y1 - 4)),   # escreve o nome da classe
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[c], 2, cv2.LINE_AA)
            tiles.append(cv2.resize(im, (320, int(320 * im.shape[0] / im.shape[1]))))  # miniatura
        # cola as miniaturas lado a lado num 'mosaico' branco
        hmax = max(t.shape[0] for t in tiles)
        canvas = np.full((hmax, 320 * len(tiles), 3), 255, np.uint8)
        for i, t in enumerate(tiles):
            canvas[:t.shape[0], i * 320:(i + 1) * 320] = t
        cv2.imwrite(os.path.join(figs, f"qa_{split}.jpg"), canvas)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("dataset ->", OUT)


if __name__ == "__main__":
    main()
