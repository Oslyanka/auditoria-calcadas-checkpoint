"""
Preenche os placeholders de metricas do README.md usando o docs/metrics.json.

O README vem com marcadores tipo {{VAL320_MAP50}} no lugar dos numeros. Esse script
le as metricas que o treino gerou e troca cada marcador pelo valor real. Assim a gente
nao precisa copiar numero na mao (e nao corre o risco de digitar errado).
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    mp = os.path.join(ROOT, "docs", "metrics.json")
    if not os.path.exists(mp):
        # sem metricas nao tem o que preencher -> avisa e para
        sys.exit("docs/metrics.json ainda não existe")
    m = json.load(open(mp, encoding="utf-8"))
    readme = os.path.join(ROOT, "README.md")
    txt = open(readme, encoding="utf-8").read()

    def val(key, field):
        """Pega m[key][field] com seguranca; se nao existir, devolve None."""
        v = m.get(key)
        if not isinstance(v, dict) or field not in v:
            return None
        return v[field]

    # mapa: marcador no README -> valor correspondente no metrics.json
    # (val/test na resolucao de treino 320 e tambem na resolucao do celular 192)
    pairs = {
        "{{VAL320_MAP50}}": val("val@320", "mAP50"),
        "{{VAL320_MAP5095}}": val("val@320", "mAP50-95"),
        "{{VAL320_P}}": val("val@320", "precision"),
        "{{VAL320_R}}": val("val@320", "recall"),
        "{{TEST320_MAP50}}": val("test@320", "mAP50"),
        "{{TEST320_MAP5095}}": val("test@320", "mAP50-95"),
        "{{TEST320_P}}": val("test@320", "precision"),
        "{{TEST320_R}}": val("test@320", "recall"),
        "{{TEST192_MAP50}}": val("test@192", "mAP50"),
        "{{TEST192_MAP5095}}": val("test@192", "mAP50-95"),
        "{{TEST192_P}}": val("test@192", "precision"),
        "{{TEST192_R}}": val("test@192", "recall"),
    }
    # troca cada marcador; se o valor for numero formata com 4 casas, senao poe `n/d`
    for k, v in pairs.items():
        txt = txt.replace(k, f"{v:.4f}" if isinstance(v, (int, float)) else "`n/d`")

    # tabela de mAP por classe (usa as metricas do split de teste na resolucao de treino)
    per = m.get("test@320", {}).get("per_class_map50", {})
    ap = m.get("test@320", {}).get("per_class_ap50_95", {})
    for i, cls in enumerate(["buraco_trinca", "piso_tatil", "obstaculo"]):
        txt = txt.replace("{{C%d_MAP50}}" % i,
                          f"{per[cls]:.4f}" if cls in per else "`n/d`")
        txt = txt.replace("{{C%d_AP}}" % i,
                          f"{ap[cls]:.4f}" if cls in ap else "`n/d`")

    # confere se sobrou algum marcador sem preencher (util pra pegar erro de digitacao)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", txt)
    open(readme, "w", encoding="utf-8").write(txt)
    print("README atualizado | placeholders restantes:", left or "nenhum")


if __name__ == "__main__":
    main()
