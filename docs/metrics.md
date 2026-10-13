# Métricas do modelo (`docs/metrics.json`)

* Modelo base: YOLOv8n | imgsz de treino: **320** | épocas: **100**

| Split | imgsz | mAP@50 | mAP@50-95 | Precision | Recall |
|:--|:--:|:--:|:--:|:--:|:--:|
| val | 320 | 0.4216 | 0.2018 | 0.4295 | 0.5406 |
| test | 320 | 0.2842 | 0.1580 | 0.7854 | 0.2240 |
| val | 192 | 0.4710 | 0.2332 | 0.6798 | 0.4342 |
| test | 192 | 0.3390 | 0.1591 | 0.4945 | 0.3387 |

### mAP@50 por classe (test)

| Classe | mAP@50 | AP@50-95 |
|:--|:--:|:--:|
| `buraco_trinca` | 0.3473 | 0.2437 |
| `piso_tatil` | 0.3102 | 0.1133 |
| `obstaculo` | 0.1950 | 0.1170 |

### Exportações
* `onnx_192`: `models\calcada_192.onnx`
* `onnx_320`: `models\calcada_320.onnx`
* `tflite`: `não exportado (AssertionError: LiteRT export only supported on Linux x86 and macOS) — app usa ONNX Runtime Web`
