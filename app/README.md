# App mobile — Auditoria de Calçadas (adaptado do `yoloinferApp`)

PWA que roda **ONNX Runtime Web** no navegador do celular: câmera **traseira** em tempo
real, caixas delimitadoras, rótulo + confiança, sliders de *Confidence*/*IoU* e medidor de FPS.

Base original: [Mjrovai/yoloinferApp](https://github.com/Mjrovai/yoloinferApp) (YOLOInfer).

## O que foi adaptado ao CPZÃO

| Arquivo | Mudança |
|:--|:--|
| `src/main.js` | `CLASS_NAMES = ['buraco_trinca','piso_tatil','obstaculo']` e modelo padrão `/models/calcada_192.onnx` |
| `index.html` | título, subtítulo e rodapé do projeto; apenas o modo **detecção** no seletor |
| `public/models/` | modelos COCO de exemplo **removidos**; entra só o `calcada_192.onnx` treinado aqui |
| `src/inference.js` | inalterado — entrada fixa `1×3×192×192` (letterbox + NMS) |

## Rodar

```bash
cd Code
npm install
npm run dev     # desenvolvimento — gera HTTPS automático (câmera exige HTTPS)
npm run build   # gera dist/ pronto para publicar
```

No celular:

1. publique o `dist/` em **HTTPS** (Netlify, GitHub Pages ou túnel);
2. abra no navegador → **Add to Home Screen** (instala como PWA);
3. toque em **Start camera** → detecção ao vivo;
4. ou toque em **Model** e escolha `models/calcada_192.onnx` manualmente.

## Detalhes técnicos

* pré-processamento: *letterbox* 192×192 → normalização 0-1 (RGB);
* pós-processamento: decode das caixas, NMS (IoU configurável) e desenho no `<canvas>`;
* rotação dos eixos: o overlay converte as coordenadas do recorte/letterbox de volta para
  a resolução original do vídeo, então as caixas acompanham a rotação do celular;
* performace: o modelo `calcada_192.onnx` roda em ~192×192 para manter a inferência fluida
  em CPU/WASM — troque a constante `INPUT_W/INPUT_H` em `src/inference.js` e use
  `calcada_320.onnx` se quiser mais precisão e tiver um aparelho mais potente.
