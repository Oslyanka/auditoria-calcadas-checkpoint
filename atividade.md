# Checkpoint CPZÃO (CP2 + CP3) — Auditoria de Calçadas

* **Valor total:** 20 pontos *(unifica CP2 e CP3 — 10 pontos cada)*
* **Formato de trabalho:** Grupos de até **5 alunos**
* **Data de entrega e apresentação:** **13 de outubro** (apresentação presencial em aula)

---

## Contexto e Objetivo

A qualidade das calçadas e a acessibilidade urbana são desafios centrais em grandes cidades. O objetivo deste checkpoint integrado (**CPZÃO**) é construir um pipeline completo de Visão Computacional aplicado à vida real: desde a aquisição e curadoria dos dados até a inferência em tempo real na borda (*edge computing* / smartphone).

O desafio consiste em auditar condições de calçadas detectando defeitos, obstáculos ou elementos de acessibilidade via detecção de objetos (**YOLO**), culminando na execução do modelo treinado diretamente em um dispositivo móvel.

---

## Materiais de Apoio e Referências

1. **Metodologia de Referência:**
   * [Auditoria de Calçadas - Projeto e Metodologia](https://falzoni.com.br/auditoriacalcadas/)
   *(Contém a fundamentação, metodologia do projeto e repositórios de dados)*

2. **Notebook de Aquisição de Dados:**
   * [Google Colab - Download e Preparação dos Dados](https://colab.research.google.com/drive/1WwkI80sm8gJfrYIiCIOiysChPeg1Km_B?usp=sharing)
   *(Notebook de apoio para coletar e estruturar a base inicial)*

3. **Repositório Base para Deploy Mobile:**
   * [GitHub - yoloinferApp (Mjrovai)](https://github.com/Mjrovai/yoloinferApp)
   *(Aplicação base para carregar o modelo exportado e rodar a inferência no celular)*

---

## Etapas do Projeto

### 1. Coleta e Curadoria de Dados
* Utilizem o notebook disponibilizado e as fontes do projeto de referência para obter as imagens.
* **Volume mínimo exigido:** Pelo menos **60 imagens** rotuladas.
* **Classes:** Definir e anotar pelo menos **3 classes** distintas relevantes para a auditoria urbana (ex.: *buraco/trinca*, *poste/obstáculo*, *piso tátil*, *degrau/desnível*, *lixeira*, etc.).
* Ferramentas de anotação sugeridas: Roboflow, CVAT ou LabelImg.

### 2. Treinamento e Validação (YOLO)
* Dividir os dados em treino, validação e teste.
* Treinar um modelo da família YOLO (por exemplo, YOLOv8n ou versão leve compatível com exportação mobile).
* Analisar as métricas de desempenho (mAP@50, Precision, Recall, Matriz de Confusão).
* Exportar os pesos do modelo para o formato adequado de inferência mobile (ex.: TFLite / ONNX, conforme suporte do app).

### 3. Deploy Mobile (Inferência no Celular)
* Integrar o modelo exportado ao projeto **`yoloinferApp`** adaptado para o projeto de calçada.
* Garantir que o aplicativo execute no celular processando a câmera *traseira* em tempo real com detecção e caixas delimitadoras funcionais.

---

## Entregáveis

1. **Repositório no GitHub:**
   * `README.md` bem documentado explicando o problema, classes escolhidas, estatísticas do dataset, métricas alcançadas e instruções para rodar o app.
   * Notebook(s) de treinamento e exportação limpos e documentados.
   * Arquivo do modelo treinado (ou link para download caso exceda o limite do Git).
   * Identificação de todos os integrantes do grupo com Nome e RM.
2. **Demonstração Prática (Presencial em Aula - 13/out):**
   * O grupo deverá demonstrar o aplicativo instalado no celular detectando os objetos/classes em tempo real.

---

## Rubrica de Avaliação (Total: 20 Pontos)

| Etapa | Critério | Pontuação |
| :--- | :--- | :---: |
| **Parte 1: Dataset & Treinamento** | **Dataset & Rotulagem:** Pelo menos 60 fotos com 3 classes anotadas com consistência e divisão correta (train/val/test). | **4,0 pts** |
| | **Treinamento & Métricas:** Pipeline de treino funcional, análise das métricas (mAP/Perdas) e exportação correta do modelo. | **6,0 pts** |
| **Parte 2: Deploy & Apresentação** | **Deploy Mobile:** Integração com o `yoloinferApp` funcional, demonstrando inferência fluida no smartphone. | **6,0 pts** |
| | **Organização & GitHub:** Repositório organizado, documentação completa e apresentação do grupo em aula. | **4,0 pts** |
| **TOTAL** | | **20,0 pts** |