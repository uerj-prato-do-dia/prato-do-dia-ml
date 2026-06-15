# Diagnostico do ML

Data da analise: 2026-06-15

Repositorio analisado: `/home/gabe/projects/prato-do-dia/prato-do-dia-ml`

## Resumo executivo

O repositorio `prato-do-dia-ml` esta em estagio de prototipo avancado / parcialmente produtivo para inferencia local, mas nao esta pronto para producao.

Existe um pipeline Python funcional com YOLO11 + SAM2 em ONNX Runtime CPU, testes, configs TOML, documentacao e metricas. A qualidade atual do modelo ainda e baixa e instavel: o relatorio versionado mostra `foreground_miou ~= 0.284` e `instance_miou ~= 0.316` em 8 imagens anotadas.

Validacoes executadas durante esta analise:

```bash
uv run pytest
uv run pytest -m onnx
```

Resultados:

- `uv run pytest`: `7 passed, 1 deselected`
- `uv run pytest -m onnx`: `1 passed, 7 deselected`

## Estrutura encontrada

Arquivos e pastas principais:

- `pyproject.toml`: projeto Python `>=3.11`, empacotado com Hatch e gerenciado com `uv`.
- `uv.lock`: lockfile de dependencias.
- `prato_do_dia_ml/`: pacote principal.
- `scripts/`: entrypoints CLI para rodar pipeline, avaliar e executar experimentos.
- `configs/`: configuracoes TOML do pipeline e experimentos.
- `docs/`: documentacao tecnica.
- `tests/`: testes automatizados.
- `data/input/`: imagens de entrada de estudo.
- `data/ground_truth/`: mascaras e anotacoes de referencia.
- `data/raw_segmentations/`: saidas YOLO TXT geradas.
- `data/masks/`: mascaras PNG geradas.
- `data/overlays/`: visualizacoes geradas.
- `data/reports/`: relatorios JSON gerados.
- `models/`: modelos ONNX locais.

Modulos principais:

- `prato_do_dia_ml/pipeline.py`: orquestra o fluxo completo de inferencia.
- `prato_do_dia_ml/detector.py`: wrapper YOLO11 ONNX.
- `prato_do_dia_ml/segmenter.py`: wrapper SAM2 ONNX.
- `prato_do_dia_ml/preprocessing.py`: normalizacao e letterbox.
- `prato_do_dia_ml/postprocessing.py`: limpeza de mascaras e resolucao de sobreposicoes.
- `prato_do_dia_ml/metrics.py`: metricas de segmentacao.
- `prato_do_dia_ml/feature_extraction.py`: extracao de atributos por instancia.
- `prato_do_dia_ml/experiment.py`: execucao e comparacao de experimentos.
- `prato_do_dia_ml/reproducibility.py`: registro de ambiente, modelos e configuracao.

Scripts encontrados:

- `scripts/run_pipeline.py`: executa inferencia em uma imagem.
- `scripts/evaluate_pipeline.py`: avalia predicoes contra ground truth.
- `scripts/run_experiment.py`: executa experimento reprodutivel.
- `scripts/extract_features.py`: extrai atributos de instancias segmentadas.
- `scripts/render_mask_previews.py`: renderiza previews de mascaras.

Configuracoes:

- `configs/default.toml`
- `configs/experiments/yolo11_sam2_baseline.toml`

Modelos locais encontrados:

- `models/yolov11_food.onnx` (~11 MB)
- `models/sam2.1_hiera_tiny.encoder.onnx` (~129 MB)
- `models/sam2.1_hiera_tiny.decoder.onnx` (~20 MB)

Nao foram encontrados:

- Dockerfile
- Makefile
- `requirements.txt`
- `environment.yml`
- servico HTTP proprio
- endpoint FastAPI/Flask/gRPC

## Pipeline identificado

O fluxo real implementado e:

1. Entrada: imagem local em `.jpg`, `.jpeg`, `.png` ou `.heic`.
2. Leitura: `load_image_bgr`, retornando imagem OpenCV BGR `uint8`.
3. Pre-processamento YOLO: letterbox para 640, conversao BGR -> RGB, normalizacao.
4. Deteccao: YOLO11 ONNX via `YoloOnnxDetector`.
5. Pre-processamento SAM2: letterbox para 1024 e normalizacao especifica do SAM.
6. Segmentacao: SAM2 ONNX usando caixas do YOLO como prompts.
7. Pos-processamento: limpeza de componentes pequenos, preenchimento de buracos e resolucao de overlaps.
8. Exportacao de artefatos:
   - TXT YOLO segmentation
   - PNG de instancias
   - PNG de classes
   - JSON de metadados
   - overlay JPG
9. Avaliacao: comparacao das mascaras preditas contra `data/ground_truth/*_instances.png`.

Entrypoint principal em codigo:

```python
FoodSegmentationPipeline.from_config(config).run_image(image_path)
```

## Estado de maturidade

Classificacao: parcialmente produtivo para inferencia local, prototipo para produto.

Justificativa:

- A inferencia local funciona com ONNX CPU.
- Ha testes unitarios e teste de inferencia real.
- Ha configuracao reprodutivel via TOML e `uv.lock`.
- Ha metricas quantitativas implementadas.
- Ha documentacao tecnica razoavel.
- O resultado atual do modelo ainda e fraco para uso em produto.
- Nao ha contrato estavel para API/mobile.
- Nao ha pipeline de treino ou fine-tuning proprio.
- Nao ha split formal de dataset.
- Nao ha criterio minimo de aceitacao do modelo.
- Nao ha deploy ou servico de inferencia.

## Entradas e saidas do modelo

Entradas aceitas:

- Imagem local `.jpg`, `.jpeg`, `.png` ou `.heic`.
- Internamente: OpenCV BGR `uint8`, formato `HxWx3`.

Configuracao padrao:

- YOLO:
  - modelo: `models/yolov11_food.onnx`
  - tamanho: `640`
  - confidence threshold: `0.15`
  - NMS IoU threshold: `0.45`
  - max detections: `30`
- SAM2:
  - encoder: `models/sam2.1_hiera_tiny.encoder.onnx`
  - decoder: `models/sam2.1_hiera_tiny.decoder.onnx`
  - tamanho: `1024`
  - mask threshold: `0.0`

Saidas geradas:

- `data/raw_segmentations/<imagem>.txt`: poligonos normalizados no formato YOLO segmentation.
- `data/masks/<imagem>_instances.png`: mascara de instancia single-channel.
- `data/masks/<imagem>_class.png`: mascara de classe/proposta.
- `data/reports/<imagem>.json`: metadados por imagem.
- `data/overlays/<imagem>_overlay.jpg`: visualizacao para inspecao.

Campos principais do JSON de metadados:

- `image`
- `width`
- `height`
- `model_versions`
- `instances`
- `instance_id`
- `proposal_class_id`
- `box_xyxy`
- `yolo_confidence`
- `sam_iou_prediction`
- `area_px`
- `polygon`

## Integracao com API/mobile

Nao ha integracao direta implementada neste repositorio.

O repositorio de ML hoje oferece:

- pacote Python importavel;
- scripts CLI;
- artefatos em disco.

Nao foram encontrados:

- endpoint HTTP proprio;
- servidor FastAPI/Flask;
- contrato JSON versionado para API;
- schema especifico para o app mobile consumir.

Forma provavel de integracao futura:

1. API recebe imagem enviada pelo mobile.
2. API chama o pipeline ML como biblioteca Python ou como servico separado.
3. ML retorna um objeto estruturado de predicao.
4. API converte esse resultado em contrato JSON estavel para o mobile.
5. Mobile consome apenas o JSON final e, se necessario, URLs de mascaras/overlays.

O app mobile nao deveria consumir diretamente os arquivos TXT/PNG gerados pelo pipeline experimental.

Possiveis problemas para o mobile:

- `proposal_class_id` nao representa identificacao final de alimento.
- Ainda nao existe nome do alimento, porcao ou informacao nutricional.
- Poligonos podem ser grandes para payload mobile.
- Mascaras PNG sao artefatos pesados para consumo direto.
- Nao ha contrato para erro, imagem invalida, prato ausente ou zero deteccoes.
- Nao ha versao de schema de resposta.

## Lacunas e problemas

### Criticos

- Qualidade atual insuficiente para produto: `instance_miou ~= 0.316`.
- Dataset pequeno e sem split formal.
- Ausencia de contrato de integracao API/mobile.
- Ausencia de pipeline proprio de treino/fine-tuning.

### Importantes

- Modelos ONNX dependem de arquivos locais em `models/`.
- Nao ha hash/checksum dos modelos.
- README referencia scripts que nao existem fisicamente:
  - `scripts/compare_experiments.py`
  - `scripts/import_labelstudio_brush.py`
- A funcao `compare_experiments` existe em `prato_do_dia_ml/experiment.py`, mas nao ha script CLI correspondente.
- Ha metricas, mas nao ha criterio minimo de aceite.
- Nao ha benchmark formal de latencia/memoria no ambiente alvo.
- Nao ha contrato de estabilidade para o JSON de metadados.

### Menores

- Nao ha Dockerfile.
- Nao ha Makefile.
- Saidas derivadas ficam dentro de `data/`, misturando fonte e artefatos.
- `pyproject.toml` aparece modificado localmente, aparentemente apenas por movimentacao do bloco de configuracao do pytest.

## Testes, metricas e reprodutibilidade

O que existe:

- `uv.lock` para lock de dependencias.
- CI em GitHub Actions.
- Ruff para formatacao e lint.
- Pytest com marcador `onnx`.
- Testes unitarios para preprocessamento, metricas, pos-processamento e interfaces.
- Teste real de inferencia ONNX.
- Registro de ambiente em experimentos.
- Registro de configuracao e modelos nos experimentos.
- Metricas por instancia:
  - IoU
  - Dice
  - boundary F-score
  - precision
  - recall
  - falsos positivos
  - instancias perdidas
  - erro de area

O que falta:

- Split formal de dataset.
- Baseline congelado com criterio de aprovacao.
- Teste de contrato para resposta da API.
- Teste de latencia e memoria com limites definidos.
- Hash/checksum dos pesos ONNX.
- Documentacao clara de deploy.
- Estrategia para versionar dataset e modelos.

## Riscos tecnicos

### Criticos

- Modelo ainda nao entrega segmentacao confiavel para produto.
- API e mobile ainda nao tem contrato de consumo definido.
- Dataset atual parece pequeno e potencialmente pouco representativo.

### Importantes

- Reprodutibilidade dos pesos depende de arquivos locais.
- Metricas baixas indicam risco de falhas graves em alimentos pequenos, multiplas instancias ou pratos complexos.
- Falta validacao robusta para entradas reais de usuario.
- Falta estrategia de deploy e concorrencia alem de ONNX CPU single-thread.
- Falta criterio objetivo para decidir quando integrar ao produto.

### Menores

- Documentacao parcialmente desalinhada com scripts existentes.
- Artefatos derivados versionados ou mantidos em `data/` podem gerar confusao operacional.
- Sem container ou receita padronizada de runtime fora do `uv`.

## Plano recomendado

### Fase 1: tornar executavel e reprodutivel

Objetivo: garantir que outra pessoa consiga rodar e reproduzir o estado atual.

Acoes:

1. Padronizar setup:

   ```bash
   uv sync --locked
   ```

2. Manter comandos minimos de validacao:

   ```bash
   uv run ruff format --check .
   uv run ruff check .
   uv run pytest
   uv run pytest -m onnx
   ```

3. Corrigir README/docs para refletir scripts reais.
4. Criar manifest do dataset com:
   - imagem;
   - origem;
   - anotacao correspondente;
   - split;
   - observacoes de qualidade.
5. Separar claramente:
   - dados fonte;
   - anotacoes;
   - artefatos derivados;
   - outputs de experimentos.
6. Adicionar checksum dos modelos ONNX.
7. Documentar como obter exatamente os pesos usados.

### Fase 2: estabilizar inferencia

Objetivo: transformar o pipeline em componente consumivel pela API.

Acoes:

1. Criar interface publica limpa, por exemplo:

   ```python
   predict(image_bytes: bytes) -> PredictionResponse
   ```

2. Definir schema JSON versionado.
3. Definir campos obrigatorios para API/mobile:
   - `schema_version`
   - `image_width`
   - `image_height`
   - `instances`
   - `instance_id`
   - `polygon`
   - `bbox`
   - `confidence`
   - `area_px`
   - `warnings`
4. Definir comportamento para:
   - zero deteccoes;
   - imagem invalida;
   - modelo ausente;
   - erro interno;
   - imagem com dimensao muito grande;
   - formato nao suportado.
5. Decidir arquitetura:
   - ML como biblioteca chamada pela API; ou
   - ML como servico separado.
6. Criar testes de contrato.
7. Medir latencia e memoria no ambiente alvo.

### Fase 3: melhorar qualidade do modelo

Objetivo: sair de prototipo para modelo avaliavel e integravel.

Acoes:

1. Congelar baseline atual com:
   - configuracao;
   - modelos;
   - dataset;
   - metricas;
   - overlays.
2. Aumentar e qualificar o dataset anotado.
3. Definir split treino/validacao/teste.
4. Definir metricas-alvo minimas.
5. Avaliar piores casos antes de mudar o app.
6. Experimentar fine-tuning do detector YOLO ou substituicao do detector generico.
7. Versionar experimentos e modelos.
8. Criar relatorio comparativo entre baselines.

## Proximas acoes sugeridas

1. Corrigir README/docs para remover ou criar referencias a scripts inexistentes.
2. Criar contrato JSON de inferencia para API/mobile.
3. Criar manifest do dataset e split formal.
4. Registrar hash dos modelos ONNX.
5. Rodar novo experimento baseline em `outputs/experiments`.
6. Definir criterio minimo de qualidade antes de integrar ao mobile.
7. Implementar camada fina de inferencia voltada para a API.
8. Integrar com a API somente depois que entrada, saida, erros e versao de schema estiverem estaveis.

