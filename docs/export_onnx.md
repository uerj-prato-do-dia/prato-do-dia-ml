# Obtenção dos Modelos ONNX

O pipeline executa inferência local em CPU usando ONNX Runtime. Por isso, os
pesos devem estar em formato ONNX antes de serem colocados em `models/`.

Arquivos esperados:

```text
models/yolov11_food.onnx
models/sam2.1_hiera_tiny.encoder.onnx
models/sam2.1_hiera_tiny.decoder.onnx
```

Os arquivos de modelo são grandes e não entram no Git.

## YOLO11

Para experimentos iniciais, um modelo YOLO11 pequeno é suficiente para validar
o fluxo detector -> segmentador. A exportação pode ser feita em um ambiente com
Ultralytics instalado:

```bash
uv sync --group train
```

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
path = model.export(format="onnx", opset=14, dynamic=False)
print(path)
```

Renomeie o arquivo exportado para:

```text
models/yolov11_food.onnx
```

Em etapas posteriores, esse detector deve ser substituído ou ajustado com dados
mais próximos do domínio alimentar estudado.

## SAM 2.1

O SAM 2 é usado em dois blocos ONNX: encoder de imagem e decoder de prompt.
Essa divisão evita recomputar a representação da imagem para cada caixa
detectada.

Instale o grupo de dependências para aquisição de modelos:

```bash
uv sync --group models
```

Exemplo de download de pesos ONNX pré-exportados:

```python
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo = "vietanhdev/segment-anything-2-onnx-models"
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

encoder = hf_hub_download(repo_id=repo, filename="sam2_hiera_tiny.encoder.onnx")
decoder = hf_hub_download(repo_id=repo, filename="sam2_hiera_tiny.decoder.onnx")

shutil.copy(encoder, models_dir / "sam2.1_hiera_tiny.encoder.onnx")
shutil.copy(decoder, models_dir / "sam2.1_hiera_tiny.decoder.onnx")
```

## Verificação

Depois de posicionar os arquivos em `models/`, rode:

```bash
uv run pytest -m onnx
```

Ou execute uma imagem de exemplo:

```bash
uv run python scripts/run_pipeline.py data/input/imagem1.jpg --confidence 0.05 --max-detections 3
```

Se a entrada ou saída esperada do modelo mudar, atualize também
`configs/default.toml` e os testes de interface ONNX.
