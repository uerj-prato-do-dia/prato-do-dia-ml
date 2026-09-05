# Guia Definitivo de Rotulagem de Imagens com X-AnyLabeling

**Projeto:** Prato do Dia (ML Pipeline)  
**Versão do Documento:** 2.0 (Canônica - Anti Train-Serving Skew)  
**Diretório Alvo do Dataset:** `data/processed_640/`  
**Modelo de Predição:** YOLOv11-seg (16 classes fixas, ID `0` a `15`)

---

## ⚠️ Princípios Inegociáveis da Arquitetura de Visão Computacional

1. **Nunca Rotular Fotos Brutas (`data/raw_images`):** Rotular imagens brutas antes do recorte canônico causa **corrupção de coordenadas normalizadas** ao recortar a imagem posteriormente, além de reintroduzir o *Train-Serving Skew* (o modelo aprende proporções que a API de produção nunca envia). A anotação deve ser feita **estritamente sobre as imagens processadas em `data/processed_640/`**.
2. **Salvamento Nativo em JSON + Daemon Douglas-Peucker:** O SAM 2 gera polígonos ultra-densos (300 a 1.000 vértices por objeto). Salve no X-AnyLabeling em formato **JSON nativo** e deixe o daemon de segundo plano (`watch_and_simplify.py`) simplificar automaticamente os polígonos para 15 a 40 vértices no padrão YOLO `.txt`.
3. **Alinhamento do Manifesto:** O manifesto de treino `data/dataset.yaml` deve ser gerado apontando para `data/processed_640/`.

---

## 🛠️ O Fluxo Linear e Seguro de Rotulagem (4 Passos)

### Passo 1: Ingestão e Padronização Canônica
Pegue as fotos brutas recebidas do celular (em `data/raw_images/` ou `data/raw_segmentations/`) e execute o recorte quadrado de 85% do menor lado com ajuste EXIF e letterbox $640 \times 640$:

```bash
cd /home/gabe/projects/prato-do-dia
python3 prato-do-dia-ml/scripts/crop_manual.py
```
*(As imagens recortadas a $640 \times 640$ serão gravadas automaticamente em `data/processed_640/`)*.

---

### Passo 2: Iniciar o Daemon de Simplificação em Segundo Plano
Em um terminal separado, inicie o daemon observador em tempo real:

```bash
cd /home/gabe/projects/prato-do-dia
python3 prato-do-dia-ml/scripts/watch_and_simplify.py
```
*(Ele monitora `data/processed_640/`, aguarda os salvamentos em `.json` do X-AnyLabeling, aplica o filtro Douglas-Peucker e grava o arquivo `.txt` no padrão YOLO de 16 classes em milissegundos)*.

---

### Passo 3: Rotulagem no X-AnyLabeling
Abra o **X-AnyLabeling** e configure as opções:

* **Pasta de Imagens (Open Dir):** `/home/gabe/projects/prato-do-dia/data/processed_640/`
* **Taxonomia (Custom Labels):** Selecione `/home/gabe/projects/prato-do-dia/data/processed_640/classes.txt`
* **Modelo Auxiliar (Auto Labeling):** Selecione **Segment Anything 2.1 (Tiny)**.
* **Modo de Salvamento:** Salve normalmente no formato **JSON nativo** (`Ctrl + S` ou pressione `D` para avançar). O daemon do Passo 2 cuidará da geração do `.txt` YOLO simplificado em tempo real.

---

### Passo 4: Auditoria de Integridade e Geração do Manifesto
Ao concluir a sessão de rotulagem, valide o ground truth e gere o manifesto com split anti-leakage (70% train, 20% val, 10% test):

```bash
cd /home/gabe/projects/prato-do-dia
python3 prato-do-dia-ml/scripts/audit_ground_truth.py --labels-dir data/processed_640
python3 prato-do-dia-ml/scripts/create_dataset_manifest.py --input-dir data/processed_640 --output-dir data
```

---

## 📊 Taxonomia das 16 Classes Canônicas (ID 0 a 15)

| ID | Nome da Classe | ID | Nome da Classe |
| :---: | :--- | :---: | :--- |
| `0` | `tomate` | `8` | `cenoura` |
| `1` | `salada_verde` | `9` | `ovo_frito` |
| `2` | `feijao` | `10` | `massa_macarrao` |
| `3` | `batata_frita` | `11` | `frango_grelhado` |
| `4` | `arroz` | `12` | `azeitona` |
| `5` | `carne_moida` | `13` | `batata_palha` |
| `6` | `pure_batata` | `14` | `estrogonofe` |
| `7` | `farofa` | `15` | `carne_bovina_bife` |

---

## 🎨 Validação Visual
Para renderizar overlays das máscaras e inspecionar visualmente se a simplificação preservou os contornos das refeições:

```bash
python3 prato-do-dia-ml/scripts/render_mask_previews.py
```
