# Veritas — Classificação de Imagens (repositório)

Este repositório contém utilitários, modelos e scripts para treino e inferência
de classificadores de imagem usados no projeto *Veritas*.

## Estrutura principal
- `v3/` — pacote principal com helpers, dataset, treino e predição.
- `models/` — modelos serializados (`.pkl`, `.joblib`).
- `dataset/`, `faces/`, `diffusion/`, `outputs/` — pastas de dados e saídas.
- `pipeline.py` — exemplo de pipeline de treino/avaliação usado por desenvolvedores.

## Requisitos
Recomenda-se usar um ambiente virtual Python (venv/virtualenv).

Exemplo rápido:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r v3/requirements.txt
```

OBS: o `requirements.txt` dentro de `v3/` lista dependências úteis como `torch`, `timm`, `torchvision`, `joblib`, `Pillow`.

## Uso rápido
Importe o pacote principal com:

```python
import v3 as pedrita
```

O pacote `v3` reexporta funções e utilitários úteis para treino e predição, permitindo chamá-las via `pedrita.<nome>`.

### Dispositivo (CPU/GPU/MPS)
Use `best_device()` para detectar o melhor dispositivo disponível e gravá-lo internamente:

```python
pedrita.best_device()        # detecta CUDA, MPS, DirectML ou CPU
pedrita.best_device(True)    # força o uso de CPU
```

Detalhes: `best_device(cpu=True)` chama `helper.best_device(cpu=True)` e define `force_cpu=True`, fazendo com que modelos e tensores sejam movidos para `torch.device('cpu')`.

### Carregar / definir modelo
Use `set_model()` para carregar um modelo global usado por outras rotinas:

```python
# carrega um objeto serializado (joblib) ou instancia um modelo se `force=True`
pedrita.set_model('models/ultra_7.pkl')

# também aceita uma instância de torch.nn.Module diretamente
model = MyModel(...)
pedrita.set_model(model)
```

Comportamento:
- Se o argumento for `None`, `set_model()` retorna o modelo atual.
- Se for uma instância de `torch.nn.Module`, o modelo é movido para o dispositivo detectado e colocado em `eval()`.
- Se for um caminho para arquivo, a função tenta `joblib.load(path)`. Se o objeto carregado for um `Module`, ele é usado.
- Se o arquivo não existir e `force=True`, o código pode criar um backbone via `timm.create_model(..., num_classes=...)` e salvar um `.pkl` auxiliar.

### Datasets
Use `pedrita.DirDataset(path, limit=...)` para criar datasets a partir de pastas com imagens organizadas por classe.

Exemplo:

```python
ds = pedrita.DirDataset('./dataset/train', limit=3000)
```

### Treino e avaliação
As funções de treino e predição são expostas pelo pacote. Exemplos:

```python
pedrita.best_device()
pedrita.set_model('models/ultra_7.pkl')
ds = pedrita.DirDataset('./dataset/train', limit=3000)
pedrita.train(ds, epochs=3, lr=1e-4)

# avaliar uma pasta inteira (usa predict.evaluate_folder)
acc = pedrita.evaluate_folder('./diffusion/', limit=500)
print('ACCURACY', acc)
```

Veja `pipeline.py` para um exemplo de uso automatizado que alterna treino/avaliação e salva modelos.

## Exemplo prático (treino + checagem)
Trecho simplificado (baseado em `pipeline.py`):

```python
import v3 as pedrita

pedrita.best_device()
pedrita.set_model('models/ultra_7.pkl')

ds = pedrita.DirDataset('./dataset/train', limit=3000)
pedrita.train(ds, epochs=3, lr=5e-5, check=lambda: print('check called'))
pedrita.save_model('models/ultra_8.pkl')
```

## Onde estão as implementações
- `v3/helper.py` — funções centrais (`set_model`, `best_device`, transforms, utilitários de I/O).
- `v3/predict.py` — rotinas de predição/avaliação e helpers para gerar heatmaps/visualizações.
- `v3/train.py` — loop de treino e hooks.
- `v3/dset.py` — definição de `DirDataset` e outros datasets.

Links rápidos (arquivo no repo):
- [v3/helper.py](v3/helper.py)
- [v3/predict.py](v3/predict.py)
- [v3/train.py](v3/train.py)
- [v3/dset.py](v3/dset.py)

## Contribuição
- Abra issues para bugs ou melhorias.
- Faça PRs pequenas e focadas; adicione testes quando possível.

## Licença
Coloque aqui a licença do projeto, se aplicável.
