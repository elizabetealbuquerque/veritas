# Veritas — Classificação de Imagens autênticas ou manipuladas (repositório)

## Informações Gerais  
Este repositório contém utilitários, modelos e scripts para treino e inferência
de classificadores de imagem usados no projeto *Veritas*.

**Grupo:**  

  Ivan Edward @ iers-bd   
  Elizabete Barbosa @ elizabetealbuquerque   
  Joao Rietra @ jhlr  
  
**Nome da disciplina:** Machine Learning I - School Innovation.   
**Nome da instituição de ensino:** CESAR School.  
**Nome da solução desenvolvida:** Veritas    

## Breve descrição da solução  
Veritas (referenciada como Pedrita no repositório) é um projeto baseado em visão computacional voltada para a detecção de imagens manipuladas ou geradas por IA, centralizando utilitários, modelos e pipelines para treino e inferência de classificadores.
O pacote principal, localizado em v3/, expõe uma API coesa com três responsabilidades principais:

Treino — helpers para configuração de experimentos, data augmentation e ciclos de otimização de classificadores reais vs. sintéticos  
Predição — interfaces de inferência para identificar se uma imagem é autêntica ou de origem artificial  
Avaliação — métricas e ferramentas para análise de desempenho e robustez dos modelos    

## Estrutura principal  

- `v3/` — pacote principal com helpers, dataset, treino e predição.  
- `models/` — modelos serializados (`.pkl`, `.joblib`).  
- `dataset/`, `faces/`, `diffusion/`, `outputs/` — pastas de dados e saídas.  
- `pipeline.py` — exemplo de pipeline de treino/avaliação usado por desenvolvedores.

## Requisitos  

Recomenda-se usar um ambiente virtual Python (venv/virtualenv).

**Exemplo rápido:**  

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r v3/requirements.txt
```

OBS: o `requirements.txt` dentro de `v3/` lista dependências úteis como `torch`, `timm`, `torchvision`, `joblib`, `Pillow`.

## Instruções detalhadas para compilar e executar.  

O pacote `v3` fornece um ponto de entrada via `__main__` para uso por linha de comando. Há duas formas comuns de usar o repositório:

**1) Usando o pacote como módulo Python (recomendado):**

```bash
# sintaxe geral
python v3 test <model_path> -i <pasta> -l <quantidade>

# exemplo prático: avalia pastas de imagens usando um modelo salvo
python v3 test models/ultra_7.pkl -i faces -l 10000
```

Parâmetros principais:  

- `<model_path>`: caminho para o arquivo do modelo serializado (por exemplo `models/Pedrita_v3.joblib`).  
- `test`: subcomando para avaliação (outros subcomandos: `train`, `video`).  
- `-i` / `--image`: pasta com imagens para avaliar (pasta com subpastas por classe).  
- `-l` / `--limit`: limite do número de amostras a usar (opcional).  

**2) Usando o pacote dentro de um script Python ou REPL:**  

```python
import v3 as pedrita
pedrita.best_device()
pedrita.set_model('models/ultra_7.pkl')
ds = pedrita.DirDataset('./dataset/train', limit=3000)
pedrita.train(ds, epochs=3, lr=5e-5)
```

## Exemplos adicionais de CLI

**- Treinar com uma pasta organizada:**  

```bash
python v3 train models/ultra_7.pkl -i /caminho/para/treino --epochs 3 -l 1000
```  

**- Executar predição em vídeo:**  

```bash
python v3 video models/ultra_7.pkl -v caminho/para/video.mp4 -n 30
```  

## Onde estão as implementações

- `v3/helper.py` — funções centrais (`set_model`, `best_device`, transforms, utilitários de I/O).
- `v3/predict.py` — rotinas de predição/avaliação e helpers para gerar heatmaps/visualizações.
- `v3/train.py` — loop de treino e hooks.
- `v3/dset.py` — definição de `DirDataset` e outros datasets.

## Links rápidos (arquivo no repo):
- [v3/helper.py](v3/helper.py)
- [v3/predict.py](v3/predict.py)
- [v3/train.py](v3/train.py)
- [v3/dset.py](v3/dset.py)

## Contribuição  

- Abra issues para bugs ou melhorias.  
- Faça PRs pequenas e focadas; adicione testes quando possível.  

## Licença

Coloque aqui a licença do projeto, se aplicável.  

## Links importantes para o projeto.  
DATSET: https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k  
DATASET: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images  
DATASET: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces  
DRIVE: https://drive.google.com/drive/folders/1ywYBYQ1KoG0kaChqxis3LcpLXM4J5CHt?usp=sharing  
APLICAÇÃO: https://app-veritas.netlify.app  
MIRO: https://miro.com/app/board/uXjVJQuHnVQ=/?share_link_id=526481362291  
GOOGLE SITES:https://sites.google.com/view/projetoveritas/nosso-prop%C3%B3sito    
FIGMA ATUALIZADO:https://www.figma.com/design/DffaA6K2I4fEAXVPLmYwG0/Sem-t%C3%ADtulo?node-id=0-1&t=6W5Z6Trawm0E4F0M-1     
