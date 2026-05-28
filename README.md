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

## Introdução
DISCLAIMER: Devido à participação do grupo na trilha de Innovation, as atividades deste projeto encontram-se distanciadas da disciplina tradicional de Projetos e de seus frameworks usuais de requisitos. Por isso, os parâmetros de modelagem, a seleção de variáveis e os requisitos de visualização gráfica apresentados a seguir foram estritamente personalizados e direcionados para atender às demandas
técnicas e científicas específicas da cadeira de Machine Learning I (ML). O foco das análises reside na investigação forense digital, integridade de mídia e comportamento estatístico dos pixels, priorizando a coerência metodológica necessária para o desenvolvimento do modelo preditivo do Veritas.

## Breve descrição da solução  
Veritas (referenciada como Pedrita no repositório) é um projeto baseado em visão computacional voltada para a detecção de imagens manipuladas ou geradas por IA, centralizando utilitários, modelos e pipelines para treino e inferência de classificadores.
O pacote principal, localizado em v3/, expõe uma API coesa com três responsabilidades principais:

Treino — helpers para configuração de experimentos, data augmentation e ciclos de otimização de classificadores reais vs. sintéticos  
Predição — interfaces de inferência para identificar se uma imagem é autêntica ou de origem artificial  
Avaliação — métricas e ferramentas para análise de desempenho e robustez dos modelos    

## Contextualização
**Problema abordado**
O projeto Veritas enfrenta a crescente incapacidade de verificar, em escala e com agilidade, a autenticidade de imagens que circulam em fluxos intensos de informação digital. A proliferação de ferramentas de geração de imagens por inteligência artificial, como modelos de difusão latente e redes adversariais generativas (GANs), democratizou a produção de conteúdo visual sintético de altíssimo realismo, tornando a distinção entre imagens autênticas e fabricadas um desafio técnico sem precedentes. No âmbito político e institucional, esse cenário afeta diretamente campanhas eleitorais, coberturas de conflito armado e crises sanitárias, contextos nos quais a
desinformação visual causa danos concretos e irreversíveis, corroendo a confiança pública nas instituições de imprensa e fragilizando os pilares da democracia deliberativa. No entanto, o impacto das imagens sintéticas não se restringe ao campo jornalístico ou político. No ambiente corporativo, imagens manipuladas têm sido utilizadas como instrumento de ataques à reputação de empresas e executivos, com a criação de supostas evidências visuais de escândalos, comportamentos inapropriados ou situações comprometedoras que nunca ocorreram. Tais conteúdos, uma vez viralizados, podem provocar quedas abruptas no valor de mercado de companhias,
demissões injustificadas, rupturas contratuais e processos judiciais de alta complexidade, mesmo quando posteriormente desmentidos. Além disso, imagens geradas por IA têm sido exploradas como vetor de golpes e
fraudes financeiras de grande sofisticação. Entre os padrões mais recorrentes estão: a falsificação de documentos e registros fotográficos para suporte a solicitações de crédito ou processos de seguro; a criação de identidades visuais falsas para aplicações de romance scam e burlas em plataformas de namoro; a simulação de eventos corporativos ou declarações públicas forjadas para manipular preços de
ativos financeiros; e o uso de rostos sintéticos para burlar sistemas de verificação de identidade (KYC) em instituições bancárias e fintechs. Em todas essas dimensões, política, jornalística, corporativa e financeira, o
denominador comum é a ausência de mecanismos automáticos, escaláveis e interpretáveis para triagem de conteúdo visual sintético. É precisamente essa lacuna que o Veritas se propõe a preencher.

**Objetivo da solução**
O objetivo central do Veritas é desenvolver um sistema inteligente de classificação binária capaz de determinar, a partir da análise de características latentes, padrões estatísticos e metadados de arquivos de imagem, se uma mídia visual é autêntica ou manipulada/sintética. O produto final visa fornecer a jornalistas, agências de checagem e investigadores digitais uma camada automatizada de triagem pericial, onde interpretabilidade, velocidade de inferência e robustez são requisitos tão críticos quanto a acurácia.

**Domínio da aplicação**  
O Veritas se posiciona na interseção entre jornalismo investigativo, perícia digital e aprendizado de máquina. O domínio de aplicação abrange ambientes reais de checagem de fatos, monitoramento de mídia e análise forense de conteúdo visual. O classificador opera sobre imagens estáticas nos formatos PNG e JPG, explorando artefatos e inconsistências deixados por processos de geração ou edição artificial, desde descontinuidades de frequência detectáveis via análise espectral até anomalias em canais de ruído, incoerências de iluminação e rastros forenses em metadados EXIF. O escopo futuro inclui a atribuição de proveniência gerativa, identificando a arquitetura de IA responsável pela geração da imagem.

**Importância da análise exploratória para o projeto**  
A Análise Exploratória de Dados (EDA) é etapa fundamental para o Veritas, pois permite compreender a distribuição das classes (imagens reais vs. sintéticas), identificar desequilíbrios que possam enviesar o treinamento, mapear padrões visuais e estatísticos que diferenciam as duas categorias, e orientar as decisões de pré-processamento e engenharia de features. Dado que o dataset conta com aproximadamente 100.000 registros de treino e 213 atributos, a EDA também é essencial para detectar outliers, valores ausentes e redundâncias que poderiam comprometer a qualidade e a generalização dos modelos treinados.

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

## Links rápidos:
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
DATASET: https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k  
DATASET: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images  
DATASET: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces  
DRIVE: [https://drive.google.com/drive/folders/1ywYBYQ1KoG0kaChqxis3LcpLXM4J5CHt?usp=sharing](https://drive.google.com/drive/folders/1ywYBYQ1KoG0kaChqxis3LcpLXM4J5CHt?usp=sharing)    
APLICAÇÃO: https://app-veritas.netlify.app  
MIRO: https://miro.com/app/board/uXjVJQuHnVQ=/?share_link_id=526481362291  
GOOGLE SITES:https://sites.google.com/view/projetoveritas/nosso-prop%C3%B3sito    
FIGMA:https://www.figma.com/design/DffaA6K2I4fEAXVPLmYwG0/Sem-t%C3%ADtulo?node-id=0-1&t=6W5Z6Trawm0E4F0M-1     
