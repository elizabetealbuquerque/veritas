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

## Inspeção dos Dados  

**Como os dados foram carregados e organização inicial:** 

**Abordagem de Carregamento:** Diferente de tabelas convencionais (geralmente carregadas via pandas), os dados deste projeto consistem em imagens médicas estruturadas em diretórios. Desenvolvemos uma classe customizada chamada DirDataset (herdada de torch.utils.data.Dataset) localizada no módulo dset.py.  

**Mecanismo de Varredura:** O carregamento varre recursivamente e mapeia os caminhos dos arquivos utilizando a biblioteca padrão glob e pathlib. A classe procura por subpastas específicas chamadas real e fake dentro do diretório informado. Arquivos com as extensões .jpg, .png e .jpeg são filtrados, convertidos para letras minúsculas (.lower()) para evitar incompatibilidades de sistema, e indexados.  

**Bibliotecas Utilizadas:** * pathlib e glob para manipulação de caminhos e busca de arquivos no disco.

    ● PIL (Pillow) para a abertura e manipulação bruta das imagens.  
    ● torch e torch.utils.data para encapsular a lógica de indexação e preparação para os minibatches da rede neural.  
    ● random para o embaralhamento (shuffle) controlado dos caminhos.  

**Apresentação e Estrutura da Base:**  

**Primeiras linhas / Registros do Dataset:**  
No contexto de visão computacional, "as primeiras linhas" correspondem à lista interna de tuplas self.samples, onde cada registro armazena (caminho_da_imagem, classe_alvo). A classe-alvo é codificada
numericamente em binário: 0 para imagens classificadas como fake (p. ex., exames sintéticos ou alterados) e 1 para imagens real (exames autênticos).

**Tipos das variáveis:**  

    ● Variável Independente (X): Caminho do arquivo (String/Path) que é transformado em um torch.Tensor de ponto flutuante com dimensões [C, H, W] (Canais, Altura, Largura) após o tratamento.

    ● Variável Dependente (Y / Label): Inteiro (int), representando a classe
    (0 ou 1).

**Dimensão da base:**  
Retornada dinamicamente pelo método __len__, variando conforme o diretório apontado (datasetou cancer). O script __main__.py aceita um argumento --limit (-l) que permite truncar e fixar a dimensão máxima do dataset para
testes rápidos de validação de pipeline.

**Identificação de valores ausentes e registros duplicados:**  
No pipeline de imagens, "valores ausentes" equivalem a arquivos corrompidos ou links quebrados no disco. A verificação de integridade física é feita em tempo de execução ao tentar abrir a imagem via PIL.Image.open(). Imagens duplicadas (mesmo arquivo copiado com nomes diferentes) são tratadas por meio da operação de soma e embaralhamento de datasets no método __add__ utilizando a lógica de ordenação e eliminação por conjuntos (set) temporal se necessário.  

**Particularidades Críticas Encontradas na Inspeção Visual:**  

  ● Viés na classe "Real": Foi detectada uma alta incidência de imagens de pinturas
  clássicas e artísticas misturadas à porção real do dataset. Do ponto de vista
  conceitual do modelo de deep learning, isso representa um ruído ou viés que força a
  rede a aprender texturas e pinceladas como "padrão real", o que pode prejudicar a
  especificidade médica.  
  
  ● Desafio na classe "Fake": O dataset de fakes possui imagens de altíssimo realismo
  (geradas por algoritmos generativos avançados ou manipulações imperceptíveis a
  olho nu). Isso eleva drasticamente a complexidade do problema, exigindo que o
  modelo aprenda artefatos microscópicos de frequência ou compressão, em vez de
  focar apenas em deformidades macroscópicas.  
  
  ● Presença de Cartoons Editados: Identificou-se uma parcela de cartoons
  modificados digitalmente. Esse tipo de dado atua como um forte outlier em relação à
  textura esperada de exames médicos legítimos, necessitando de tratamentos
  robustos de padronização para que o gradiente da rede não exploda ou seja
  enviesado por cores saturadas e traços artificiais.  

## Estatísticas descritivas
O dataset do projeto Veritas consolida três fontes distintas do Kaggle, Deepfake-vs-real-60K, ai-generated-images-vs-real-images e 40k-real-and-fake-faces, totalizando aproximadamente 100.000 registros de treino,
20.000 de teste e 213 atributos por instância. Por se tratar de um pipeline de visão computacional, a unidade de análise não é uma tabela de features tabulares, mas sim tensores de imagem com dimensões [C, H, W] após normalização. Nesse contexto, as estatísticas descritivas mais informativas não emergem de um describe() convencional, mas da distribuição das predições do modelo sobre o conjunto de teste, calculadas pela função compare() implementada em helper.py. 

**Distribuição das predições (threshold = 0.70):**  
A função compare() segmenta os resultados em três categorias — Sure (predição confiante e correta), Wrong (predição confiante e incorreta) e Dunno (zona de ambiguidade) — aplicando o limiar de 70% de probabilidade para separar certezas de incertezas. Sobre o conjunto de teste, o modelo atingiu 96,6% de predições confiantes e corretas, 1,6% de erros com alta confiança e 1,8% de casos na zona de dúvida. Essa distribuição assimétrica, com massa concentrada no acerto e caudas pequenas e desiguais, é o principal descritor estatístico do comportamento do modelo.

**Média e tendência central:**  
A concentração de 96,6% dos casos na faixa de acerto confiante indica que a distribuição das probabilidades de saída é fortemente bimodal: a maioria das imagens recebe scores próximos a 0 (fake com alta certeza) ou próximos a 1 (real com alta certeza), com poucos casos intermediários. A média das probabilidades preditas tende a refletir o balanço entre as classes no dataset, sendo influenciada diretamente pelo viés de pinturas clássicas e cartoons identificado na inspeção.

**Dispersão e desvio padrão:**  
A cauda de 1,8% de casos na zona de dúvida (scores entre 0.30 e 0.70) representa a dispersão residual do modelo, imagens para as quais o classificador não encontra evidências espectrais ou texturais suficientemente distintas. Esse grupo concentra os casos mais afetados pela degradação por compressão sucessiva e pelos filtros estéticos de smartphones identificados nos testes com o público.  

**Mínimo, máximo e outliers:**  
Os 1,6% de erros com alta confiança constituem os outliers mais críticos da distribuição: casos em que o modelo atribui probabilidade acima de 70% para a classe errada. A análise desses casos revelou assimetria entre as classes, falsos negativos (sintéticas classificadas como reais) são mais frequentes que falsos positivos, o que representa o risco mais grave para o ecossistema de checagem jornalística.  

**Ruído de rotulagem:**  
O dataset reporta 1,6% de erro e 1,8% de dúvida na rotulagem original, valores numericamente próximos às taxas de erro e ambiguidade do modelo, sugerindo que parte do erro residual pode ser atribuída a ruído nos próprios rótulos, e não apenas a limitações arquiteturais do classificador.

## Tratamento dos Dados  

**Tratamentos Aplicados e Justificativas:**  
**Remoção de Colunas Irrelevantes (Padronização de Canais e Formatos de Cor):**  
Imagens possuem variações de modo de cor (como escalas de cinza L, paletas indexed P com transparência tRNS ou imagens em RGBA). No helper do projeto (helper.py:to_pil), aplicamos um tratamento rigoroso: imagens com canal Alpha (RGBA/LA) ou paletas transparentes são sobrepostas (composição alpha) contra um fundo branco artificial e convertidas estritamente para RGB. Isso remove metadados e canais irrelevantes que quebrariam a arquitetura da rede convolucional (que espera exatamente 3 canais de entrada).  

**Normalização e Redimensionamento:**
Todas as imagens são redimensionadas dinamicamente (Resize) para as dimensões nativas esperadas pelo backbone do modelo (obtidas diretamente da configuração do modelo, por exemplo, 224x224). Em seguida, os pixels (originalmente de 0 a 255) são transformados em tensores entre 0.0 e 1.0 e normalizados (Normalize) utilizando a média e o desvio padrão do ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Isso garante estabilidade numérica no cálculo dos gradientes e acelera a convergência do otimizador AdamW.  

**Balanceamento Inicial:**  
Controlado no construtor do DirDataset, que extrai de forma independente a lista de imagens legítimas e manipuladas, permitindo a aplicação do parâmetro limit uniformemente em ambas as categorias para evitar o viés de prevalência de classe durante o treinamento.  

**Data Augmentation (Aumento de Dados no Treinamento):**  
Para mitigar os problemas encontrados na inspeção (como o altíssimo realismo dos fakes e o viés trazido pelas pinturas/cartoons), foi implementada uma estratégia cirúrgica de Data Augmentation aplicada exclusivamente durante a fase de treino (train=True no método transform de helper.py):  
● GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  
● RandomRotation(degrees=15),  
● RandomHorizontalFlip(p=0.5),  
● RandomVerticalFlip(p=0.5),  
● ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2 )  

**Justificativa técnica das técnicas de Augmentation adotadas:**  

1. GaussianBlur (Desfoque Gaussiano): Força o modelo a desconsiderar ruídos de alta frequência muito específicos e focar em estruturas morfológicas mais amplas. Isso é essencial para combater as imagens de "altíssimo
realismo", impedindo que a rede decore assinaturas digitais ou padrões de compressão de softwares geradores específicos.  
2. RandomRotation (Rotação de até 15°): Simula variações reais de posicionamento de sensores de captura ou inclinações biológicas nos exames. Garante invariância rotacional à rede.  
3. RandomHorizontalFlip e RandomVerticalFlip (Espelhamentos de 50%): Duplicam a diversidade espacial do dataset sem alterar a semântica da patologia. Um exame espelhado continua sendo real ou fake, mas obriga a
rede a aprender propriedades geométricas que não dependam da orientação esquerda/direita ou topo/fundo.  
4. ColorJitter (Perturbação de Brilho, Contraste, Saturação e Matiz): Esta é a defesa direta contra a presença indesejada de cartoons e pinturas. Ao variar agressivamente as propriedades de iluminação e cor em tempo de
execução, o modelo deixa de confiar em cores saturadas (comuns nos cartoons) ou tons específicos de iluminação (comuns em quadros a óleo) para ditar sua predição, focando puramente no conteúdo estrutural das imagens e aumentando a robustez para o domínio médico real.

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
