# Veritas — Classificação de Imagens autênticas ou manipuladas (repositório)

Este repositório contém utilitários, modelos e scripts para treino e inferência
de classificadores de imagem usados no projeto *Veritas*.

**Grupo: Ivan Edward @ iers-bd ; Elizabete Barbosa @ elizabetealbuquerque ; Joao Rietra @ jhlr ;**  
**Nome da disciplina: Machine Learning I - School Innovation.**  
**Nome da instituição de ensino: CESAR School.**  
**Nome da solução desenvolvida: Veritas**  

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

## Instruções detalhadas para compilar e executar.  

## Links importantes para o projeto.  
DATSET: https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k  
DATASET: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images  
DATASET: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces  
DRIVE: https://drive.google.com/drive/folders/1ywYBYQ1KoG0kaChqxis3LcpLXM4J5CHt?usp=sharing  
APLICAÇÃO: https://app-veritas.netlify.app  
MIRO: https://miro.com/app/board/uXjVJQuHnVQ=/?share_link_id=526481362291  
GOOGLE SITES:  
FIGMA ATUALIZADO:   
