# MiniTales — TinyStories com vocabulário otimizado

Um fork do [nanoGPT](https://github.com/karpathy/nanoGPT) adaptado para treinar
um GPT-2 clássico de **5.25M parâmetros** no dataset **TinyStories**, usando um
vocabulário de **8.192 palavras** construído no próprio corpus.

**Licença:** MIT

O resultado é um modelo que **gera histórias coerentes em inglês**, ocupa
**21 MB em disco** (pesos fp32) e foi treinado em **~4h numa GPU T4 gratuita**.
---

## Sumário

1. [O que é](#o-que-é)
2. [Como funciona](#como-funciona)
3. [Resultados](#resultados)
4. [Comparação com o paper](#comparação-com-o-paper)
5. [Como rodar](#como-rodar)
6. [Estrutura do repo](#estrutura-do-repo)
7. [Formatos de checkpoint](#formatos-de-checkpoint)
8. [Limitações](#limitações)
9. [O que vem depois](#o-que-vem-depois)
10. [Exemplos de geração](#exemplos-de-geração)
11. [Ferramentas utilizadas](#ferramentas-utilizadas)
12. [Referências](#referências)
13. [Licença](#licença)

---

## O que é

Este repositório implementa um GPT-2 clássico (decoder-only Transformer)
treinado do zero no dataset
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(Eldan & Li, 2023).

A diferença em relação ao nanoGPT original está na **tokenização**:

| | nanoGPT (padrão) | MiniTales |
|---|---|---|
| Tokenizador | BPE (GPT-2) | Palavras inteiras + pontuação |
| Vocab | 50.257 | **8.192** |
| Embedding (n_embd=256) | 12.8M | **2.10M** |
| Cobertura do corpus | — | **99.84%** |

Com um vocabulário **6x menor**, o embedding deixa de dominar o orçamento de
parâmetros, liberando espaço para os blocos Transformer.

**Em termos simples:** em vez de gastar 60% dos parâmetros em embeddings
(como aconteceria com BPE de 50K), o MiniTales gasta 40% — e os outros 60% vão
para os blocos que aprendem a narrativa.

---

## Como funciona

```
TinyStories (Hugging Face)
       │
       ▼
prepare.py (normalização Unicode + tokenização por palavra)
       │
       ▼
train.bin / val.bin (uint16)
       │
       ▼
MiniTales (5.25M params)
  ├─ 4 camadas Transformer
  ├─ 4 heads
  ├─ d_model = 256
  ├─ context = 256 tokens
  └─ vocab = 8192
       │
       ▼
ckpt.pt
       │
       ▼
sample.py → texto
```

---

## Resultados

| Métrica | Valor |
|---|---|
| Parâmetros | 5.25M |
| Loss final (train) | **1.8417** |
| Loss final (val) | **1.8347** |
| Gap train/val | **-0.007** |
| Iterações | 50.000 |
| Tempo por iteração | ~152ms |
| Tokens por iteração | 8.192 |
| Tempo total de treino | ~4h no T4 |
| Pesos em disco (fp32) | **~21 MB** |
| `ckpt.pt` (com otimizador) | 60.8 MB |

O gap train/val ligeiramente negativo (val < train) indica que o modelo
**não está overfitting**.

---

## Comparação com o paper

### Arquitetura dos modelos oficiais

Os modelos oficiais do paper usam a arquitetura **GPT-Neo** (não GPT-2),
com as seguintes especificações:

| Modelo | Params | Camadas | Contexto | Vocab | `pytorch_model.bin` |
|---|---|---|---|---|---|
| TinyStories-1M | 1M | 8 | 512 | ~50K (GPT-Neo) | 48.6 MB |
| TinyStories-3M | 3.65M | 8 | 512 | ~50K (GPT-Neo) | 66.7 MB |
| TinyStories-8M | 8M | 8 | 512 | ~50K (GPT-Neo) | 116 MB |

Todos usam **8 camadas**, **contexto 512** e o tokenizador **GPT-Neo** (~50K).

### Comparação direta

| Modelo | Params | Camadas | Contexto | Vocab | Loss | Espaço em disco |
|---|---|---|---|---|---|---|
| TinyStories-1M (paper) | 1M | 8 | 512 | ~50K | ~2.5 | 48.6 MB |
| TinyStories-3M (paper) | 3.65M | 8 | 512 | ~50K | ~2.1 | 66.7 MB |
| **MiniTales (este repo)** | **5.25M** | **4** | **256** | **8192** | **1.835** | **~21 MB** |
| TinyStories-8M (paper) | 8M | 8 | 512 | ~50K | ~1.9 | 116 MB |

**Atenção:** a loss **não é diretamente comparável** entre modelos com
vocabulários diferentes. Com 8.192 classes (vocab 8192), a loss mínima
possível é `ln(8192) ≈ 9.01`. Com 50.257 classes (vocab GPT-Neo), é
`ln(50257) ≈ 10.82`. **A diferença esperada é de ~1.8 na loss**, o que
favorece artificialmente modelos com vocabulário menor.

Portanto, **a loss de 1.835 do MiniTales não significa que ele é melhor que os
modelos do paper**. Significa apenas que ele está resolvendo uma tarefa
diferente (prever entre 8.192 classes em vez de 50.257).

### Tamanho dos arquivos

O `pytorch_model.bin` dos modelos oficiais **inclui overhead do HuggingFace**
(configs, buffers, metadados). Os pesos puros são significativamente menores.
O MiniTales é armazenado em formato PyTorch puro, sem overhead.

**O MiniTales ocupa ~3x menos espaço em disco que o TinyStories-3M**, apesar de
ter mais parâmetros. Isso deve-se exclusivamente ao vocabulário menor
(8192 vs. ~50K).

---

## Como rodar

### 1. Instalar dependências

```bash
pip install torch numpy datasets tqdm
```

### 2. Preparar o dataset

```bash
python data/tinystories/prepare.py
```

Isso baixa o TinyStories do Hugging Face, normaliza Unicode, tokeniza por
palavra, e gera `train.bin`, `val.bin` e `meta.pkl`.

**Tempo:** ~7 min.

**Tamanho dos arquivos gerados:**

| Arquivo | Tamanho |
|---|---|
| `train.bin` | 900.07 MB |
| `val.bin` | 9.05 MB |
| `meta.pkl` | 140.04 KB |

### 3. Treinar

```bash
python train.py config/train_tinystories.py
```

**Tempo:** ~4h no T4 (Kaggle, Colab).

### 4. Gerar texto

```bash
python sample.py --out_dir=out-tinystories --start="once upon a time"
```

---

## Estrutura do repo

```
.
├── model.py                    # GPT-2 clássico
├── train.py                    # Loop de treino CPU/GPU autodetect
├── sample.py                   # Geração
├── configurator.py             # Override de config via CLI
├── config/
│   └── train_tinystories.py    # Hiperparâmetros
├── data/
│   └── tinystories/
│       └── prepare.py          # HF → train.bin / val.bin
└── README.md
```

---

## Formatos de checkpoint

O `ckpt.pt` (60.8 MB) contém **modelo + otimizador + metadados** — é o que
permite retomar o treino de onde parou. Para inferência, você pode extrair
versões menores:

| Formato | Tamanho | Uso |
|---|---|---|
| `ckpt.pt` | 60.8 MB | Retomar treino |
| `model.pt` (fp32) | ~21 MB | Inferência |
| `model_fp16.pt` | ~10.5 MB | Inferência otimizada |
| `model_int8.pt` | ~5 MB | Edge/mobile |

**Como extrair o `model.pt`:**

```python
import torch
from model import GPTConfig, GPT

ckpt = torch.load('out-tinystories/ckpt.pt', map_location='cpu')
model = GPT(GPTConfig(**ckpt['model_args']))
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)

torch.save({
    'model': model.state_dict(),
    'model_args': ckpt['model_args'],
}, 'out-tinystories/model.pt')
```

---

## Limitações

Esta seção é a mais importante do README. O modelo tem **várias limitações
reais**, e é importante conhecê-las antes de usá-lo.

### 1. O corpus define o teto

TinyStories tem aproximadamente **450M tokens**. O modelo de 5.25M saturou por
volta do step 40.000. Os últimos 10.000 iters ganharam apenas **~0.03 na loss**.

Treinar mais tempo refina o resultado, mas não ensina nada novo. Para
resultados qualitativamente melhores, seria necessário:
- Um corpus maior
- Um corpus mais diverso
- Um modelo maior

### 2. `<unk>` para palavras raras

Com um vocabulário de 8.192 palavras, aproximadamente **0.16% do corpus é
mapeado para `<unk>`**. Nos samples gerados, cerca de **6% dos outputs
contêm `<unk>` visível** (geralmente como `â œ`, resultado de aspas curvas
não normalizadas).

Isso é aceitável para TinyStories, mas seria um problema em corpus maiores ou
mais diversos.

### 3. O modelo reproduz o estilo do corpus, não o supera

**Este ponto é crucial.** O modelo **não é criativo no sentido literário**.
Ele aprende a distribuição estatística do TinyStories e a reproduz.

As histórias geradas são:
- **Gramaticalmente corretas** (~8/10)
- **Coerentes localmente** (as frases se conectam)
- **Mas previsíveis** — começam quase sempre com `"once upon a time"`, usam
  personagens com nomes comuns (`"Lily"`, `"Timmy"`, `"Sarah"`), e terminam
  com uma moral simples.

**Isso não é uma limitação do modelo em si, mas do corpus.** TinyStories foi
criado para ser simples e repetitivo por design. Um modelo treinado nele
aprende exatamente isso.

### 4. Diversidade limitada

Comparado aos modelos do paper, o MiniTales tem **diversidade menor**. Isso
deve-se a três fatores:

- **Vocabulário reduzido** (8192 vs. ~50K) — menos palavras disponíveis
- **Menor profundidade** (4 vs. 8 camadas) — menos capacidade de manter
  coerência global
- **Ausência de fine-tuning com instruções** (TinyStories-Instruct) — o
  modelo só viu histórias cruas, sem instruções explícitas

**Métricas qualitativas estimadas** (comparadas ao paper):

| Aspecto | MiniTales | Paper (modelos oficiais) |
|---|---|---|
| Gramática | ~8/10 | ~8/10 |
| Criatividade | ~5/10 | ~7/10 |
| Consistência | ~6/10 | ~8/10 |

**O MiniTales está abaixo do paper em criatividade e consistência, mas próximo
em gramática.**

### 5. Deriva de tópico

Em vários samples, o modelo **começa no prompt e depois "escapa"**:

- Prompt: `"Once upon a time, a small dog found a big bone."`
- Output: `"...one day, a little boy named Tim found a long stick..."`

O modelo não consegue manter o tópico do prompt por muitas frases. Isso é
uma consequência de:
- Contexto curto (256 tokens)
- Poucas camadas (4)
- Treino sem instruções

### 6. Repetição de estrutura

Cerca de **24% dos samples reiniciam com `"once upon a time"` no meio da
geração**, criando uma história dentro da história. Isso é um artefato do
treino: o modelo aprendeu que `"once upon a time"` é um bom começo, mas não
aprendeu quando parar.

### 7. Contexto de 256 tokens

As histórias do TinyStories têm em média ~200 tokens. O contexto de 256 cobre
a maioria dos casos, mas não sobra espaço para prompts longos ou histórias
mais complexas.

### 8. Sem técnicas modernas

O modelo **não usa**:
- RoPE (rotary position embeddings)
- SwiGLU (ativação moderna)
- RMSNorm (normalização moderna)
- GQA (grouped-query attention)
- Flash Attention otimizado (usa SDPA, mas sem garantia de kernel Flash)

É um GPT-2 clássico. O paper também usa GPT-Neo clássico, então a comparação
é justa. Mas técnicas modernas poderiam dar ganhos marginais (~0.05 na loss).

### 9. Sem GPT-4 Eval

O paper usa **GPT-4 como professor** para avaliar grammar, creativity e
consistency. Este repositório usa **loss de validação** como métrica única.

A loss **não captura** criatividade nem consistência tão bem quanto o GPT-4
Eval. Portanto, **não é possível afirmar que o MiniTales é "melhor" ou "pior"
que os modelos do paper** com base apenas na loss.

### 10. Sem TinyStories-Instruct

O paper tem uma variante do dataset chamada **TinyStories-Instruct**, que
inclui instruções explícitas (summaries, features, words, sentences). Modelos
treinados nessa variante são capazes de **seguir instruções** e **gerar
histórias com características específicas**.

O MiniTales **não foi treinado nessa variante**. Portanto, **não é capaz de
seguir instruções**. Ele só sabe gerar histórias no estilo padrão.

### 11. Arquitetura GPT-2 vs. GPT-Neo

O paper usa **GPT-Neo**, que tem diferenças sutis em relação ao GPT-2:
- **Activation:** `gelu_new` (vs. GELU padrão)
- **Attention:** às vezes usa attention local + global
- **Position embeddings:** aprendidos (como GPT-2)

Para modelos pequenos, a diferença é **mínima**, mas existe.

### 12. Sem avaliação humana

O paper usa prompts manuais e avaliação com GPT-4. O MiniTales **não tem
avaliação humana sistemática**. Os samples no README foram inspecionados
manualmente, mas **não há uma avaliação formal**.

### 13. Sem benchmark padronizado

Não há comparação com benchmarks como LAMBADA, CLOZE, TriviaQA ou Winograd.
O modelo foi avaliado apenas em **loss** e **inspeção qualitativa**.

### Resumo das limitações

| Limitação | Impacto |
|---|---|
| Corpus limitado (450M tokens) | Teto de qualidade |
| `<unk>` para palavras raras | 0.16% do corpus |
| Modelo reproduz estilo, não supera | Criatividade limitada |
| Diversidade menor que o paper | 5/10 vs. 7/10 |
| Deriva de tópico | Inconsistência em prompts longos |
| Repetição de estrutura | 24% dos samples |
| Contexto 256 | Limita prompts longos |
| Sem técnicas modernas | Ganho marginal perdido |
| Sem GPT-4 Eval | Métrica incompleta |
| Sem TinyStories-Instruct | Não segue instruções |
| GPT-2 vs. GPT-Neo | Diferença mínima |
| Sem avaliação humana | Sem métrica qualitativa formal |
| Sem benchmark padronizado | Sem comparação externa |

---

## O que vem depois

**Melhorias possíveis (não implementadas):**

- **Mais camadas:** 6 ou 8 (`n_layer=6` ou `n_layer=8`) → melhor consistência
- **Vocab maior:** 16K ou 32K → maior diversidade
- **Correção Unicode:** eliminar `â œ` dos samples
- **TinyStories-Instruct:** fine-tuning com instruções
- **GPT-4 Eval:** avaliação multidimensional com GPT-4
- **RoPE:** position embeddings rotacionais
- **SwiGLU:** ativação moderna

**Outros projetos (não este):**

- Treinar com código (novo pipeline)
- Aplicar a outro corpus

---

## Exemplos de geração

**Prompt:** `"once upon a time"` | **temperature:** 0.8 | **top_k:** 50

**Sample 1:**
```
once upon a time there was a girl named sarah . she was very brave
and loved to swim in the ocean . one day , sarah was swimming near a
big rock . suddenly she heard a noise coming from the rock . it was a
big , scary bear ! sarah was scared but a brave knight who could help
her . he used his strong legs and strong teeth to help her . they were
very happy and together they swam around the ocean . sarah was so
proud of herself for not being scared . she was happy that she could
help her friend . from that day on , sarah always remembered to be
brave and to protect herself in the sea . she learned that sometimes
it ' s best to help others whenever they need it .
```

**Sample 2:**
```
once upon a time , there was a boy named timmy . timmy loved to take
walks in the park with his mommy and daddy . one day , they saw a big
dog when timmy was barking . timmy felt anxious because he wanted to
take his dog home . mommy and daddy said , " don ' t worry , timmy .
we ' ll stay home soon . " they walked towards the dog and timmy
started to bark . mommy and daddy said , " you ' re so brave ! i love
you so much . " timmy felt happy that he listened to daddy and he
promised to take good care of him .
```

**Sample 3 (com `<unk>` visível):**
```
the train was late , so the people waited . after a few minutes , the
train was high up in the sky . but then the train stopped and looked
up . â œitâ tms okay , â said the people . â œyes , itâ tms just the
right size . letâ tms go get another one and we can go on ! â the
train was so happy . he had found the perfect size to ride .
```

**Sample 4 (deriva de tópico):**
```
once upon a time , a small dog found a big bone . he was very nice and
had a big heart . he liked to take care of it . one day , the dog
found a small , soft blanket . he thought it would be fun to keep it .
he put the blanket around him and played with it all day . the dog was
happy to have his best friend . one day , a little boy named tim found
a long stick . he thought it would be fun to throw it . [...]
```

---

## Ferramentas utilizadas

- **DeepSeek-V4 Chat** — utilizado como assistente durante todo o
  desenvolvimento: debate de arquitetura, análise de logs de treino, comparação
  com o paper, revisão de trade-offs, e escrita desta documentação.
- **Kaggle (GPU T4)** — treino final.
- **Google Colab (GPU T4)** — primeiras iterações e testes.
- **Hugging Face** — hospedagem do dataset TinyStories.

---

## Referências

- **nanoGPT** — Andrej Karpathy.
  [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **TinyStories: How Small Can Language Models Be and Still Speak Coherent
  English?** — Ronen Eldan, Yuanzhi Li (Microsoft Research, 2023).
  [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- **TinyStories dataset** —
  [huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

---

## Licença

MIT (mesma do nanoGPT).
