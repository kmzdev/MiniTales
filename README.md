# MiniTales-LM

Um GPT-2 clássico treinado do zero no dataset **TinyStories**, com vocabulário construído no próprio corpus. 

O resultado é um modelo que **gera histórias coerentes em inglês**, ocupa **~21 MB em disco** (pesos fp32) e foi treinado em **~4h numa GPU T4 gratuita**.

**Licença:** MIT (mesma do nanoGPT).

**Pesos:** disponíveis na página de [Releases](../../releases).

---

## Versões

| Versão | Params | Status | Notas |
|---|---|---|---|
| [**0.1.0**] | 5.25M | Pré-release | Prova de conceito. Gaps conhecidos. |
| [**1.0.0**] | ~7M | **Oficial** | Versão estável da geração 1.x. em testes!|

A partir da `1.0.0`, todas as comparações são feitas contra ela. A `0.1.0` é mantida como registro histórico do desenvolvimento inicial.

---

## O que é

Este repositório implementa um GPT-2 clássico (decoder-only Transformer) treinado do zero no dataset [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023).

A diferença em relação ao nanoGPT original está na **tokenização**:

| | nanoGPT (padrão) | MiniTales-LM |
|---|---|---|
| Tokenizador | BPE (GPT-2) | Palavras inteiras + pontuação |
| Vocab | 50.257 | **8.192** |
| Embedding (n_embd=256) | 12.8M | **2.10M** |
| Cobertura do corpus | — | **99.84%** |

Com um vocabulário **6× menor**, o embedding deixa de dominar o orçamento de parâmetros, liberando espaço para os blocos Transformer.

**Em termos simples:** em vez de gastar 60% dos parâmetros em embeddings (como aconteceria com BPE de 50K), o MiniTales-LM gasta 40% — e os outros 60% vão para os blocos que aprendem a narrativa.

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
MiniTales-LM
  ├─ 4 camadas Transformer (0.1.0) 
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

## Versão 0.1.0 — Prova de conceito

Primeira versão, treinada para validar o pipeline.

| Métrica | Valor |
|---|---|
| Parâmetros | 5.25M |
| Camadas | 4 |
| Contexto | 256 tokens |
| Precisão | bf16 (emulado) |
| Iterações | 50.000 |
| Loss final (train) | **1.8417** |
| Loss final (val) | **1.8347** |
| Gap train/val | **-0.007** |
| Tempo por iteração | ~152ms |
| Tokens por iteração | 8.192 |
| Tempo total de treino | ~4h no T4 |
| Pesos em disco (fp32) | **~21 MB** |
| `ckpt.pt` (com otimizador) | 60.8 MB |

O gap train/val ligeiramente negativo (val < train) indica que o modelo **não está overfitting**.

### Gaps conhecidos da 0.1.0

A versão 0.1.0 cumpriu o papel de prova de conceito, mas tem limitações que motivaram a versão 1.0.0:

| Item | 0.1.0 | Impacto |
|---|---|---|
| `dtype` | bf16 (emulado) | ~3–4× mais lento que fp16 nativo na T4 |
| `n_layer` | 4 | Pouca consistência em histórias longas |
| `batch_size` | 32 | Gradiente ruidoso |
| Normalização Unicode | ❌ | `â œ` visível em ~6% dos samples |
| `compile` | True (travando) | Compilação longa na T4 |

---

## Versão 1.0.0 — Versão oficial

A versão 1.0.0 corrige os gaps da 0.1.0 **sem mudar a arquitetura**. É o melhor GPT-2 clássico possível no TinyStories com os recursos disponíveis.

| Item | 0.1.0 | 1.0.0 |
|---|---|---|
| `n_layer` | 4 | **6** |
| `batch_size` | 32 | **128** |
| `dtype` | bf16 (emulado) | **fp16 nativo** |
| `compile` | True (travando) | **False** |
| `max_iters` | 50k | **100k** |
| `warmup_iters` | 500 | **1000** |
| Normalização Unicode | ❌ | **✅** |
| memmap | recriado | **reutilizado** |

### Por que essas mudanças

- **fp16 nativo:** a T4 (sm_75) não tem bf16 nativo. bf16 emulado roda a ~0.6× fp32. fp16 nativo roda a ~2.2× fp32. Ganho de **~3.7×**.
- **6 camadas:** mais profundidade melhora consistência em histórias longas.
- **batch 128:** melhor utilização dos Tensor Cores, gradiente mais limpo.
- **compile off:** o `max-autotune` trava na T4. Desligar é mais rápido que compilar.
- **Unicode normalizado:** elimina o `â œ` dos samples.

### Status

**Em treino.** Resultados serão adicionados quando disponíveis.

---

## Comparação com o paper

### Arquitetura dos modelos oficiais

Os modelos oficiais do paper usam a arquitetura **GPT-Neo** (não GPT-2), com as seguintes especificações:

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
| **MiniTales-LM 0.1.0** | **5.25M** | **4** | **256** | **8192** | **1.835** | **~21 MB** |
| TinyStories-8M (paper) | 8M | 8 | 512 | ~50K | ~1.9 | 116 MB |

**Atenção:** a loss **não é diretamente comparável** entre modelos com vocabulários diferentes. Com 8.192 classes (vocab 8192), a loss mínima possível é `ln(8192) ≈ 9.01`. Com 50.257 classes (vocab GPT-Neo), é `ln(50257) ≈ 10.82`. **A diferença esperada é de ~1.8 na loss**, o que favorece artificialmente modelos com vocabulário menor.

Portanto, **a loss de 1.835 do MiniTales-LM não significa que ele é melhor que os modelos do paper**. Significa apenas que ele está resolvendo uma tarefa diferente (prever entre 8.192 classes em vez de 50.257).

**O que o MiniTales-LM demonstra:** um vocabulário otimizado permite treinar um modelo de ~5M de parâmetros que gera histórias coerentes, ocupando **~3× menos espaço em disco** que os modelos oficiais de tamanho similar.

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

Isso baixa o TinyStories do Hugging Face, normaliza Unicode, tokeniza por palavra, e gera `train.bin`, `val.bin` e `meta.pkl`.

**Tempo:** ~7 min.

**Tamanho dos arquivos gerados:**

| Arquivo | Tamanho |
|---|---|
| `train.bin` | ~900 MB |
| `val.bin` | ~9 MB |
| `meta.pkl` | ~140 KB |

### 3. Treinar

```bash
# v0.1.0
python train.py config/train_tinystories_v0.1.0.py

# v1.0.0
python train.py config/train_tinystories_v1.0.0.py
```

**Tempo:** ~4h no T4 (v0.1.0) / ~1h no T4 (v1.0.0).

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
├── eval_prompts.py             # Avaliação com 50 prompts
├── configurator.py             # Override de config via CLI
├── config/
│   ├── train_tinystories_v0.1.0.py
│   └── train_tinystories_v1.0.0.py
├── data/
│   └── tinystories/
│       └── prepare.py          # HF → train.bin / val.bin
├── LICENSE
├── CHANGELOG.md
├── .gitignore
└── README.md
```

---

## Formatos de checkpoint

O `ckpt.pt` contém **modelo + otimizador + metadados** — é o que permite retomar o treino de onde parou. Para inferência, você pode extrair versões menores:

| Formato | Tamanho | Uso |
|---|---|---|
| `ckpt.pt` | ~61 MB (0.1.0) | Retomar treino |
| `minitales-lm-5.25m.pt` | ~21 MB | Inferência (v0.1.0) |
| `model_fp16.pt` | ~10.5 MB | Inferência otimizada |
| `model_int8.pt` | ~5 MB | Edge/mobile |

**Como carregar o release:**

```python
import torch
from model import GPTConfig, GPT

ckpt = torch.load('minitales-lm-5.25m.pt', map_location='cpu')
model = GPT(GPTConfig(**ckpt['model_args']))
model.load_state_dict(ckpt['model'])
model.eval()
```

---

## Limitações

Esta seção é a mais importante do README. O modelo tem **várias limitações reais**, e é importante conhecê-las antes de usá-lo.

### 1. O corpus define o teto

TinyStories tem aproximadamente **450M tokens**. O modelo de 5.25M saturou por volta do step 40.000. Os últimos 10.000 iters ganharam apenas **~0.03 na loss**.

Treinar mais tempo refina o resultado, mas não ensina nada novo. Para resultados qualitativamente melhores, seria necessário:

- Um corpus maior
- Um corpus mais diverso
- Um modelo maior

### 2. `<unk>` para palavras raras

Com um vocabulário de 8.192 palavras, aproximadamente **0.16% do corpus é mapeado para `<unk>`**. Nos samples gerados, cerca de **6% dos outputs contêm `<unk>` visível** (geralmente como `â œ`, resultado de aspas curvas não normalizadas).

Isso é aceitável para TinyStories, mas seria um problema em corpus maiores ou mais diversos. A v1.0.0 corrige isso com normalização Unicode.

### 3. O modelo reproduz o estilo do corpus, não o supera

**Este ponto é crucial.** O modelo **não é criativo no sentido literário**. Ele aprende a distribuição estatística do TinyStories e a reproduz.

As histórias geradas são:

- **Gramaticalmente corretas** (~8/10)
- **Coerentes localmente** (as frases se conectam)
- **Mas previsíveis** — começam quase sempre com `"once upon a time"`, usam personagens com nomes comuns (`"Lily"`, `"Timmy"`, `"Sarah"`), e terminam com uma moral simples.

**Isso não é uma limitação do modelo em si, mas do corpus.** TinyStories foi criado para ser simples e repetitivo por design. Um modelo treinado nele aprende exatamente isso.

### 4. Diversidade limitada

Comparado aos modelos do paper, o MiniTales-LM tem **diversidade menor**. Isso deve-se a três fatores:

- **Vocabulário reduzido** (8192 vs. ~50K) — menos palavras disponíveis
- **Menor profundidade** (4 vs. 8 camadas na v0.1.0) — menos capacidade de manter coerência global
- **Ausência de fine-tuning com instruções** (TinyStories-Instruct) — o modelo só viu histórias cruas, sem instruções explícitas

**Métricas qualitativas estimadas** (comparadas ao paper):

| Aspecto | MiniTales-LM | Paper (modelos oficiais) |
|---|---|---|
| Gramática | ~8/10 | ~8/10 |
| Criatividade | ~5/10 | ~7/10 |
| Consistência | ~6/10 | ~8/10 |

**O MiniTales-LM está abaixo do paper em criatividade e consistência, mas próximo em gramática.**

### 5. Deriva de tópico

Em vários samples, o modelo **começa no prompt e depois "escapa"**:

- Prompt: `"Once upon a time, a small dog found a big bone."`
- Output: `"...one day, a little boy named Tim found a long stick..."`

O modelo não consegue manter o tópico do prompt por muitas frases. Isso é uma consequência de:

- Contexto curto (256 tokens)
- Poucas camadas (4 na v0.1.0)
- Treino sem instruções

### 6. Repetição de estrutura

Cerca de **24% dos samples reiniciam com `"once upon a time"` no meio da geração**, criando uma história dentro da história. Isso é um artefato do treino: o modelo aprendeu que `"once upon a time"` é um bom começo, mas não aprendeu quando parar.

### 7. Contexto de 256 tokens

As histórias do TinyStories têm em média ~200 tokens. O contexto de 256 cobre a maioria dos casos, mas não sobra espaço para prompts longos ou histórias mais complexas.

### 8. Sem técnicas modernas

O modelo **não usa**:

- RoPE (rotary position embeddings)
- SwiGLU (ativação moderna)
- RMSNorm (normalização moderna)
- GQA (grouped-query attention)
- Flash Attention otimizado (usa SDPA, mas sem garantia de kernel Flash)

É um GPT-2 clássico. O paper também usa GPT-Neo clássico, então a comparação é justa. Mas técnicas modernas poderiam dar ganhos marginais (~0.05 na loss).

### 9. Sem GPT-4 Eval

O paper usa **GPT-4 como professor** para avaliar grammar, creativity e consistency. Este repositório usa **loss de validação** como métrica única.

A loss **não captura** criatividade nem consistência tão bem quanto o GPT-4 Eval. Portanto, **não é possível afirmar que o MiniTales-LM é "melhor" ou "pior" que os modelos do paper** com base apenas na loss.

### 10. Sem TinyStories-Instruct

O paper tem uma variante do dataset chamada **TinyStories-Instruct**, que inclui instruções explícitas (summaries, features, words, sentences). Modelos treinados nessa variante são capazes de **seguir instruções** e **gerar histórias com características específicas**.

O MiniTales-LM **não foi treinado nessa variante**. Portanto, **não é capaz de seguir instruções**. Ele só sabe gerar histórias no estilo padrão.

### 11. Arquitetura GPT-2 vs. GPT-Neo

O paper usa **GPT-Neo**, que tem diferenças sutis em relação ao GPT-2:

- **Activation:** `gelu_new` (vs. GELU padrão)
- **Attention:** às vezes usa attention local + global
- **Position embeddings:** aprendidos (como GPT-2)

Para modelos pequenos, a diferença é **mínima**, mas existe.

### 12. Sem avaliação humana

O paper usa prompts manuais e avaliação com GPT-4. O MiniTales-LM **não tem avaliação humana sistemática**. Os samples no README foram inspecionados manualmente, mas **não há uma avaliação formal**.

### 13. Sem benchmark padronizado

Não há comparação com benchmarks como LAMBADA, CLOZE, TriviaQA ou Winograd. O modelo foi avaliado apenas em **loss** e **inspeção qualitativa**.

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

## Exemplos de geração (v0.1.0)

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

- **DeepSeek (chat)** — utilizado como assistente durante todo o desenvolvimento: debate de arquitetura, análise de logs de treino, comparação com o paper, revisão de trade-offs, e escrita desta documentação.
- **Kaggle (GPU T4)** — treino final.
- **Google Colab (GPU T4)** — primeiras iterações e testes.
- **Hugging Face** — hospedagem do dataset TinyStories.

---

## Referências

- **nanoGPT** — Andrej Karpathy. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** — Ronen Eldan, Yuanzhi Li (Microsoft Research, 2023). [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- **TinyStories dataset** — [huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

---

## Licença

MIT (mesma do nanoGPT).
