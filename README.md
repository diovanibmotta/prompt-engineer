# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

Sistema para otimização de prompts de baixa qualidade do LangSmith Hub usando técnicas avançadas de Prompt Engineering, avaliação LLM-as-Judge e refinamento iterativo.

---

## Sumário

- [Técnicas Aplicadas (Fase 2)](#técnicas-aplicadas-fase-2)
- [Resultados Finais](#resultados-finais)
- [Como Executar](#como-executar)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Técnicas Aplicadas (Fase 2)

### 1. Few-shot Learning *(obrigatório)*

**Por quê:** O modelo v1 produzia user stories genéricas por não ter referência do formato esperado. Com exemplos concretos de entrada/saída, o modelo aprende o padrão sem precisar de instruções exaustivas.

**Como aplicou:** 3 exemplos com complexidade variada (simples, médio, complexo), cada um demonstrando:
- Bug report de entrada
- User story completa de saída no formato "Como [papel], quero [funcionalidade], para que [benefício]"
- Critérios de aceite em formato Given/When/Then

```yaml
# Exemplo no prompt v2
EXAMPLES:
Example 1 - Simple Bug:
Input: "Add to cart button doesn't work on product ID 1234."
Output: "As a customer browsing the store, I want to successfully add products
to my shopping cart, so that I can complete my purchase..."
```

### 2. Chain of Thought (CoT)

**Por quê:** Bugs têm contexto técnico implícito que precisa ser raciocínado antes de gerar a user story. CoT força o modelo a considerar impacto, papel do usuário e critérios de aceite antes de escrever.

**Como aplicou:** Instrução explícita de raciocínio passo a passo no system prompt:

```
APPROACH - Think step by step:
1. Understand the bug's impact and affected user
2. Identify the desired outcome from user perspective
3. Define clear acceptance criteria
4. Structure the output with all required sections
```

### 3. Role Prompting

**Por quê:** Definir persona especializada eleva a qualidade e consistência do output. O modelo se comporta como expert ao receber contexto de autoridade.

**Como aplicou:**

```
You are an expert Product Manager with 10+ years of experience
converting bug reports into clear, actionable user stories for
agile development teams.
```

### 4. Structured Output Format

**Por quê:** O prompt v1 não especificava formato de saída, gerando respostas inconsistentes. Especificar estrutura exata garante outputs parseáveis e completos.

**Como aplicou:** Seções obrigatórias no output com formato Markdown:

```
OUTPUT FORMAT:
## User Story
As a [role], I want [feature], so that [benefit].

## Acceptance Criteria
- Given [context], When [action], Then [result]

## Technical Context (if applicable)
[Technical details relevant to implementation]
```

---

## Resultados Finais

### Comparação v1 vs v2

| Métrica | v1 (original) | v2 (otimizado) | Melhoria |
|---------|--------------|----------------|----------|
| Helpfulness | 0.45 ✗ | **0.9450** ✓ | +0.495 |
| Correctness | 0.52 ✗ | **0.9263** ✓ | +0.406 |
| F1-Score | 0.48 ✗ | **0.9058** ✓ | +0.426 |
| Clarity | 0.50 ✗ | **0.9433** ✓ | +0.443 |
| Precision | 0.46 ✗ | **0.9467** ✓ | +0.487 |
| **Status** | ❌ REPROVADO | ✅ **APROVADO** | — |

### Scores Finais (v2)

```
======================================================================
RESULTADOS DA AVALIAÇÃO
======================================================================

📊 Scores das Métricas:
----------------------------------------------------------------------
  helpfulness    : 0.9450 ✓
  correctness    : 0.9263 ✓
  f1_score       : 0.9058 ✓
  clarity        : 0.9433 ✓
  precision      : 0.9467 ✓

----------------------------------------------------------------------
✅ APROVADO - Todas as métricas >= 0.9
```

### LangSmith Dashboard

- **Projeto:** MBA
- **Prompt publicado:** https://smith.langchain.com/hub/diovanimotta/bug_to_user_story_v2
- **Tracing:** Habilitado (`LANGSMITH_TRACING=true`) — execuções visíveis no dashboard do projeto MBA

---

## Como Executar

### Pré-requisitos

- Python 3.9+
- Conta no [LangSmith](https://smith.langchain.com) (tier gratuito)
- API Key da [OpenAI](https://platform.openai.com/api-keys) **ou** [Google Gemini](https://aistudio.google.com/app/apikey)

### 1. Clonar e configurar ambiente

```bash
git clone <url-do-repositorio>
cd prompt-engineer

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Editar `.env` com suas credenciais:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_sk_...
LANGSMITH_PROJECT=MBA
USERNAME_LANGSMITH_HUB=seu_username

# OpenAI (padrão)
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EVAL_MODEL=gpt-4o

# Gemini (alternativa gratuita)
# GOOGLE_API_KEY=AIzaSy...
# LLM_PROVIDER=google
# LLM_MODEL=gemini-2.5-flash
# EVAL_MODEL=gemini-2.5-flash
```

### 3. Pull do prompt original

```bash
python -m src.pull_prompts
```

Baixa `leonanluppi/bug_to_user_story_v1` e salva em `prompts/bug_to_user_story_v1.yml`.

### 4. Push do prompt otimizado

```bash
python -m src.push_prompts
```

Publica `prompts/bug_to_user_story_v2.yml` no LangSmith Hub como `{seu_username}/bug_to_user_story_v2`.

### 5. Executar avaliação

```bash
python -m src.evaluate
```

Avalia o prompt v2 com 15 exemplos do dataset e exibe scores das 5 métricas.

### 6. Rodar testes

```bash
pytest tests/test_prompts.py -v
```

Executa os 6 testes de validação do prompt v2.

---

## Métricas de Avaliação

Todas as métricas usam **LLM-as-Judge** com temperatura 0 para resultados determinísticos.

| Métrica | Fórmula | O que mede |
|---------|---------|-----------|
| Helpfulness | (Clarity + Precision) / 2 | Utilidade para devs |
| Correctness | (F1-Score + Precision) / 2 | Precisão factual |
| F1-Score | 2 × (P × R) / (P + R) | Equilíbrio precisão/recall |
| Clarity | LLM-as-Judge | Clareza e organização |
| Precision | LLM-as-Judge | Foco e ausência de alucinações |

**Critério de aprovação:** todas as métricas >= 0.9

---

## Estrutura do Projeto

```
prompt-engineer/
├── .env.example              # Template de variáveis de ambiente
├── .env                      # Suas credenciais (não commitar)
├── requirements.txt          # Dependências Python
├── README.md                 # Esta documentação
│
├── prompts/
│   ├── bug_to_user_story_v1.yml  # Prompt original (baixo desempenho)
│   └── bug_to_user_story_v2.yml  # Prompt otimizado (aprovado ✅)
│
├── datasets/
│   └── bug_to_user_story.jsonl   # 15 exemplos de bugs
│
├── src/
│   ├── pull_prompts.py       # Pull do LangSmith Hub
│   ├── push_prompts.py       # Push ao LangSmith Hub
│   ├── evaluate.py           # Orquestração de avaliação
│   ├── metrics.py            # 5 métricas implementadas
│   ├── llm_provider.py       # Abstração multi-provider (OpenAI/Gemini)
│   ├── config.py             # Gerenciamento de configuração
│   └── utils.py              # Funções auxiliares
│
└── tests/
    └── test_prompts.py       # 6 testes de validação do prompt
```
