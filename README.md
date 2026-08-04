# Metis

Assistente de finanças pessoais e análise cripto da OlympkusAI.

Metis é um serviço backend (FastAPI + LangGraph) que orquestra um agente LLM com raciocínio explícito, ferramentas de cálculo local, integração com APIs externas (Apollo ML, Pluto finanças pessoais) e persistência de histórico de conversas em Postgres próprio (db-metis).

## Organização

- **Organização**: [olympkusai](https://github.com/olympkusai)
- **Repositório**: [olympkusai/metis](https://github.com/olympkusai/metis)
- **Pacote Python**: `metis`
- **Deploy**: Railway (serviço `metis` + banco `db-metis`)

## Arquitetura

```
Cliente (JWT)
  │
  ↓
FastAPI (metis.main)
  ├── POST /api/chat          → agente LangGraph (finance ou crypto)
  ├── POST /api/chat/stream   → streaming SSE
  ├── GET  /api/conversations/*  → histórico, sessões, feedback
  ├── GET  /health
  └── GET  /
  │
  ↓
LangGraph Agent (metis.agent.graph)
  ├── Orchestrator            → classifica intenção, roteia domínio
  ├── Finance pipeline        → tools de finanças pessoais (Pluto API)
  ├── Crypto pipeline         → tools de mercado (cálculo local + Apollo ML)
  │   ├── FeatureAgent        → RSI, MACD, Bollinger, volatilidade, etc.
  │   ├── RiskAgent           → VaR, Sharpe, CVaR, Max Drawdown
  │   ├── SignalAgent         → MoE + trend state machine
  │   ├── PreTradeRiskGate    → circuit breaker, limites de exposição
  │   └── ExecutionLayer      → sizing, slippage model
  └── DecisionEngine          → síntese final com audit trail
  │
  ↓
Bancos de dados
  ├── db-metis (Postgres próprio)     → conversations, chat_messages, notifications
  └── k0s Postgres (externo, read-only) → market_candles (cache de dados de mercado)
  │
  ↓
Serviços externos
  ├── Apollo (ML)           → previsões, backtests, treino de modelos
  └── Pluto (finanças)      → transações, orçamentos, metas, recorrências
```

## Estrutura do projeto

```
metis/
├── metis/                    # pacote Python
│   ├── main.py               # FastAPI app + lifespan (pools, migrations)
│   ├── config.py             # Settings (pydantic-settings, lê .env)
│   ├── api/                  # endpoints FastAPI
│   │   ├── chat.py           # POST /chat, /chat/stream
│   │   └── conversations.py  # CRUD de conversas, sessões, feedback
│   ├── agent/                # LangGraph multi-agente
│   │   ├── graph.py          # grafo principal (orchestrator + pipelines)
│   │   ├── decision_engine.py
│   │   ├── finance_prompts.py
│   │   ├── forecasting.py    # integração Apollo (previsões, backtests)
│   │   ├── moe.py            # Mixture of Experts para sinais
│   │   ├── portfolio.py      # constraints de portfólio
│   │   ├── quant_engine.py   # signal score, risk level, position sizing
│   │   ├── schemas.py        # Pydantic models (outputs estruturados)
│   │   └── trend_state.py    # state machine de tendência multi-timeframe
│   ├── calculator/           # cálculos técnicos locais (sem API externa)
│   │   ├── engine.py         # CalculationEngine orquestra features/indicators
│   │   ├── rsi.py, macd.py, bollinger_bands.py, volatility.py, ...
│   │   ├── risk_metrics.py   # VaR, Sharpe, CVaR, drawdown
│   │   └── aggregation.py    # agregação multi-timeframe via SQL
│   ├── tools/                # ferramentas LangChain expostas ao agente
│   │   ├── local.py          # tools de crypto (preço, indicadores, risco)
│   │   └── finance.py        # tools de finanças pessoais (Pluto)
│   ├── memory/
│   │   └── conversation_history.py  # persistência em db-metis
│   ├── storage/              # camada de dados (Postgres)
│   │   ├── pool.py           # DatabasePool (asyncpg)
│   │   ├── migrations.py     # DDL: tabelas calculator + conversation schema
│   │   ├── market_candle.py  # leitura de market_candles (k0s Postgres)
│   │   ├── cache.py, crud.py, models.py
│   ├── apollo_client.py      # cliente HTTP Apollo (ML)
│   ├── pluto_client.py       # cliente HTTP Pluto (finanças pessoais)
│   └── utils/
│       ├── cost_tracker.py   # tracking de custo de tokens
│       └── timing.py         # decorators de timing
├── tests/                    # pytest (296 testes)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml            # Poetry, pacote `metis`
├── run_migration.py          # script: rodar migrations do k0s Postgres
├── check_db_intervals.py     # debug: checar disponibilidade de candles
└── .env.example              # template de variáveis de ambiente
```

## Pré-requisitos

- Python 3.14+
- Poetry
- Docker (opcional, para rodar local via docker-compose)
- Acesso ao db-metis (Postgres no Railway)
- Acesso ao k0s Postgres (externo, read-only — cache de market data)
- OPENAI_API_KEY
- APOLLO_BASE_URL (serviço Apollo ML)
- PLUTO_BASE_URL (serviço Pluto finanças pessoais)

## Setup local

```bash
# 1. Clonar
git clone https://github.com/olympkusai/metis.git
cd metis

# 2. Instalar dependências
poetry install

# 3. Configurar ambiente
cp .env.example .env
# Editar .env com as credenciais reais

# 4. Rodar
poetry run uvicorn metis.main:app --host 0.0.0.0 --port 8082 --reload
```

### Docker

```bash
docker-compose up --build
# Backend disponível em http://localhost:8001
```

## Variáveis de ambiente

| Variável | Descrição |
|----------|-----------|
| `OPENAI_API_KEY` | Chave da OpenAI (LLM) |
| `CONVERSATION_DATABASE_URL` | DSN do db-metis (Postgres próprio — conversas) |
| `DATABASE_URL` | DSN do k0s Postgres (externo, read-only — market data) |
| `APOLLO_BASE_URL` | URL base do serviço Apollo (ML) |
| `PLUTO_BASE_URL` | URL base do serviço Pluto (finanças pessoais) |
| `API_BASE_URL` | URL base da API de dados cripto (k0s.app) |

## Endpoints

### Chat

| Método | Path | Descrição |
|--------|------|-----------|
| `POST` | `/api/chat` | Envia mensagem, recebe resposta do agente |
| `POST` | `/api/chat/stream` | Envia mensagem, recebe resposta via SSE |

**Body:**
```json
{
  "message": "Qual o preço do BTC?",
  "user_id": "user-123",
  "session_id": "sess-456",
  "domain": "crypto"
}
```

`domain` pode ser `"finance"` (padrão) ou `"crypto"`.

### Conversas

| Método | Path | Descrição |
|--------|------|-----------|
| `GET` | `/api/conversations/sessions` | Lista sessões do usuário |
| `GET` | `/api/conversations/sessions/{session_id}` | Histórico de uma sessão |
| `GET` | `/api/conversations/recent` | Contexto global (mensagens recentes) |
| `DELETE` | `/api/conversations/sessions/{session_id}` | Soft-delete de sessão |
| `DELETE` | `/api/conversations/messages/{message_id}` | Soft-delete de mensagem |
| `POST` | `/api/conversations/messages/{message_id}/feedback` | Feedback em mensagem |
| `POST` | `/api/conversations/sessions/{session_id}/feedback` | Feedback em sessão |

### Health

| Método | Path | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status do serviço |
| `GET` | `/` | Info básica |

## Banco de dados (db-metis)

Postgres próprio, schema seguindo o domínio "communication" (dbml/communication.md):

- **`conversations`** — sessões de conversa (id, user_id, title, timestamps, soft-delete)
- **`chat_messages`** — mensagens (role, content, metadata jsonb, embedding reservado, soft-delete)
- **`chat_message_feedback`** — feedback por mensagem (rating, comment)
- **`notifications`** — notificações (reservado, sem consumidor ainda)

Migrations rodam automaticamente no startup (`run_conversation_migrations` em `metis/storage/migrations.py`).

A coluna `embedding` em `chat_messages` é reservada para recall semântico futuro — hoje não é populada nem consumida por nenhuma feature. Quando o feature for implementado, re-adicionar pgvector é localizado: `CREATE EXTENSION vector` + `ALTER COLUMN embedding TYPE vector(1536)` + índice ivfflat + chamada `embed_query` no `save_message`.

## Testes

```bash
poetry run python -m pytest
```

296 testes cobrindo: agent (routing, intent, extraction, forecasting, multi-asset state), API (chat helpers), calculator (engine, calculators, frequent calculations).

## Scripts auxiliares

```bash
# Rodar migrations do k0s Postgres (make_interval, tabelas calculator)
poetry run python run_migration.py

# Checar disponibilidade de candles por intervalo
poetry run python check_db_intervals.py BTCUSDT 30
```
