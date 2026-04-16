# Plano Completo de Implementação – Agente de Investimento Cripto com Raciocínio

## Sumário
1. [Arquitetura Geral](#1-arquitetura-geral)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Setup do Ambiente e Infraestrutura](#3-setup-do-ambiente-e-infraestrutura)
4. [Backend – FastAPI + LangChain + LangGraph](#4-backend--fastapi--langchain--langgraph)
5. [Implementação do Agente com Raciocínio (LangGraph)](#5-implementação-do-agente-com-raciocínio-langgraph)
6. [Ferramentas (Tools) de Dados e ML](#6-ferramentas-tools-de-dados-e-ml)
7. [Memória Híbrida (Redis + Pinecone)](#7-memória-híbrida-redis--pinecone)
8. [RAG com LlamaIndex e Pinecone](#8-rag-com-llamaindex-e-pinecone)
9. [Observabilidade com LangSmith](#9-observabilidade-com-langsmith)
10. [Frontend – Next.js com Streaming de Raciocínio](#10-frontend--nextjs-com-streaming-de-raciocínio)
11. [Testes e Validação](#11-testes-e-validação)
12. [Deploy e Monitoramento](#12-deploy-e-monitoramento)

---

## 1. Arquitetura Geral

```
[Next.js] → [FastAPI] → [LangGraph Agent]
                             │
          ┌──────────────────┼──────────────────┐
          ↓                  ↓                  ↓
      [Tools]           [Memory]           [RAG]
    (APIs/ML)         (Redis/Pinecone)   (LlamaIndex)
          │                  │                  │
          ↓                  ↓                  ↓
   PostgreSQL/Redis    Histórico + Perfil   Documentos/News
```

**Componentes Principais**:
- **Orquestração**: LangChain + LangGraph (loop de raciocínio explícito)
- **LLM**: OpenAI GPT-4o (ou GPT-4 Turbo)
- **Vector DB**: Pinecone (memória de longo prazo e RAG)
- **Cache**: Redis (buffer de conversa e cache de ferramentas)
- **DB relacional**: PostgreSQL (dados de mercado, portfólios, usuários)
- **Backend API**: FastAPI (endpoints de chat e admin)
- **Frontend**: Next.js (chat interativo com streaming)
- **Observabilidade**: LangSmith (traces, custos, performance)

---

## 2. Pré-requisitos

- **Contas/APIs**:
  - OpenAI API key
  - Pinecone (crie um índice)
  - LangSmith API key
  - Provedor de dados cripto (ex: Binance, CoinGecko, CoinMarketCap)
- **Software local**:
  - Python 3.11+
  - Node.js 20+
  - Docker + Docker Compose (PostgreSQL, Redis)
  - Poetry (gerenciador de dependências Python)
  - pnpm (para Next.js)

---

## 3. Setup do Ambiente e Infraestrutura

### 3.1. Estrutura de pastas

```
crypto-agent/
├── backend/
│   ├── app/
│   │   ├── api/           # endpoints FastAPI
│   │   ├── agent/         # LangGraph agent, nós, prompts
│   │   ├── tools/         # ferramentas (preço, indicadores, ML)
│   │   ├── memory/        # Redis + Pinecone memory wrappers
│   │   ├── rag/           # LlamaIndex + Pinecone indexação
│   │   ├── models/        # SQLAlchemy models (PostgreSQL)
│   │   └── config.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── pyproject.toml
├── frontend/
│   ├── app/               # Next.js App Router
│   ├── components/        # Chat UI, streaming
│   └── package.json
└── README.md
```

### 3.2. Docker Compose (infra)

```yaml
# backend/docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: crypto
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: cryptodb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  backend:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    env_file:
      - .env
    volumes:
      - ./app:/app
```

### 3.3. Variáveis de ambiente (`.env`)

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=crypto-memory
LANGSMITH_API_KEY=...
DATABASE_URL=postgresql://crypto:secret@postgres:5432/cryptodb
REDIS_URL=redis://redis:6379
CRYPTO_DATA_API_KEY=...   # Binance/CoinGecko
```

---

## 4. Backend – FastAPI + LangChain + LangGraph

### 4.1. Inicialização do projeto (Poetry)

```bash
cd backend
poetry init
poetry add fastapi uvicorn langchain langchain-openai langgraph langsmith
poetry add llama-index pinecone-client redis asyncpg sqlalchemy
poetry add python-dotenv pydantic-settings
```

### 4.2. Configuração (`app/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    langsmith_api_key: str
    database_url: str
    redis_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

### 4.3. Endpoint de chat (`app/api/chat.py`)

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.agent.graph import get_agent_graph
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: str

@router.post("/chat")
async def chat(request: ChatRequest):
    agent = get_agent_graph()
    # Estado inicial
    state = {
        "messages": [("human", request.message)],
        "user_id": request.user_id,
        "next_action": "reason"
    }
    # Execução síncrona ou streaming
    final_state = await agent.ainvoke(state)
    return {"response": final_state["messages"][-1].content}

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    agent = get_agent_graph()
    # Usar streaming via SSE
    async def event_generator():
        async for event in agent.astream_events(
            {"messages": [("human", request.message)], "user_id": request.user_id},
            version="v1"
        ):
            if event["event"] == "on_chain_end" and "reasoning" in event["data"]:
                yield f"data: {event['data']['reasoning']}\n\n"
            elif event["event"] == "on_tool_end":
                yield f"data: [TOOL] {event['name']} → {event['data']['output']}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 5. Implementação do Agente com Raciocínio (LangGraph)

### 5.1. Definição do estado e nós (`app/agent/state.py`)

```python
from typing import List, Tuple, TypedDict, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_id: str
    next_action: str                # "reason", "action", "observe", "finalize"
    intermediate_steps: List[Tuple[str, str]]  # (tool_name, result)
    reasoning: str                  # passo-a-passo acumulado
    final_answer: Optional[str]
```

### 5.2. Prompt de raciocínio (`app/agent/prompts.py`)

```python
REASONING_PROMPT = """
Você é um analista de investimentos em criptomoedas. Siga EXATAMENTE o formato abaixo:

Pensamento: (raciocínio sobre o que fazer agora)
Ação: (nome da ferramenta a chamar – deve ser uma das: get_live_price, get_indicators, predict_future, calculate_risk, optimize_portfolio, search_market_news)
Entrada: (parâmetros para a ferramenta em JSON)

Quando tiver informações suficientes, escreva:
Resposta Final: (análise completa, incluindo riscos e recomendação)

Ferramentas disponíveis:
- get_live_price(symbol): retorna preço atual
- get_indicators(symbol): RSI, MACD, médias
- predict_future(symbol, horizon): previsão com confiança
- calculate_risk(symbol, amount): VaR, Sharpe
- optimize_portfolio(symbols, constraints): alocação ótima
- search_market_news(query): notícias recentes (RAG)

Histórico da conversa: {chat_history}
Memória de longo prazo do usuário: {user_memory}
"""
```

### 5.3. Nós do grafo (`app/agent/graph.py`)

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from app.agent.state import AgentState
from app.agent.prompts import REASONING_PROMPT
from app.tools import all_tools
from app.memory import get_memory_context

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
tool_executor = ToolExecutor(all_tools)

def reasoning_node(state: AgentState):
    """Planeja a próxima ação ou finaliza."""
    # Busca memória de longo prazo do usuário via Pinecone
    user_memory = get_memory_context(state["user_id"], state["messages"][-1].content)
    prompt = REASONING_PROMPT.format(
        chat_history=state["messages"][-5:],
        user_memory=user_memory
    )
    response = llm.invoke(prompt)
    # Parsing simples: extrair "Pensamento:", "Ação:", "Entrada:", ou "Resposta Final:"
    if "Resposta Final:" in response.content:
        state["final_answer"] = response.content.split("Resposta Final:")[-1].strip()
        state["next_action"] = "finalize"
    else:
        # Extrair ação e entrada (exemplo simplificado)
        action_line = [l for l in response.content.split("\n") if "Ação:" in l][0]
        tool_name = action_line.split("Ação:")[-1].strip()
        input_line = [l for l in response.content.split("\n") if "Entrada:" in l][0]
        tool_input = eval(input_line.split("Entrada:")[-1].strip())  # cuidado, use json.loads
        state["intermediate_steps"].append((tool_name, None))
        state["next_action"] = "action"
        state["current_tool"] = tool_name
        state["current_input"] = tool_input
    state["reasoning"] += response.content + "\n"
    return state

def action_node(state: AgentState):
    """Executa a ferramenta selecionada."""
    tool = next(t for t in all_tools if t.name == state["current_tool"])
    result = tool.invoke(state["current_input"])
    # Atualiza último passo
    state["intermediate_steps"][-1] = (state["current_tool"], result)
    state["next_action"] = "observe"
    return state

def observe_node(state: AgentState):
    """Prepara observação para o próximo ciclo de raciocínio."""
    last_tool, last_result = state["intermediate_steps"][-1]
    observation = f"Observação de {last_tool}: {last_result}"
    # Adiciona ao histórico de mensagens
    state["messages"].append(("ai", observation))
    state["next_action"] = "reason"
    return state

def finalize_node(state: AgentState):
    """Retorna a resposta final."""
    state["messages"].append(("ai", state["final_answer"]))
    state["next_action"] = END
    return state

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("action", action_node)
    workflow.add_node("observe", observe_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("reason")
    workflow.add_conditional_edges(
        "reason",
        lambda s: s["next_action"],
        {"action": "action", "finalize": "finalize"}
    )
    workflow.add_edge("action", "observe")
    workflow.add_edge("observe", "reason")
    workflow.add_edge("finalize", END)

    return workflow.compile()
```

---

## 6. Ferramentas (Tools) de Dados e ML

### 6.1. Estrutura base (`app/tools/__init__.py`)

```python
from langchain.tools import tool
import httpx
from app.models.ml import predict_price  # modelo Transformer+XGBoost
from app.db.crud import get_indicators_from_db

@tool
def get_live_price(symbol: str) -> str:
    """Retorna preço atual da criptomoeda."""
    # Exemplo via Binance API
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}USDT"
    resp = httpx.get(url).json()
    return f"Preço atual de {symbol.upper()}: ${resp['price']}"

@tool
def get_indicators(symbol: str) -> str:
    """Retorna indicadores técnicos (RSI, MACD, médias)."""
    # Buscar do PostgreSQL (dados já calculados por job separado)
    indicators = get_indicators_from_db(symbol)
    return f"RSI: {indicators['rsi']}, MACD: {indicators['macd']}, MA200: {indicators['ma200']}"

@tool
def predict_future(symbol: str, horizon: str = "1h") -> str:
    """Previsão com modelo Transformer+XGBoost."""
    pred = predict_price(symbol, horizon)
    return f"Previsão para {symbol} em {horizon}: ${pred['price']:.2f}, confiança {pred['confidence']:.2%}"

@tool
def calculate_risk(symbol: str, amount: float = 1000) -> str:
    """Calcula VaR e Sharpe ratio."""
    # Usar CAPM e volatilidade histórica
    var = 0.012 * amount   # exemplo
    sharpe = 2.1
    return f"VaR 1h: ${var:.2f}, Sharpe (30d): {sharpe}"

@tool
def optimize_portfolio(symbols: list, max_risk: float = 0.05) -> str:
    """Otimização de portfólio (Markowitz)."""
    # Chama modelo de otimização
    weights = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
    return f"Alocação ótima: {weights}"

@tool
def search_market_news(query: str) -> str:
    """Busca notícias relevantes via RAG (LlamaIndex + Pinecone)."""
    from app.rag.search import search_news
    results = search_news(query, top_k=3)
    return "\n".join([r["text"] for r in results])

all_tools = [get_live_price, get_indicators, predict_future, calculate_risk, optimize_portfolio, search_market_news]
```

### 6.2. Integração com modelos ML

- Salve os modelos treinados (Transformer+XGBoost) em um bucket S3 ou local.
- Crie uma função `predict_price` que carrega os modelos (com cache) e retorna previsão.
- Armazene features históricas no PostgreSQL (tabela `features`).

---

## 7. Memória Híbrida (Redis + Pinecone)

### 7.1. Memória de curto prazo (Redis)

```python
# app/memory/short_term.py
import redis
import json
from app.config import settings

redis_client = redis.from_url(settings.redis_url)

def add_message(user_id: str, role: str, content: str):
    key = f"chat:{user_id}"
    msg = {"role": role, "content": content}
    redis_client.rpush(key, json.dumps(msg))
    redis_client.ltrim(key, -20, -1)  # mantém últimas 20

def get_recent_messages(user_id: str, k: int = 5):
    key = f"chat:{user_id}"
    msgs = redis_client.lrange(key, -k, -1)
    return [json.loads(m) for m in msgs]
```

### 7.2. Memória de longo prazo (Pinecone)

Cada vez que o agente finaliza uma resposta importante, armazenamos um resumo no Pinecone:

```python
# app/memory/long_term.py
import pinecone
from langchain_openai import OpenAIEmbeddings
from app.config import settings

pinecone.init(api_key=settings.pinecone_api_key, environment="us-west1-gcp")
index = pinecone.Index(settings.pinecone_index_name)
embeddings = OpenAIEmbeddings()

def store_memory(user_id: str, text: str, metadata: dict):
    vector = embeddings.embed_query(text)
    index.upsert(vectors=[(f"{user_id}:{hash(text)}", vector, {**metadata, "user_id": user_id})])

def retrieve_memory(user_id: str, query: str, top_k=3):
    query_vec = embeddings.embed_query(query)
    results = index.query(vector=query_vec, top_k=top_k, filter={"user_id": user_id})
    return [match["metadata"]["text"] for match in results["matches"]]
```

---

## 8. RAG com LlamaIndex e Pinecone

### 8.1. Indexação de documentos (news, relatórios)

```python
# app/rag/indexer.py
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore
import pinecone
from app.config import settings

pinecone.init(api_key=settings.pinecone_api_key)
vector_store = PineconeVectorStore(index_name="crypto-news")

def index_news_documents(folder_path: str):
    documents = SimpleDirectoryReader(folder_path).load_data()
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    return index
```

### 8.2. Busca RAG para o agente

```python
# app/rag/search.py
from llama_index import VectorStoreIndex
from app.config import settings

def search_news(query: str, top_k=3):
    vector_store = PineconeVectorStore(index_name="crypto-news")
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [{"text": node.text, "score": node.score} for node in nodes]
```

---

## 9. Observabilidade com LangSmith

### 9.1. Configuração no backend

```python
# app/main.py
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from app.config import settings

tracer = LangChainTracer(project_name="crypto-agent")
client = Client(api_key=settings.langsmith_api_key)
```

No `graph.py`, adicione o tracer ao invocar o LLM:

```python
llm = ChatOpenAI(model="gpt-4-turbo", callbacks=[tracer])
```

### 9.2. Logging manual de ferramentas

```python
from langsmith import traceable

@traceable(name="get_live_price", run_type="tool")
def get_live_price(symbol: str):
    ...
```

---

## 10. Frontend – Next.js com Streaming de Raciocínio

### 10.1. Setup do projeto Next.js

```bash
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend
pnpm add eventsource-parser
```

### 10.2. Componente de chat com SSE (`app/components/ChatInterface.tsx`)

```tsx
"use client";
import { useState } from "react";

export default function ChatInterface() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{role:string, content:string}[]>([]);

  const sendMessage = async () => {
    setMessages(prev => [...prev, {role:"user", content:input}]);
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      body: JSON.stringify({ message: input, user_id: "user123" }),
      headers: { "Content-Type": "application/json" },
    });
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let reasoningText = "";
    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") {
            setMessages(prev => [...prev, {role:"assistant", content:reasoningText}]);
          } else {
            reasoningText += data + "\n";
            // Atualizar UI em tempo real
            setMessages(prev => {
              const last = prev[prev.length-1];
              if (last?.role === "assistant") {
                return [...prev.slice(0,-1), {...last, content: reasoningText}];
              } else {
                return [...prev, {role:"assistant", content: reasoningText}];
              }
            });
          }
        }
      }
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((m, i) => (
          <div key={i} className={`mb-2 ${m.role === "user" ? "text-right" : "text-left"}`}>
            <span className="inline-block p-2 rounded bg-gray-200">{m.content}</span>
          </div>
        ))}
      </div>
      <input
        className="border p-2 m-2"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
      />
    </div>
  );
}
```

### 10.3. Proxy API route (`frontend/app/api/chat/stream/route.ts`)

```ts
import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const backendRes = await fetch("http://localhost:8000/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return new Response(backendRes.body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}
```

---

## 11. Testes e Validação

### 11.1. Teste unitário de tools

```python
# tests/test_tools.py
from app.tools import get_live_price
def test_get_live_price():
    result = get_live_price("BTC")
    assert "$" in result
```

### 11.2. Teste do grafo com LangSmith

- Crie um dataset no LangSmith com perguntas típicas e respostas esperadas.
- Execute avaliação automática (ex: critério de corretude, uso correto de ferramentas).
- Ajuste prompts com base nos traces.

### 11.3. Teste de carga (k6 ou Locust)

- Simule 50 usuários simultâneos no endpoint `/chat/stream`.
- Monitore latência e uso de memória/CPU.

---

## 12. Deploy e Monitoramento

### 12.1. Opções de deploy

| Componente | Recomendação |
|------------|---------------|
| FastAPI + LangGraph | AWS ECS (Fargate) ou GCP Cloud Run (com aumento de timeout) |
| PostgreSQL | AWS RDS ou Neon.tech |
| Redis | Redis Cloud ou Upstash |
| Pinecone | Serviço gerenciado (já é) |
| Next.js | Vercel (ótimo para streaming) |
| Modelos ML | AWS SageMaker ou Lambda (com container) |

### 12.2. Variáveis de ambiente em produção

- Use AWS Secrets Manager ou GCP Secret Manager.
- Configure rate limiting (ex: `slowapi` no FastAPI).

### 12.3. Monitoramento pós-deploy

- **LangSmith**: análise de custos e qualidade das respostas.
- **Prometheus + Grafana**: métricas de API (req/s, latência, erros).
- **Logs centralizados**: Datadog ou ELK.

### 12.4. CI/CD (GitHub Actions)

```yaml
name: Deploy Backend
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        run: |
          docker build -t crypto-agent-backend .
          docker tag crypto-agent-backend ${{ secrets.ECR_REPO }}
          docker push ${{ secrets.ECR_REPO }}
      - name: Restart ECS service
        run: aws ecs update-service --cluster crypto --service backend --force-new-deployment
```

---

## Conclusão

Este plano cobre desde a configuração zero até um agente de investimento **que raciociona explicitamente**, chama APIs e modelos ML, mantém memória de longo prazo e entrega respostas transparentes via streaming. O uso de LangGraph permite um controle fino sobre o processo de pensamento, essencial para um assistente financeiro confiável.

**Próximos passos imediatos**:
1. Configurar Docker Compose com PostgreSQL e Redis.
2. Implementar uma tool simples (`get_live_price`) e testar o grafo básico.
3. Conectar o frontend Next.js com streaming.
4. Adicionar a memória híbrida e RAG.

Precisa de ajuda para implementar algum módulo específico? Posso fornecer código detalhado de qualquer parte.
