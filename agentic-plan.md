# Plano de Execução — Transformação Agentic do Metis

> Transformar o Metis de um pipeline LangGraph fixo em um agente com loop ReAct livre, context management, permission gates e coordenação multi-agente (cowork).

---

## Contexto

### Estado atual

O Metis é um assistente de finanças pessoais (FastAPI + LangGraph) com:

- **Pipeline fixo** (DAG): `orchestrator → context → [action | data_gathering → analysis → synthesis] → finalize`
- **Tool-use loop interno** por nó (`_run_agent_loop`, máx 6 iterações) — mas o fluxo entre nós é hardcoded
- **6 tools de leitura** (Pluto) + **tools dinâmicas de escrita** (Hermes via MCP)
- **Streaming SSE** com chain-of-thought
- **Persistência** de conversas em Postgres (db-metis)
- **Personalização** via Soter (tom, idioma, moeda, obfuscação)
- **Cost tracking** de tokens

### Problema central

O LLM não controla o fluxo. Ele não pode decidir "voltei e preciso coletar mais dados" ou "executei uma ação, agora quero analisar o resultado". O roteamento (action vs analysis) é determinístico, não emergente. Isso limita:

- Respostas que precisam de múltiplas rodadas de coleta + ação
- Cenários onde a análise revela que faltam dados
- Coordenação de tarefas paralelas (cowork)

### Objetivo

Transformar o Metis em um agente onde o LLM decide o fluxo — qual tool chamar, quando parar, quando pedir permissão, quando delegar para um subagent — mantendo a especialização de domínio (finanças) e a integração com Pluto/Hermes/Soter.

---

## Arquitetura Alvo

```
                        ┌─────────────────────────────────┐
                        │         Agent Runtime           │
                        │  (loop ReAct genérico)          │
                        │                                 │
                        │  while not done:                │
                        │    response = LLM(              │
                        │      system_prompt,             │
                        │      tools_catalog,             │
                        │      context_manager,           │
                        │      history                    │
                        │    )                            │
                        │    if response.tool_calls:      │
                        │      → permission_gate          │
                        │      → execute (paralelo)       │
                        │      → context.append(results)  │
                        │    else:                        │
                        │      → return response          │
                        └──────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐
     │  Tools Catalog  │  │   Context    │  │   Subagents      │
     │                 │  │   Manager    │  │   (cowork)       │
     │ • finance_read  │  │              │  │                  │
     │ • finance_write │  │ • truncate   │  │ • spawn_agent    │
     │   (Hermes MCP)  │  │ • summarize  │  │ • parallel exec  │
     │ • memory_recall │  │ • evict      │  │ • aggregate      │
     │ • plan          │  │ • embed      │  │ • shared state   │
     │ • todo          │  │              │  │                  │
     │ • spawn_subagent│  │              │  │                  │
     └─────────────────┘  └──────────────┘  └──────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐
     │  Pluto / Hermes │  │  db-metis    │  │  Agent Runtime   │
     │  / Soter        │  │  (pgvector)  │  │  (recursão)      │
     └─────────────────┘  └──────────────┘  └──────────────────┘
```

### Princípios de design

1. **Loop livre sobre pipeline fixo** — o LLM decide o fluxo, não o grafo
2. **Tools como cidadãos de primeira classe** — catálogo unificado, não particionado por nó
3. **Contexto é gerenciado, não acumulado** — truncar, sumarizar, evictar
4. **Permissões explícitas** — operações destrutivas exigem gate
5. **Especialização via prompt, não via grafo** — o domínio (finanças) vive no system prompt, não na topologia
6. **Subagents para paralelismo** — cowork emerge de `spawn_subagent` como tool

---

## Fases de Execução

### Fase 0 — Preparação e baseline

**Objetivo**: Garantir que temos testes cobrindo o comportamento atual antes de refatorar.

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 0.1 | Auditoria de testes existentes | Rodar `poetry run python -m pytest` e mapear cobertura atual | 296 testes passando, cobertura documentada |
| 0.2 | Testes de ponta-a-ponta do pipeline atual | Criar testes que exercitam o grafo completo (orchestrator → finalize) com mocks de Pluto/Soter/Hermes | Pelo menos 5 cenários: greeting, out-of-scope, task simples, action, task com múltiplas tools |
| 0.3 | Snapshot de comportamento | Gravar respostas de exemplo para queries canônicas ("quanto gastei este mês?", "cria uma transação de R$50") | Baseline salvo em `tests/fixtures/baseline_responses.json` |

**Dependências**: nenhuma
**Duração estimada**: 1-2 dias

---

### Fase 1 — Extração do loop ReAct genérico

**Objetivo**: Transformar `_run_agent_loop` em um loop agentic genérico onde o LLM escolhe tools livremente, sem o pipeline fixo.

#### 1A — Refatorar `_run_agent_loop` para loop genérico

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 1A.1 | Extrair `AgentRuntime` class | Mover `_run_agent_loop` para `metis/agent/runtime.py` como uma classe `AgentRuntime` com método `async def run()`. Estado interno: messages, tool_cache, steps, reasoning_lines | `AgentRuntime` instanciável com system_prompt + tools + context |
| 1A.2 | Unificar catálogo de tools | Criar `build_tool_catalog()` que junta: finance_tools (leitura) + hermes_tools (escrita, via MCP) + tools genéricas futuras. O LLM recebe todas e escolhe | LLM pode chamar `get_cashflow` e `create_transaction` no mesmo loop |
| 1A.3 | Remover `tool_map_override` | Hoje cada nó restringe tools via `tool_map_override`. No loop genérico, todas as tools estão disponíveis | `tool_map_override` removido; catálogo unificado |
| 1A.4 | Aumentar `MAX_ITERATIONS` | 6 iterações é insuficiente para loop livre. Aumentar para 15-20 com detecção de no-progress mais robusta | Loop roda até 20 iterações sem travar |
| 1A.5 | Detecção de no-progress melhorada | Hoje conta `len(steps)`. Melhorar: detectar repetição de mesmas tool_calls, loops de tool A → tool B → tool A | Loop aborta se detectar padrão repetido após 3 ciclos |

**Arquivo alvo**: `metis/agent/runtime.py` (novo)
**Arquivo afetado**: `metis/agent/graph.py` (delegar para `AgentRuntime`)
**Dependências**: Fase 0
**Duração estimada**: 2-3 dias

#### 1B — Substituir StateGraph por loop único

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 1B.1 | Criar `finance_agent_v2` | Novo grafo LangGraph com **um único nó** que invoca `AgentRuntime.run()`. Sem orchestrator, context, data_gathering, analysis, synthesis separados | Grafo v2 tem 1 nó + finalize |
| 1B.2 | System prompt unificado | Combinar os prompts de orchestrator + data_gathering + analysis + synthesis em um único system prompt que descreve: domínio (finanças), tools disponíveis, quando usar leitura vs escrita, formato de resposta | Prompt em `metis/agent/finance_prompts.py` como `_FINANCE_AGENT_V2_SYSTEM` |
| 1B.3 | Injeção de contexto de perfil | O que `finance_context_node` fazia (buscar perfil + contas + personalização no Pluto/Soter) passa a ser uma **tool** (`get_user_profile`) ou um **pre-step** antes do loop | Perfil disponível no contexto do loop |
| 1B.4 | Feature flag para v2 | Adicionar `AGENT_VERSION=v1|v2` em config. v1 = pipeline fixo atual, v2 = loop ReAct | Ambos funcionam; v2 behind flag |
| 1B.5 | Testes A/B | Rodar queries canônicas da Fase 0 contra v1 e v2, comparar qualidade | v2 não regrediu em nenhum cenário canônico |

**Arquivo alvo**: `metis/agent/graph_v2.py` (novo)
**Arquivo afetado**: `metis/config.py` (flag), `metis/api/chat.py` (selecionar versão)
**Dependências**: 1A
**Duração estimada**: 3-4 dias

---

### Fase 2 — Context Management

**Objetivo**: Gerenciar o contexto do LLM para suportar conversas longas e múltiplas tool calls sem estourar a janela.

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 2.1 | Token counter | Integrar `tiktoken` para contar tokens do contexto antes de cada chamada LLM | `context_manager.token_count()` retorna número preciso |
| 2.2 | Truncagem de tool results | Hoje já trunca em `result[:3000]`. Melhorar: truncagem baseada em tokens (ex: máx 2000 tokens por tool result), com indicação `[truncado, N chars omitidos]` | Tool results não excedem limite configurável |
| 2.3 | Sumarização de histórico | Quando histórico + contexto > 80% da janela, sumarizar mensagens antigas em um bloco `[RESUMO DA CONVERSA]` via LLM barato (gpt-4o-mini) | Conversas longas não estouram janela; resumo preserva fatos chave |
| 2.4 | Eviction de reasoning_trail | O `reasoning_trail` cresce a cada nó. No loop livre, limitar a últimas 5 entradas + sumário das anteriores | Trail não excede 5 entradas + sumário |
| 2.5 | Context window budget | Definir orçamento: system_prompt (fixo) + tools_schema (fixo) + history (gerenciado) + tool_results (gerenciado) + margem para resposta | Budget respeitado em cada iteração do loop |

**Arquivo alvo**: `metis/agent/context_manager.py` (novo)
**Arquivo afetado**: `metis/agent/runtime.py` (integrar context_manager)
**Dependências**: Fase 1
**Duração estimada**: 2-3 dias

---

### Fase 3 — Permission Gates

**Objetivo**: Operações destrutivas (escrita no Pluto via Hermes) exigem confirmação explícita antes de executar.

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 3.1 | Classificação de tools | Marcar cada tool como `read` ou `write`. Finance tools = read. Hermes tools = write (classificar via metadata do MCP ou convenção de nome) | `tool.metadata["destructive"] = True/False` |
| 3.2 | Gate no runtime | Antes de executar tool marcada como `write`, o runtime pausa e emite um evento `permission_request` com: tool_name, args, descrição human-readable | Tools de escrita não executam sem aprovação |
| 3.3 | API de aprovação | Endpoint `POST /api/chat/permissions/{request_id}` para o frontend aprovar/rejeitar. O SSE stream inclui o `permission_request` | Frontend recebe evento, usuário aprova, runtime continua |
| 3.4 | Timeout de permissão | Se não houver aprovação em N segundos (configurável, default 120s), abortar a tool com mensagem amigável | Timeout não trava o loop |
| 3.5 | Auto-aprovação para modo não-interativo | Em requests não-streaming (`POST /api/chat`), decidir política: bloquear writes ou auto-aprovar com log de auditoria | Política documentada e configurável |

**Arquivo alvo**: `metis/agent/permissions.py` (novo)
**Arquivo afetado**: `metis/agent/runtime.py` (chamar gate), `metis/api/chat.py` (endpoint de aprovação)
**Dependências**: Fase 1
**Duração estimada**: 2-3 dias

---

### Fase 4 — Plan Mode

**Objetivo**: Modo onde o agente explora (leitura), formula plano, apresenta ao usuário, e só executa após aprovação.

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 4.1 | Tool `create_plan` | Tool que o LLM chama para estruturar um plano: lista de steps com tool + args esperadas. Armazena no estado | LLM pode formular e armazenar plano |
| 4.2 | Tool `get_plan_approval` | Tool que pausa o loop e emite evento `plan_approval_request` com o plano. Usuário aprova/rejeita/modifica via API | Plano não executa sem aprovação |
| 4.3 | Execução guiada por plano | Após aprovação, o loop executa os steps do plano sequencialmente, com permissão já concedida | Steps executam na ordem aprovada |
| 4.4 | Plano rejeitado | Se usuário rejeita, LLM recebe feedback e pode reformular ou abandonar | Loop reage à rejeição sem travar |
| 4.5 | Toggle de plan mode | `plan_mode: bool` no request body. Se True, agente deve formular plano antes de qualquer escrita | Plan mode ativável por request |

**Arquivo alvo**: `metis/agent/plan.py` (novo)
**Arquivo afetado**: `metis/agent/runtime.py` (tools de plano), `metis/api/chat.py` (endpoint de aprovação de plano)
**Dependências**: Fase 3
**Duração estimada**: 3-4 dias

---

### Fase 5 — Long-term Memory (RAG)

**Objetivo**: Recall semântico de conversas e fatos anteriores do usuário.

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 5.1 | Reativar pgvector | `CREATE EXTENSION vector` + `ALTER COLUMN embedding TYPE vector(1536)` + índice ivfflat em `chat_messages` | Extensão ativa, coluna tipada |
| 5.2 | Embedding no save | Ao salvar mensagem em `save_message`, gerar embedding via `text-embedding-3-small` e armazenar | Mensagens novas têm embedding |
| 5.3 | Backfill de históricos | Script para embeddar mensagens existentes (batch) | Mensagens antigas embeddadas |
| 5.4 | Tool `recall_memory` | Tool que o LLM chama com query → busca por similaridade em `chat_messages` do usuário → retorna top-K mensagens relevantes | LLM pode recuperar contexto de conversas antigas |
| 5.5 | Injeção automática | Opcional: antes do loop, buscar memórias relevantes à query atual e injetar como contexto | Memória relevante aparece no contexto sem tool call explícita |

**Arquivo alvo**: `metis/agent/memory_rag.py` (novo), `metis/storage/migrations.py` (pgvector)
**Arquivo afetado**: `metis/memory/conversation_history.py` (embedding no save), `metis/agent/runtime.py` (tool recall_memory)
**Dependências**: Fase 1
**Duração estimada**: 3-4 dias

---

### Fase 6 — Subagents e Cowork

**Objetivo**: Permitir que o agente spawn subagentes para tarefas paralelas (cowork).

#### 6A — Infraestrutura de subagents

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 6A.1 | Tool `spawn_subagent` | Tool que cria um novo `AgentRuntime` com: prompt próprio, subconjunto de tools, contexto isolado. Retorna agent_id | Subagent instanciado e rodando |
| 6A.2 | Execução paralela | Subagents rodam via `asyncio.gather`. O agente pai pode spawnar N subagents e esperar todos | Múltiplos subagents rodando simultaneamente |
| 6A.3 | Tool `get_subagent_result` | Tool que bloqueia até subagent finalizar e retorna seu resultado | Pai recebe resultado do subagent |
| 6A.4 | Compartilhamento de estado | Subagents recebem snapshot do estado do pai (perfil, contas, reasoning_trail). Resultados voltem para o pai via return | Subagent tem contexto necessário; pai recebe output |
| 6A.5 | Limites de recursão | Máx 3 níveis de subagents. Máx 5 subagents simultâneos por agente pai | Não há explosão de agentes |

**Arquivo alvo**: `metis/agent/subagent.py` (novo)
**Arquivo afetado**: `metis/agent/runtime.py` (tools de spawn)
**Dependências**: Fase 1, Fase 2
**Duração estimada**: 4-5 dias

#### 6B — Padrões de cowork

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 6B.1 | Padrão "divide and conquer" | System prompt instrui o agente a decompor tarefas complexas em subtasks e spawnar subagents | Tarefa complexa é decomposta e executada em paralelo |
| 6B.2 | Padrão "specialist delegation" | Subagents com personas diferentes: `analyst` (só leitura + análise), `executor` (só escrita via Hermes), `researcher` (RAG + web) | Subagents têm tools/prompts especializados |
| 6B.3 | Agregação de resultados | Após subagents finalizarem, agente pai sintetiza resultados em resposta coerente | Resposta final integra outputs de múltiplos subagents |
| 6B.4 | Tratamento de falhas de subagent | Se um subagent falha, pai decide: retry, reformular, ou prosseguir com resultado parcial | Falha de subagent não derruba o agente pai |

**Arquivo alvo**: `metis/agent/cowork_patterns.py` (novo)
**Arquivo afetado**: `metis/agent/finance_prompts.py` (instruções de cowork no system prompt)
**Dependências**: 6A
**Duração estimada**: 3-4 dias

---

### Fase 7 — Hooks e Extensibilidade

**Objetivo**: Callbacks pre/post tool e eventos do ciclo de vida do agente.

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 7.1 | Definir eventos | Eventos: `pre_tool`, `post_tool`, `pre_loop_iteration`, `post_loop_iteration`, `on_permission_request`, `on_error` | Eventos documentados |
| 7.2 | Registro de hooks | Config `agent.hooks` em settings: lista de callables (async) por evento | Hooks registráveis via config |
| 7.3 | Hooks built-in | Logging hook (post_tool), audit hook (post_tool para writes), rate-limit hook (pre_tool) | Hooks padrão funcionando |
| 7.4 | Skills/instructions dinâmicas | Carregar arquivos `.md` de `metis/skills/` e injetar no system_prompt sob demanda (similar a skills do Devin/Claude Code) | Skill carregada modifica comportamento do agente |

**Arquivo alvo**: `metis/agent/hooks.py` (novo), `metis/skills/` (diretório)
**Arquivo afetado**: `metis/agent/runtime.py` (dispatch de hooks), `metis/config.py` (config de hooks)
**Dependências**: Fase 1
**Duração estimada**: 2-3 dias

---

### Fase 8 — Observabilidade e Avaliação

**Objetito**: Tracing completo do loop agentic para debug e melhoria contínua.

| # | Tarefa | Detalhe | Critério de sucesso |
|---|--------|---------|---------------------|
| 8.1 | Trace estruturado | Cada iteração do loop: timestamp, tool_calls, args, results, tokens, latency. Salvar em `agent_traces` no db-metis | Trace completo de cada execução |
| 8.2 | LangFuse integration | Enviar traces para LangFuse (callback do LangChain + tracing manual) | Traces visíveis no dashboard LangFuse |
| 8.3 | Dataset de avaliação | Curar 50+ queries canônicas com respostas esperadas em `tests/fixtures/eval_dataset.json` | Dataset versionado |
| 8.4 | Avaliação automática | Script que roda dataset contra o agente e pontua via LLM-as-judge (gpt-4o-mini) | Score > 80% no dataset |
| 8.5 | Dashboard de métricas | Latência média, tokens/request, tool calls/request, taxa de sucesso, taxa de permissão negada | Métricas acessíveis |

**Arquivo alvo**: `metis/agent/tracing.py` (novo), `metis/storage/migrations.py` (tabela agent_traces)
**Arquivo afetado**: `metis/agent/runtime.py` (emitir traces)
**Dependências**: Fase 1
**Duração estimada**: 3-4 dias

---

## Resumo de Fases

| Fase | Nome | Duração | Dependências | Prioridade |
|------|------|---------|--------------|------------|
| 0 | Preparação e baseline | 1-2 dias | — | Crítica |
| 1 | Loop ReAct genérico | 5-7 dias | 0 | Crítica |
| 2 | Context management | 2-3 dias | 1 | Alta |
| 3 | Permission gates | 2-3 dias | 1 | Alta |
| 4 | Plan mode | 3-4 dias | 3 | Média |
| 5 | Long-term memory (RAG) | 3-4 dias | 1 | Média |
| 6 | Subagents e cowork | 7-9 dias | 1, 2 | Média |
| 7 | Hooks e extensibilidade | 2-3 dias | 1 | Baixa |
| 8 | Observabilidade | 3-4 dias | 1 | Alta |

**Total estimado**: 28-39 dias

### Ordem recomendada de execução

```
Fase 0 (baseline)
  ↓
Fase 1 (loop ReAct) ← bloqueante para tudo
  ↓
  ├─ Fase 2 (context)  ─┐
  ├─ Fase 3 (perms)    ─┤
  └─ Fase 8 (tracing)  ─┘  ← paralelizáveis
         ↓
    Fase 4 (plan mode) ← depende de 3
    Fase 5 (RAG)       ← independente
         ↓
    Fase 6 (cowork)    ← depende de 1, 2
         ↓
    Fase 7 (hooks)     ← último, nice-to-have
```

Fases 2, 3 e 8 podem ser desenvolvidas em paralelo após a Fase 1.

---

## Estrutura de Arquivos Alvo

```
metis/
├── metis/
│   ├── agent/
│   │   ├── runtime.py          # NOVO — AgentRuntime (loop ReAct genérico)
│   │   ├── context_manager.py  # NOVO — token counting, truncagem, sumarização
│   │   ├── permissions.py      # NOVO — gates para tools destrutivas
│   │   ├── plan.py             # NOVO — plan mode (create_plan, get_approval)
│   │   ├── subagent.py         # NOVO — spawn_subagent, execução paralela
│   │   ├── cowork_patterns.py  # NOVO — padrões de delegação multi-agente
│   │   ├── hooks.py            # NOVO — callbacks pre/post evento
│   │   ├── tracing.py          # NOVO — trace estruturado do loop
│   │   ├── memory_rag.py       # NOVO — recall semântico via pgvector
│   │   ├── graph.py            # EXISTENTE — v1 (pipeline fixo, mantido)
│   │   ├── graph_v2.py         # NOVO — v2 (loop único com AgentRuntime)
│   │   ├── finance_prompts.py  # EXISTENTE — + _FINANCE_AGENT_V2_SYSTEM
│   │   └── schemas.py          # EXISTENTE (se aplicável)
│   ├── tools/
│   │   ├── finance.py          # EXISTENTE — tools de leitura (Pluto)
│   │   ├── agent_tools.py      # NOVO — tools meta (create_plan, spawn_subagent, recall_memory)
│   │   └── __init__.py         # EXISTENTE — + build_tool_catalog()
│   ├── skills/                 # NOVO — instruções injetáveis (.md)
│   │   └── README.md
│   ├── storage/
│   │   ├── migrations.py       # EXISTENTE — + pgvector, agent_traces
│   │   └── ...
│   └── ...
├── tests/
│   ├── fixtures/
│   │   ├── baseline_responses.json  # NOVO — snapshot v1
│   │   └── eval_dataset.json        # NOVO — dataset de avaliação
│   ├── agent/
│   │   ├── test_runtime.py          # NOVO
│   │   ├── test_context_manager.py  # NOVO
│   │   ├── test_permissions.py      # NOVO
│   │   ├── test_plan.py             # NOVO
│   │   ├── test_subagent.py         # NOVO
│   │   └── test_cowork.py           # NOVO
│   └── ...
└── agentic-plan.md  # ESTE ARQUIVO
```

---

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| v2 (loop livre) regredir vs v1 (pipeline fixo) em queries simples | Alto | Feature flag + testes A/B (Fase 0, 1B.5) |
| Custo de tokens aumentar com loop livre (mais iterações) | Médio | Context management (Fase 2) + limites de iteração + modelo barato para steps intermediários |
| LLM chamar tools de escrita sem necessidade | Alto | Permission gates (Fase 3) + system prompt claro |
| Subagents causarem recursão infinita | Médio | Limites de recursão e concorrência (6A.5) |
| pgvector não disponível no Railway | Baixo | Verificar antes da Fase 5; fallback para embeddings em memória |
| Latência aumentando com tracing | Baixo | Tracing assíncrono (fire-and-forget) |

---

## Decisões de Design Pendentes

Estas decisões devem ser tomadas antes/início de cada fase:

1. **Fase 1B**: O `finance_context` (buscar perfil no Pluto) vira tool ou pre-step?
   - **Tool**: mais alinhado com loop livre, mas adiciona 1 iteração sempre
   - **Pre-step**: mais rápido, mas quebra o princípio de "loop puro"
   - **Recomendação**: pre-step para perfil (sempre precisa), tool para dados opcionais

2. **Fase 3**: Em modo não-interativo (`POST /api/chat` sem SSE), writes são bloqueados ou auto-aprovados?
   - **Bloqueados**: mais seguro, mas limita uso via API
   - **Auto-aprovados**: mais útil, mas risco de ações indesejadas
   - **Recomendação**: bloqueados por default, flag `allow_writes=true` para auto-aprovar com audit log

3. **Fase 5**: Embedding model — `text-embedding-3-small` (barato, 1536 dims) vs `text-embedding-3-large` (melhor, 3072 dims)?
   - **Recomendação**: small para começar, upgradar se recall for insuficiente

4. **Fase 6**: Subagents compartilham a mesma conexão MCP/Hermes ou abrem conexões próprias?
   - **Compartilhada**: mais eficiente, mas race conditions
   - **Própria**: mais isolada, mas overhead
   - **Recomendação**: própria, com pool de conexões

---

## Critério de Sucesso Global

O plano é considerado completo quando:

1. **Loop agentic funcional**: agente v2 resolve tarefas que exigem múltiplas rodadas de coleta + ação sem travar
2. **Sem regressão**: v2 passa em 100% dos cenários canônicos da Fase 0
3. **Contexto gerenciado**: conversas de 50+ mensagens não estouram janela
4. **Permissões funcionando**: nenhuma tool de escrita executa sem aprovação em modo interativo
5. **Cowork demonstrável**: pelo menos 1 cenário onde subagents paralelos resolvem uma tarefa mais rápido que sequencial
6. **Observabilidade**: cada execução tem trace completo visível em LangFuse
7. **Avaliação automatizada**: score > 80% no dataset de avaliação
