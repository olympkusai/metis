# Fases de Implementação – Agente de Investimento Cripto

Este documento divide o plano de implementação em fases sequenciais para facilitar o desenvolvimento e acompanhamento do progresso.

---

## Fase 1: Setup Inicial e Infraestrutura

**Objetivo**: Configurar ambiente básico e serviços de infraestrutura

### Tarefas
- [x] Criar estrutura de pastas do projeto
- [x] Configurar Docker Compose (PostgreSQL, Redis)
- [x] Criar arquivo `.env` com variáveis de ambiente
- [x] Configurar Poetry no backend
- [ ] Criar contas/APIs necessárias (OpenAI, LangFuse, provedor de dados cripto)

### Entregáveis
- Estrutura de pastas completa
- Docker Compose funcional com PostgreSQL e Redis rodando
- Arquivo `.env` configurado
- Projeto backend (Poetry) inicializado

### Critérios de Sucesso
- `docker-compose up` inicia PostgreSQL e Redis sem erros
- Variáveis de ambiente carregadas corretamente
- Poetry instalado e funcionando

---

## Fase 2: Backend Básico (FastAPI + LangGraph)

**Objetivo**: Implementar backend com endpoint de chat básico e agente LangGraph simples

### Tarefas
- [x] Criar `app/config.py` com configurações
- [x] Implementar endpoint `/chat` básico em `app/api/chat.py`
- [x] Criar estrutura modular de estado (`app/agent/state/`)
- [x] Criar estrutura modular de prompts (`app/agent/prompts/`)
- [x] Criar grafo avançado com 9 nós em `app/agent/graph/`
- [x] Implementar engine quant (`app/agent/quant_engine/`)
- [x] Implementar portfolio (`app/agent/portfolio/`)
- [x] Implementar execution (`app/agent/execution/`)
- [x] Implementar tools (`get_live_price`, `get_indicators`, `calculate_risk`, `get_feature_rsi`, `get_feature_macd`, `get_feature_bollinger`, `get_feature_volatility`, `get_feature_sharpe`, `get_feature_cvar`, `get_feature_max_drawdown`, `get_feature_sma`, `get_feature_ema_return`, `get_ohlcv_history`)
- [ ] Testar endpoint `/chat` com curl/Postman

### Entregáveis
- Configuração centralizada em `config.py`
- Endpoint `/chat` funcional
- Grafo LangGraph com 4 nós implementados
- Tool `get_live_price` funcionando

### Critérios de Sucesso
- Endpoint `/chat` responde com mensagem do agente
- Grafo executa o ciclo reason → action → observe → finalize
- Tool `get_live_price` retorna preço atual de uma criptomoeda

---

## Fase 3: Implementação das Ferramentas (Tools)

**Objetivo**: Desenvolver todas as ferramentas de dados e ML

### Tarefas
- [x] Implementar `get_indicators` (RSI, MACD, médias, OHLCV)
- [x] Implementar `calculate_risk` (VaR, CVaR, Sharpe, drawdown)
- [x] Implementar integração com API externa (k0s.app) via HTTP REST
- [ ] Implementar `predict_future` (integração com modelo ML)
- [ ] Implementar `optimize_portfolio` (Markowitz)
- [ ] Criar modelos ML (Transformer + XGBoost)
- [ ] Configurar job para calcular indicadores e armazenar no PostgreSQL
- [ ] Criar tabela de features no PostgreSQL

### Entregáveis
- Todas as 6 tools implementadas e funcionando
- Modelos ML treinados e salvos
- Job de cálculo de indicadores rodando periodicamente
- Tabela de features populada

### Critérios de Sucesso
- Cada tool retorna dados válidos
- Modelos ML carregam e fazem previsões
- Indicadores técnicos são calculados e armazenados

---

## Fase 4: Memória Híbrida (Redis + ChromaDB)

**Objetivo**: Implementar sistema de memória de curto e longo prazo

### Tarefas
- [ ] Configurar ChromaDB (criar coleções)
- [ ] Implementar `app/memory/short_term.py` (Redis)
- [ ] Implementar `app/memory/long_term.py` (ChromaDB)
- [ ] Criar função `get_memory_context` para o agente
- [ ] Integrar memória no nó de raciocínio
- [ ] Testar armazenamento e recuperação de memórias

### Entregáveis
- Memória de curto prazo funcional (últimas 20 mensagens)
- Memória de longo prazo funcional (ChromaDB)
- Contexto de memória injetado no prompt do agente
- Resumos de conversas importantes armazenados

### Critérios de Sucesso
- Redis armazena e recupera mensagens recentes
- ChromaDB armazena e recupera memórias por similaridade
- Agente usa memória de longo prazo em raciocínios

---

## Fase 5: RAG com ChromaDB

**Objetivo**: Implementar busca de notícias e documentos via RAG

### Tarefas
- [ ] Configurar coleção ChromaDB para documentos
- [ ] Implementar `app/rag/indexer.py` (indexação de documentos)
- [ ] Implementar `app/rag/search.py` (busca RAG)
- [ ] Integrar tool `search_market_news` com RAG
- [ ] Criar pipeline de ingestão de notícias
- [ ] Testar busca de notícias relevantes

### Entregáveis
- Coleção ChromaDB para documentos configurado
- Pipeline de indexação funcional
- Tool `search_market_news` usando RAG
- Notícias indexadas e recuperáveis

### Critérios de Sucesso
- Documentos são indexados no ChromaDB
- Busca retorna notícias relevantes baseadas na query
- Agente usa notícias em suas análises

---

## Fase 6: Testes e Validação

**Objetivo**: Garantir qualidade e robustez do sistema

### Tarefas
- [ ] Escrever testes unitários para todas as tools
- [ ] Escrever testes para o grafo LangGraph
- [ ] Criar dataset de avaliação no LangFuse
- [ ] Executar avaliação automática no LangFuse
- [ ] Implementar teste de carga (k6 ou Locust)
- [ ] Ajustar prompts com base nos traces
- [ ] Corrigir bugs encontrados

### Entregáveis
- Suíte de testes unitários completa
- Dataset de avaliação criado
- Relatório de avaliação LangFuse
- Relatório de teste de carga
- Prompts otimizados

### Critérios de Sucesso
- Todos os testes unitários passam
- Avaliação LangFuse mostra >80% de acurácia
- Sistema suporta 50 usuários simultâneos
- Latência média <2s

---

## Fase 7: Observabilidade (LangFuse)

**Objetivo**: Configurar monitoramento e tracing completo

### Tarefas
- [ ] Configurar LangFuse no backend
- [ ] Adicionar callback ao LLM no grafo
- [ ] Adicionar tracing manual nas tools
- [ ] Configurar logging de custos
- [ ] Criar dashboards no LangFuse
- [ ] Testar tracing de uma conversa completa

### Entregáveis
- LangFuse configurado e funcionando
- Traces visíveis no dashboard
- Custos monitorados
- Logs de ferramentas rastreados

### Critérios de Sucesso
- Cada execução do agente é traceada
- Custos são calculados e exibidos
- Dashboards mostram métricas relevantes

---

## Fase 8: Deploy e Monitoramento

**Objetivo**: Deploy em produção e monitoramento contínuo

### Tarefas
- [ ] Escolher provedores (AWS/GCP)
- [ ] Configurar deploy do FastAPI (ECS/Cloud Run)
- [ ] Configurar PostgreSQL gerenciado (RDS/Neon)
- [ ] Configurar Redis gerenciado (Redis Cloud/Upstash)
- [ ] Deploy de modelos ML (SageMaker/Lambda)
- [ ] Configurar Secrets Manager
- [ ] Implementar rate limiting
- [ ] Configurar Prometheus + Grafana
- [ ] Configurar logs centralizados
- [ ] Criar pipeline CI/CD (GitHub Actions)
- [ ] Testar deploy completo

### Entregáveis
- Sistema em produção
- CI/CD configurado
- Monitoramento funcionando
- Secrets gerenciadas

### Critérios de Sucesso
- Sistema acessível em produção
- Deploy automático via CI/CD
- Métricas visíveis no Grafana
- Logs centralizados funcionando

---

## Fase 9: Documentação e Handoff

**Objetivo**: Documentar sistema e preparar para manutenção

### Tarefas
- [ ] Documentar arquitetura
- [ ] Documentar APIs
- [ ] Criar guia de setup para desenvolvedores
- [ ] Documentar processo de deploy
- [ ] Criar runbooks para operações
- [ ] Documentar modelos ML
- [ ] Revisar e finalizar README

### Entregáveis
- Documentação completa
- Guia de setup
- Runbooks
- README atualizado

### Critérios de Sucesso
- Novo desenvolvedor consegue setup em <1h
- Operações são documentadas
- Sistema é maintenível

---

## Resumo das Fases

| Fase | Nome | Duração Estimada | Dependências |
|------|------|------------------|---------------|
| 1 | Setup Inicial | 1-2 dias | - |
| 2 | Backend Básico | 3-4 dias | 1 |
| 3 | Ferramentas | 5-7 dias | 2 |
| 4 | Memória Híbrida | 3-4 dias | 2 |
| 5 | RAG | 3-4 dias | 4 |
| 6 | Testes | 3-4 dias | 3,4,5 |
| 7 | Observabilidade | 2-3 dias | 2 |
| 8 | Deploy | 4-5 dias | 6,7 |
| 9 | Documentação | 2-3 dias | 8 |

**Total estimado**: 25-35 dias

---

## Notas Importantes

- As fases 4 e 5 podem ser desenvolvidas em paralelo após a fase 2
- A fase 7 deve ser feita em paralelo com as fases 3-6 para máximo benefício
- A fase 6 (testes) deve ser contínua, não apenas no final
- Ajuste durações baseado no tamanho da equipe e experiência
