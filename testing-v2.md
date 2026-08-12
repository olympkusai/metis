# Guia de Testes — Metis v2 (ReAct Loop)

> Como validar o agente v2, comparar com v1, e encontrar problemas antes dos usuários.

---

## Como usar este guia

1. **Ative v2**: `railway variable set AGENT_VERSION=v2` (já feito)
2. **Aguarde o deploy** terminar: `railway deployment list`
3. **Para cada teste**: envie a query via frontend OU curl (abaixo)
4. **Anote o resultado**: o que respondeu, quanto tempo levou, se chamou tools certas
5. **Compare com v1**: troque para `AGENT_VERSION=v1`, repita a mesma query, compare
6. **Marque passou/falhou** na coluna à direita

### Endpoint para testes via curl

```bash
# Não-streaming (mais fácil de ler a resposta completa)
curl -X POST https://api.olympkusai.com/metis/api/chat \
  -H "Authorization: Bearer <SEU_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"message": "SUA_PERGUNTA_AQUI"}'

# Streaming (para ver tokens em tempo real + eventos SSE)
curl -N -X POST https://api.olympkusai.com/metis/api/streaming/chat \
  -H "Authorization: Bearer <SEU_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"message": "SUA_PERGUNTA_AQUI"}'
```

> Substitua `<SEU_JWT>` por um token válido do Soter. Pegue do frontend (DevTools → Network → Authorization header).

---

## Categoria 1: Saudações e fora de escopo

**O que validar**: o agente v2 deve responder saudações sem chamar tools e recusar educadamente assuntos não-financeiros. No v1 isso era determinístico (regex + LLM classifier). No v2 é o LLM decidindo sozinho.

| # | Query | Comportamento esperado | Red flags | Passou? |
|---|-------|----------------------|-----------|---------|
| 1.1 | "oi" | Saudação amigável, lista o que pode ajudar | Chamou tools desnecessárias | ☐ |
| 1.2 | "bom dia" | Saudação contextual (bom dia) | — | ☐ |
| 1.3 | "eae mano, tudo bem?" | Saudação casual | Resposta formal/rígida | ☐ |
| 1.4 | "qual o sentido da vida?" | Recusa educada, explica escopo | Tentou responder sobre filosofia | ☐ |
| 1.5 | "me escreve um poema" | Recusa, redireciona para finanças | Escreveu o poema | ☐ |
| 1.6 | "quanto é 2+2?" | Recusa OU responde rápido sem tools | Chamou tools financeiras | ☐ |

---

## Categoria 2: Consultas simples (uma tool)

**O que validar**: o agente deve chamar a tool de leitura correta, receber os dados, e responder em Markdown formatado. Cada query abaixo deve exercitar uma tool específica.

| # | Query | Tool esperada | Comportamento esperado | Red flags | Passou? |
|---|-------|---------------|----------------------|-----------|---------|
| 2.1 | "quanto gastei este mês?" | get_spending_by_category | Lista gastos por categoria, total, em Markdown | Respondeu sem chamar tool; inventou valores | ☐ |
| 2.2 | "como está meu fluxo de caixa?" | get_cashflow | Mostra receitas vs despesas por mês | — | ☐ |
| 2.3 | "meu orçamento está ok?" | get_budget_progress | Mostra % usado vs planejado por categoria | — | ☐ |
| 2.4 | "como andam minhas metas?" | get_goal_summary | Lista metas, % concluído, dias restantes | — | ☐ |
| 2.5 | "tenho alguma conta pra pagar essa semana?" | get_recurrences_due | Lista recorrências vencendo nos próximos 7 dias | — | ☐ |
| 2.6 | "quais minhas últimas transações?" | list_transactions_filtered | Lista transações recentes | — | ☐ |
| 2.7 | "me mostra minhas transações de mercado" | list_transactions_filtered (category) | Filtra por categoria corretamente | Não filtrou; chamou tool errada | ☐ |

### O que observar em cada resposta

- [ ] **Chamou a tool certa?** (verifique nos `reasoning` do response ou nos SSE events)
- [ ] **Usou os dados reais?** (não inventou números)
- [ ] **Formatou em Markdown?** (tabelas, negrito, bullets)
- [ ] **Aplicou o símbolo da moeda certa?** (R$, $, € — depende do perfil)
- [ ] **Incluiu sugestões práticas?** (1-2 no final)
- [ ] **Latência aceitável?** (< 15s para consulta simples)

---

## Categoria 3: Consultas multi-step (múltiplas tools)

**O que validar**: esta é a **principal vantagem do v2**. O v1 sempre fazia data_gathering → analysis → synthesis numa sequência fixa. O v2 deve poder chamar múltiplas tools em paralelo ou em sequência, conforme necessário.

| # | Query | Tools esperadas | Comportamento esperado | Red flags | Passou? |
|---|-------|-----------------|----------------------|-----------|---------|
| 3.1 | "me dá uma visão geral das minhas finanças" | get_cashflow + get_spending_by_category + get_budget_progress | Visão consolidada com múltiplas fontes | Chamou só 1 tool; não consolidou | ☐ |
| 3.2 | "estou gastando mais do que ganho?" | get_cashflow + get_spending_by_category | Compara receitas vs despesas, identifica déficit/superávit | Respondeu sem dados | ☐ |
| 3.3 | "tenho dinheiro pra pagar as contas desse mês?" | get_recurrences_due + get_cashflow | Compara contas a pagar vs saldo/renda | — | ☐ |
| 3.4 | "como está meu orçamento de mercado?" | get_budget_progress + list_transactions_filtered (mercado) | Cruzamento de orçamento com transações reais | — | ☐ |
| 3.5 | "minhas metas estão viáveis no ritmo atual?" | get_goal_summary + get_cashflow | Analisa viabilidade considerando fluxo de caixa | Resposta genérica sem dados | ☐ |

### O que observar (diferença v1 vs v2)

- [ ] **v2 chamou tools em paralelo?** (mais rápido) ou em sequência? (mais conservador)
- [ ] **v2 decidiu sozinho quantas tools chamar?** (v1 sempre passava por data_gathering)
- [ ] **A análise foi mais integrada?** (v2 pode cruzar dados de múltiplas tools numa resposta só)
- [ ] **Latência: v2 foi mais rápido ou mais lento que v1?**

---

## Categoria 4: Ações (escrita via Hermes)

**O que validar**: o agente deve interpretar o pedido, verificar parâmetros obrigatórios, chamar a tool de escrita correta, e confirmar. **NUNCA deve inventar valores.**

### 4A: Transações

| # | Query | Tool esperada | Comportamento esperado | Red flags | Passou? |
|---|-------|---------------|----------------------|-----------|---------|
| 4A.1 | "gastei 50 reais no mercado" | create_transaction | Pede conta se múltiplas; cria com valor=50, type=expense, side=debit | Inventou account_id; não perguntou conta | ☐ |
| 4A.2 | "recebi 5000 de salário" | create_transaction | Cria com valor=5000, type=income, side=credit | — | ☐ |
| 4A.3 | "comprei pão" | create_transaction | **PERGUNTA o valor** — não inventa | Criou sem valor; assumiu valor | ☐ |
| 4A.4 | "gastei 100 no posto ontem" | create_transaction | Calcula data = hoje - 1 dia | Usou data de hoje; usou data futura | ☐ |
| 4A.5 | "gastei 30 no café às 15h" | create_transaction | Passa time=15:00 (UTC) | Ignorou horário; horário errado | ☐ |
| 4A.6 | "transferi 200 pra poupança" | create_transaction | type=transfer, side=debit | — | ☐ |

### 4B: Outras ações

| # | Query | Tool esperada | Comportamento esperado | Red flags | Passou? |
|---|-------|---------------|----------------------|-----------|---------|
| 4B.1 | "paga a conta de luz" | pay_recurrence | Identifica recorrência e paga | — | ☐ |
| 4B.2 | "cria uma meta de 10 mil reais" | create_goal | Pergunta detalhes faltantes (prazo, descrição) | Criou com dados incompletos | ☐ |
| 4B.3 | "arquiva meu orçamento de mercado" | archive_budget | Identifica orçamento certo | Arquivou orçamento errado | ☐ |
| 4B.4 | "cria uma conta bancária" | create_account | Pergunta nome, tipo, saldo inicial | — | ☐ |

### 4C: Edge cases de ação

| # | Query | Comportamento esperado | Red flags | Passou? |
|---|-------|----------------------|-----------|---------|
| 4C.1 | "cria uma transação" (sem nenhum detalhe) | Pergunta TODOS os parâmetros: valor, descrição, conta | Cria com valores vazios/zero | ☐ |
| 4C.2 | "gastei -50 reais" | Rejeita valor negativo OU pede esclarecimento | Criou transação com valor negativo | ☐ |
| 4C.3 | "gastei 50 no mercado amanhã" | Recusa data futura, sugere hoje | Criou com data futura | ☐ |
| 4C.4 | "exclui minha conta principal" | Pede confirmação (ou deveria — sem permission gate, pelo menos pede) | Excluiu sem confirmar | ☐ |

---

## Categoria 5: Consulta + Ação combinada

**O que validar**: o v2 deve poder consultar dados E executar ações no mesmo turno. No v1 isso era impossível (action e analysis eram pipelines separados).

| # | Query | Comportamento esperado | Red flags | Passou? |
|---|-------|----------------------|-----------|---------|
| 5.1 | "quanto gastei no mercado? Cria uma transação de R$ 30 lá também" | Primeiro consulta gastos de mercado, depois cria transação | Só fez um dos dois; fez na ordem errada | ☐ |
| 5.2 | "me mostra minhas contas a pagar e paga a primeira" | Lista recorrências, paga a primeira | — | ☐ |
| 5.3 | "como está meu orçamento? Se estourou mercado, cria uma meta pra economizar" | Consulta orçamento, analisa, possivelmente cria meta | Não fez a análise intermediária | ☐ |

> **Estes testes são os mais importantes para validar a vantagem do v2 sobre v1.** Se o v2 não conseguir fazer consulta + ação no mesmo turno, a refatoração não agregou valor.

---

## Categoria 6: Multi-turn (conversa com contexto)

**O que validar**: o agente deve manter contexto entre mensagens da mesma sessão.

| # | Turno | Query | Comportamento esperado | Red flags | Passou? |
|---|-------|-------|----------------------|-----------|---------|
| 6.1 | 1 | "quanto gastei este mês?" | Responde com gastos | — | ☐ |
| 6.1 | 2 | "e no mês passado?" | Entende "mês passado" no contexto da pergunta anterior | Tratou como pergunta nova sem contexto | ☐ |
| 6.1 | 3 | "qual categoria mais chama minha atenção?" | Usa dados das queries anteriores OU busca novamente | — | ☐ |
| 6.2 | 1 | "gastei 50 no mercado" | Cria transação | — | ☐ |
| 6.2 | 2 | "gastei mais 30 lá" | Entende "lá" = mercado, cria outra transação | Perguntou "onde?" de novo | ☐ |
| 6.2 | 3 | "quanto gastei no mercado agora?" | Consulta refletindo as 2 transações criadas | Não incluiu as transações novas | ☐ |

### O que observar

- [ ] **Contexto mantido entre turnos?**
- [ ] **Pronomes/resolução referencial funcionam?** ("lá", "no mês passado", "ela")
- [ ] **Estado atualizado após ações?** (transação criada aparece em consultas subsequentes)
- [ ] **Conversa longa (10+ turnos) ainda funciona?** (sem estourar contexto — pode não funcionar ainda, é esperado na Fase 2)

---

## Categoria 7: Personalização

**O que validar**: tom, idioma, moeda e obfuscação devem ser aplicados consistentemente.

| # | Config do usuário | Query | Comportamento esperado | Red flags | Passou? |
|---|-------------------|-------|----------------------|-----------|---------|
| 7.1 | tone=formal | "quanto gastei?" | Resposta formal, linguagem técnica | Tom casual | ☐ |
| 7.2 | tone=casual | "quanto gastei?" | Resposta descontraída | Tom formal | ☐ |
| 7.3 | language=en_US | "quanto gastei?" | Responde em inglês | Respondeu em português | ☐ |
| 7.4 | language=es_ES | "quanto gastei?" | Responde em espanhol | — | ☐ |
| 7.5 | currency=USD | "qual meu saldo?" | Usa $ em vez de R$ | Usou R$ | ☐ |
| 7.6 | obfuscation=strict | "qual meu saldo?" | Usa faixas ("entre $1.000 e $5.000") | Mostrou valor exato | ☐ |
| 7.7 | obfuscation=standard | "qual meu saldo total?" | Faixas para saldo, valores exatos para gastos individuais | Ocultou gastos individuais | ☐ |
| 7.8 | display_name="Felipe" | qualquer query | Usa "Felipe" ocasionalmente | Usou em toda frase; não usou nunca | ☐ |

> Para testar personalização, precisa de um usuário com preferências setadas no Soter. Se não tiver, pule esta categoria por enquanto.

---

## Categoria 8: Streaming (SSE)

**O que validar**: os eventos SSE devem funcionar corretamente no endpoint `/streaming/chat`.

| # | Teste | Comportamento esperado | Red flags | Passou? |
|---|-------|----------------------|-----------|---------|
| 8.1 | Query simples via streaming | Eventos `token` chegam em tempo real | Buffering; tudo chega de uma vez | ☐ |
| 8.2 | Query com tools via streaming | Eventos `node_execution` com CoT aparecem | Sem eventos intermediários | ☐ |
| 8.3 | Evento `completion` final | Chega com `response` completo e `session_id` | Faltou evento de conclusão | ☐ |
| 8.4 | Query de ação via streaming | Tokens da confirmação streamam em tempo real | — | ☐ |
| 8.5 | Saudação via streaming | Fake-streaming da mensagem estática funciona | — | ☐ |

### Como testar streaming via curl

```bash
curl -N -X POST https://api.olympkusai.com/metis/api/streaming/chat \
  -H "Authorization: Bearer <SEU_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"message": "quanto gastei este mês?"}' 2>&1
```

Você deve ver eventos `data: {"type": "token", ...}` chegando em tempo real.

---

## Categoria 9: Erros e edge cases

**O que validar**: o agente deve falhar graciosamente.

| # | Cenário | Como simular | Comportamento esperado | Red flags | Passou? |
|---|---------|-------------|----------------------|-----------|---------|
| 9.1 | Token inválido | Enviar JWT expirado/inválido | 401 Unauthorized | 500 Internal Server Error | ☐ |
| 9.2 | Pluto indisponível | (difícil simular em prod — pular) | Mensagem amigável "tente novamente" | Stack trace vazou | ☐ |
| 9.3 | Hermes indisponível | Tentar ação quando Hermes está down | Mensagem "não consegui conectar" + modo read-only | Agente trava; erro 500 | ☐ |
| 9.4 | Query vazia | `{"message": ""}` | Saudação ou pedido de esclarecimento | Erro 500; resposta vazia | ☐ |
| 9.5 | Query muito longa | Mensagem de 5000+ caracteres | Processa normalmente | Erro de contexto | ☐ |
| 9.6 | Muitas tools chamadas | "me dá um relatório completo de tudo" | Chama várias tools, consolida | Loop infinito; estoura iterações | ☐ |

---

## Categoria 10: Comparação v1 vs v2 (regressão)

**O que validar**: v2 não deve regredir em nenhum cenário onde v1 funcionava bem.

### Como comparar

```bash
# 1. Ative v1
railway variable set AGENT_VERSION=v1
# Aguarde deploy, rode as queries, anote respostas

# 2. Ative v2
railway variable set AGENT_VERSION=v2
# Aguarde deploy, rode as MESMAS queries, compare
```

### Queries canônicas para comparação

| # | Query | v1 respondeu bem? | v2 respondeu bem? | v2 foi melhor, igual, ou pior? |
|---|-------|-------------------|-------------------|-------------------------------|
| C1 | "oi" | ☐ | ☐ | |
| C2 | "quanto gastei este mês?" | ☐ | ☐ | |
| C3 | "como está meu orçamento?" | ☐ | ☐ | |
| C4 | "gastei 50 no mercado" | ☐ | ☐ | |
| C5 | "me dá uma visão geral" | ☐ | ☐ | |
| C6 | "tenho contas a pagar?" | ☐ | ☐ | |
| C7 | "quanto é 2+2?" | ☐ | ☐ | |
| C8 | "cria uma meta de 10 mil" | ☐ | ☐ | |
| C9 | "me mostra minhas metas" | ☐ | ☐ | |
| C10 | "estou gastando mais do que ganho?" | ☐ | ☐ | |

### Critério de aprovação

- **v2 deve ser ≥ v1 em todos os cenários C1-C10**
- Se v2 for pior em qualquer cenário, **não aprovar** — ajustar prompt e retestar
- Se v2 for igual em simples e melhor em complexos (C5, C10), **aprovar**

---

## Categoria 11: Performance e custo

**O que validar**: v2 não deve ser dramaticamente mais caro ou lento que v1.

| # | Métrica | Como medir | Esperado | Aceitável | Preocupante |
|---|---------|-----------|----------|-----------|-------------|
| 11.1 | Latência consulta simples | Tempo de resposta C2 | < 8s | < 15s | > 30s |
| 11.2 | Latência consulta complexa | Tempo de resposta C5 | < 15s | < 30s | > 60s |
| 11.3 | Latência ação | Tempo de resposta C4 | < 10s | < 20s | > 30s |
| 11.4 | Tokens por request | Verificar `reasoning` no response ou logs | < 5K | < 10K | > 20K |
| 11.5 | Tool calls por request | Contar tools chamadas | 1-3 | 3-6 | > 10 (loop problem) |
| 11.6 | Custo por request | tokens × preço do modelo | < $0.05 | < $0.15 | > $0.50 |

### Como medir tokens

No response do `/api/chat` (não-streaming), o campo `reasoning` contém os steps. Para medir tokens, verifique os logs do Railway:

```bash
railway logs | grep -i "token\|cost\|iteration"
```

---

## Resumo: o que testar primeiro

Se você tem 30 minutos, teste nesta ordem:

1. **C2** — "quanto gastei este mês?" (consulta básica, tool única)
2. **C4** — "gastei 50 no mercado" (ação, cria transação)
3. **C5** — "me dá uma visão geral" (multi-tool, principal vantagem v2)
4. **C10** — "estou gastando mais do que ganho?" (análise com cruzamento de dados)
5. **5.1** — "quanto gastei no mercado? Cria uma transação de R$ 30 lá também" (consulta + ação, impossível no v1)
6. **4A.3** — "comprei pão" (deve perguntar valor, não inventar)
7. **C1** — "oi" (saudação, não deve chamar tools)

Se esses 7 passarem, v2 está funcionando. O resto é refinamento.

---

## O que anotar para cada falha

Quando encontrar um problema, registre:

```
### Falha #N
- **Query**: "..."
- **Comportamento esperado**: ...
- **O que aconteceu**: ...
- **v1 faz melhor?**: sim/não/não testado
- **Tools chamadas**: ...
- **Latência**: ...s
- **Hipótese do problema**: prompt / tool design / context / loop / outro
- **Sugestão de fix**: ...
```

Essas anotações alimentam a próxima rodada de iteração de prompt e dizem qual fase do `agentic-plan.md` priorizar.
