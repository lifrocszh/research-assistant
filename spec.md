# Autonomous Research Agent Prototype

## 1. Objective

Build small prototype of general-purpose research agent that can handle:

- Simple question answering
- Multi-step research
- Web/source research
- Local document research
- Document preparation
- Multi-hop investigation
- Parallel specialist research
- Source conflict detection and reconciliation

Core architectural goal:

> Let orchestrator dynamically choose **which fixed subagents to invoke, in what order, and how many times**, rather than following fixed LangGraph workflow paths.

LangGraph remains responsible for durable state/checkpointing. Custom orchestration layer controls dynamic agent/skill/tool discovery and delegation.

## 2. Target Architecture

```text
                         USER
                          │
                          ▼
                 ┌─────────────────┐
                 │   ORCHESTRATOR  │
                 │    AGENT LOOP   │
                 └────────┬────────┘
                          │
                ┌─────────▼─────────┐
                │ Capability Registry│
                │ agents / skills   │
                │ tools / sources   │
                └─────────┬─────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           Agents       Skills       Tools
              │
              ▼
       Dynamic Dispatcher
              │
       ┌──────┴───────┐
       ▼              ▼
   Sequential       Parallel
       │              │
       └──────┬───────┘
              ▼
        Evidence Store
              │
      gaps / conflicts
              │
              ▼
       Follow-up research
              │
              ▼
          Synthesis
              │
              ▼
         Final response
```

## 3. Runtime Responsibilities

### LangGraph

Use LangGraph for:

- State
- Checkpoints
- Durable execution
- Run recovery
- Human interruption
- Session persistence

Do not encode research workflow as fixed graph branches wherever possible.

### Custom orchestration layer

Implement small runtime responsible for:

- Agent selection
- Dynamic delegation
- Parallel execution
- Agent result collection
- Evidence aggregation
- Conflict/gap detection
- Follow-up delegation
- Maximum depth / concurrency / budget controls
- Run event streaming

Core loop:

```python
while not resolved:
    decision = orchestrator.decide(state, capabilities)
    actions = dispatcher.execute(decision)
    state = update_state(state, actions)
```

## 4. Fixed Specialist Agents

Start with 5 specialists.

### Web Research Agent

Purpose:

- General public-web research
- Find relevant sources
- Extract claims and supporting evidence

Free tools:

- Search provider abstraction
- HTTP fetch
- HTML extraction

### Academic Research Agent

Purpose:

- Research scientific / technical questions

Sources:

- arXiv
- Semantic Scholar where practical
- Public papers/pages

Tools:

- arXiv API
- HTTP fetch
- PDF/text extraction

### Company / Financial Research Agent

Purpose:

- Company research
- Financial information
- Regulatory filings

Sources:

- SEC EDGAR
- Company investor-relations pages
- Public filings

Tools:

- SEC API / EDGAR endpoints
- HTTP fetch
- Document extraction

### Document Research Agent

Purpose:

- Search and reason over user-provided documents

Tools:

- Local file loader
- PDF/text extraction
- Document search

### Synthesis / Critic Agent

Purpose:

- Compare findings
- Detect unsupported claims
- Detect contradictory evidence
- Identify unanswered questions
- Recommend next research step

This agent should **not** be responsible for broad discovery.

## 5. Tools

Keep tool set deliberately small.

### Core tools

```text
web_search(query)
fetch_url(url)
fetch_arxiv(query)
fetch_sec(query)
read_document(path)
search_document(query)
```

Optional:

```text
extract_pdf(path)
calculate(expression)
```

Tools should perform concrete actions.

Avoid exposing large numbers of narrowly differentiated tools.

Live search adapters receive concise queries rather than orchestration instructions. Newline-delimited directives and follow-up metadata are removed before Tavily or arXiv calls. Fixture search keeps the original deterministic matching behavior.

## 6. Skills

Skills represent reusable procedures rather than capabilities.

Initial skills:

```text
general_research
source_evaluation
company_research
literature_review
document_analysis
evidence_reconciliation
report_writing
```

Example:

```text
company_research
  1. identify primary sources
  2. retrieve filings
  3. extract relevant metrics
  4. compare periods
  5. identify inconsistencies
  6. return claims + citations
```

Skills should be progressively discoverable rather than permanently placed in every agent context.

## 7. Capability Registry

Create registry containing:

```python
AgentSpec(
    name="company_researcher",
    description="Research companies using primary financial/regulatory sources",
    sources=["SEC", "company filings"],
    skills=["company_research"],
    core_tools=["fetch_sec", "fetch_url"],
)
```

Similar `SkillSpec` and `ToolSpec`.

Registry supports:

```text
discover_agents(query)
discover_skills(query)
discover_tools(query)
```

Do not place every agent, skill, and tool definition into orchestrator context.

## 8. Dynamic Delegation

The initial plan may split a multi-topic question into independent, topic-scoped actions. The prototype currently recognizes architecture/implementation and competitor/comparison topics. Each action carries its topic into evidence collection and final Markdown sections.

Orchestrator should be able to produce plans like:

```text
Task
 ↓
Web Research Agent
 ↓
Company Research Agent
 ↓
compare findings
 ↓
conflict detected
 ↓
SEC Research Agent
 ↓
Synthesis Agent
```

Or:

```text
Task
 ↓
parallel:
    Academic Agent
    Web Agent
    Document Agent
 ↓
Synthesis Agent
```

Or:

```text
Task
 ↓
Document Agent
 ↓
gap identified
 ↓
Web Agent
 ↓
follow-up
 ↓
Synthesis
```

No fixed maximum number of research hops beyond runtime safety limits.

Topic follow-ups target the missing topic. A generic source-category gap must not route a topic-specific question to an unrelated academic search.

## 9. Evidence Model

Every research result should return structured evidence.

```python
Finding(
    claim=str,
    source=str,
    source_type=str,
    evidence=str,
    confidence=float,
    citation=str | None,
    topic=str | None,
    entities=list[str],
    time_period=str | None,
)
```

Before storage, findings pass bounded quality checks. Fragmentary or obvious repeated-text claims are rejected. Architecture findings require the requested subject and implementation vocabulary. Competitor findings require a named competitor in the title or claim and recommendation-system implementation vocabulary in the claim.

Maintain:

```text
EvidenceStore
  ├── findings
  ├── sources
  ├── claims
  ├── contradictions
  └── unanswered_questions
```

This gives orchestrator structured material for subsequent reasoning.

## 10. Multi-Hop Behavior

Prototype must demonstrate:

### Hop 1

Research primary question.

### Hop 2

Inspect findings and identify:

- missing evidence
- contradiction
- ambiguity
- weak source

### Hop 3

Delegate targeted investigation.

### Hop 4

Reconcile evidence.

### Hop 5

Synthesize answer.

Example:

```text
"What caused Company X margin decline?"

Company Agent
    ↓
finds margin decline

SEC Agent
    ↓
finds reported cost increase

Transcript Agent / Web Agent
    ↓
management gives different explanation

Critic
    ↓
detects discrepancy

SEC Agent
    ↓
verify accounting detail

Synthesis
    ↓
final explanation + citations
```

## 11. Parallelism

Dispatcher must support:

```python
parallel([
    delegate("web_researcher", task1),
    delegate("company_researcher", task2),
    delegate("academic_researcher", task3),
])
```

Parallelism should be selected by orchestrator when tasks are independent.

Runtime controls:

```text
max_parallel_agents
max_total_agents
max_research_depth
max_runtime
max_tokens
```

## 12. Web UI

Build minimal UI.

### Backend

FastAPI + WebSocket.

### Frontend

Small web UI showing live execution.

Example:

```text
Researching...
│
├─ Orchestrator: planning
├─ Company Researcher: searching SEC
├─ Web Researcher: searching investor relations
├─ Company Researcher: found 10-K
├─ Orchestrator: conflict detected
├─ SEC Researcher: verifying claim
├─ Critic: evidence reconciled
└─ Synthesis: drafting answer
```

WebSocket event schema:

```python
RunEvent(
    run_id,
    timestamp,
    event_type,
    agent,
    task,
    status,
    result_summary,
)
```

Event types:

```text
run_started
planning
agent_started
tool_started
tool_finished
agent_finished
parallel_started
evidence_added
conflict_detected
followup_started
synthesis_started
run_finished
```

## 13. Model Agnosticism

No provider-specific orchestration logic.

Use model abstraction supporting:

```text
orchestrator_model
specialist_model
critic_model
```

Allow different models later.

Prototype should work with any LangChain-compatible chat model.

## 14. Prototype Success Criteria

Use 10 representative questions covering:

1. Simple factual question
2. Multi-source research question
3. Company research
4. Academic question
5. User-document + web research
6. Contradictory sources
7. Multi-hop question
8. Question requiring parallel investigation
9. Question requiring follow-up after evidence gap
10. Document/report generation

Compare dynamic system against fixed workflow.

Measure:

```text
accuracy
citation quality
source quality
claim relevance and completeness
query quality
number of useful research hops
unnecessary tool calls
parallelization quality
conflict detection
unanswered questions
latency
token cost
```

Primary success criterion:

> Different questions should produce meaningfully different execution paths without developer-defined workflow branches.

## 15. Non-Goals

Do not build initially:

- Large agent marketplace
- Hundreds of tools
- Complex memory architecture
- Autonomous browser computer-use
- Paid proprietary research APIs
- Production authentication/permissions
- Fully autonomous external actions
- Complex frontend
- Sophisticated reinforcement learning

## 16. Recommended Prototype Principle

Keep distinction strict:

```text
Tool
= action

Skill
= procedure / knowledge

Subagent
= independent reasoning context

Orchestrator
= decides which capabilities to compose

LangGraph
= durable execution infrastructure
```

Do not optimize upfront for perfect tool/skill bundling.

Instead implement **progressive capability discovery** and collect execution traces. Use traces from prototype runs to determine which tools belong permanently inside specialist agents versus which should remain discoverable.

## 17. Main Experiment

Run same question set through:

```text
A. Fixed LangGraph research workflow

B. Dynamic orchestrator
   + fixed specialist agents
   + capability registry
   + dynamic delegation
   + parallel execution
   + evidence store
   + multi-hop follow-up
```

The prototype succeeds when B can discover and execute different research strategies based on task requirements, while remaining observable, bounded, and recoverable.

### Useful reference architecture

Deep Agents remains a future comparison point, not a current dependency. The prototype first validates its smaller LangGraph plus custom bounded orchestrator. Reconsider Deep Agents after evaluation shows that filesystem context management, recursive delegation, or runtime-generated workflows solve a measured limitation.
