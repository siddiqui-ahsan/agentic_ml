# Prompt to Production
**Technische Dokumentation — Agentic ML Pipeline**

**LangGraph + FastAPI + Railway | Airbnb Price Tier Prediction**

---

## 1. Projektüberblick

Ziel des Projekts ist der Bau eines autonomen KI-Agenten, der den **Price-Tier** von New Yorker Airbnb-Unterkünften vorhersagt.

Es gibt vier Klassen:

- **0 — Budget** → günstige Unterkünfte
- **1 — Standard** → mittleres Preissegment
- **2 — Premium** → gehobene Unterkünfte
- **3 — Ultra-Luxury** → Luxus-Penthäuser (seltene Klasse)

Der Agent kombiniert **strukturierte Tabellendaten** mit **unstrukturierten Textbeschreibungen** der Hosts. Ein lokales LLM (Llama 3.1 8B via Ollama) extrahiert boolesche Feature-Flags aus den Beschreibungen, die dann in das ML-Modell einfließen.

Der komplette Agent wird mit **LangGraph** orchestriert, als **FastAPI**-Service verpackt und auf **Railway** deployed.

---

## 2. Architektur & Datenfluss

### 2.1 LangGraph-Graph

LangGraph verwaltet einen gemeinsamen `AgentState`, der durch alle Nodes fließt. Conditional Edges steuern den Ablauf.

**Haupt-Pipeline:**