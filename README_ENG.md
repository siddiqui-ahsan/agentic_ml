# Prompt to Production
**Technical Documentation — Agentic ML Pipeline**

**LangGraph + FastAPI + Railway | Airbnb Price Tier Prediction**

---

## 1. Project Overview

The goal of this project is to build an autonomous AI agent that predicts the **Price Tier** of New York Airbnb listings.

There are four classes:

- **0 — Budget** → cheap accommodations
- **1 — Standard** → mid-range segment
- **2 — Premium** → upscale accommodations
- **3 — Ultra-Luxury** → luxury penthouses and rare high-end properties

The agent combines **structured tabular data** (location, room type, availability, etc.) with **unstructured text descriptions** from the hosts. A local LLM (Llama 3.1 8B via Ollama) extracts boolean feature flags from the descriptions, which are then used together with the tabular data to train a machine learning model.

The entire agent is orchestrated with **LangGraph**, wrapped as a **FastAPI** web service, and deployed on **Railway**.

---

## 2. Architecture & Data Flow

### 2.1 LangGraph Graph

LangGraph manages a shared `AgentState` that flows through all nodes. Each node reads from the state, modifies it, and passes it on. Conditional edges enable intelligent routing.

**Main Pipeline:**