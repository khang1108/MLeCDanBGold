# STRAIT Production Architecture Design

**Date:** 2026-08-18  
**Status:** Design review candidate  
**Path:** `docs/superpowers/specs/2026-08-18-strait-production-architecture-design.md`  
**Scope:** STRAIT Story-ready MVP and the production platform foundations required to support it  
**Implementation target:** Subproject 1 — Platform Foundation + Requirement Intelligence Vertical Slice

---

## 1. Executive Summary

STRAIT is an event-driven, artifact-centric AI SDLC system that transforms software specification documents into clarified, normalized, traceable requirements and then into high-quality user stories and acceptance criteria.

The architecture separates four forms of state:

1. **Business truth** — PostgreSQL plus immutable artifact revisions.
2. **Durable workflow execution** — Temporal.
3. **Bounded agent reasoning state** — LangGraph checkpoints/state.
4. **Integration and audit facts** — append-only Event Journal plus an event broker.

LLMs never directly mutate canonical business 