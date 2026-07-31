---
name: experiment-status-artifact
description: Render a fleet of experiment runs as a compact, operational status artifact. Use when a user needs a dashboard or overview across multiple runs, arms, datasets, or configurations rather than a prose report.
---

# Experiment fleet status artifact

Build an instrument panel for scanning and acting on a fleet. Derive facts from the supplied run
records and current evidence; do not invent state or preserve campaign facts in this skill.

## Design the board

1. Put the fleet-level answer first: counts by health, capacity or progress summary, and the most
   urgent actionable condition.
2. Use one row or card per comparable unit. Keep identity, state, progress, throughput, quality,
   resource use, and next action in consistent positions.
3. Encode state redundantly with text, shape/icon, and restrained color. Never rely on color alone.
4. Separate observed facts from inference. Label stale, missing, estimated, or conflicting inputs.
5. Show trends only when the time window and denominator are known. Prefer a small sparkline or
   delta to a decorative chart.
6. Keep controls operational: sort, filter, group, and link to evidence. Avoid controls that do not
   change the view.

## Information hierarchy

- **Header:** scope, capture time, data freshness, and source links.
- **Fleet strip:** total, productive, starting, stalled, terminal, and indeterminate.
- **Primary table:** stable columns with the identifier frozen or visually anchored.
- **Exceptions:** blockers and recommended actions, ordered by urgency.
- **Details:** expandable evidence, configuration, and metric context for a selected row.

Use a table when rows share a schema; use cards only when each run needs materially different
fields. Preserve readable typography, keyboard navigation, visible focus, sufficient contrast, and
responsive overflow.

## Data contract

For each unit, carry the smallest useful record:

```text
id, group, state, stage, progress, freshness,
throughput, quality, resources, evidence_url, next_action
```

Omit unavailable values or mark them unknown. Do not convert absence into zero. Normalize units and
timestamps before comparison, and expose the capture time used for every freshness calculation.

## Delivery checks

- The top-level status agrees with the row data.
- Sorting and filters preserve all records and deterministic ordering.
- Links open the evidence they label.
- Empty, loading, error, stale, and partial-data states are visible.
- The board remains usable at narrow widths and without hover.
- If the artifact is refreshed repeatedly, publish to the same user-approved path so its URL stays
  stable.

Return the artifact link or file plus a one-sentence summary of the most important current finding.
