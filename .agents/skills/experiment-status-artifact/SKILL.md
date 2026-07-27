---
name: experiment-status-artifact
description: >-
  Render a fleet of experiment runs as a published status Artifact — an instrument readout, not
  a document. Use when someone asks for an overview, status table, or dashboard across a set of
  runs, arms, or datasets and wants it rendered rather than printed to the terminal. Produces a
  board that is scanned and operated. Load the artifact-design skill first; this specializes it.
---

# Experiment status artifact

A status board is **scanned and operated, not read top to bottom**. The craft is information
design rather than typography: surface the summary before the detail, and encode state in
*form* as well as number so what needs attention reads at a glance.

The treatment is deliberately **utilitarian** — no hero, no scroll animation, no display serif.
`artifact-design` governs; this narrows it to one page type.

## The rule that makes this format worth building

**Find the single quantity that decides each run's outcome, and give it a visual axis.**

Everything else on the page is a table. That one column is why the page exists, because a
reader scanning down it sees the campaign's finding directly — which runs sit inside the range
where the method can work and which sit outside it — and that is not recoverable from a column
of numbers.

If you cannot name that quantity, do not build the artifact yet. Find it first. A status board
without it is a worse version of a markdown table.

The quantity is usually the one a launch decision already turns on: whichever measurement the
owning policy screens on before committing hardware. Read that policy rather than inventing a
metric.

## Structure, in order

1. **Header** — eyebrow naming the campaign, an H1 stating scope as a fraction
   (`N of M attempted`), and a subtitle carrying the fixed parameters plus an explicit
   **as-of timestamp in UTC**. Readers open these hours later.
2. **Summary strip** — 4–6 counts. Include at least one that is uncomfortable. A strip of only
   flattering numbers is decoration, not an instrument.
3. **Sections by lifecycle state**, most actionable first: running, then finished, then ended
   early. Not alphabetical, not chronological. Each heading carries a count and a qualifier
   that says something true about the group.
4. **The decisive-quantity column**, in every section where it applies.
5. **Legend**, only if a gauge is drawn. Explain what the shaded region *means*, not what the
   colors are.
6. **Callout** — two or three short paragraphs of what the data establishes. Every claim tied
   to a number already in the tables above. This is where a reader who scrolled looks for the
   conclusion.
7. **Footer** — reconciliation notes: why the run count differs from the subject count, what is
   retired, anything a careful reader would otherwise flag as inconsistent.

## Content rules

- **Name each row by what it is to the reader** — the dataset, model, or config under test —
  not by its job id. Put the id in the note only when it is needed to act.
- **The "why it ended" cell is the most valuable on the page.** Give it real numbers, not
  categories: a measured collapse with its endpoints beats the word "collapse". Cap it near
  44ch so it stays scannable.
- **Distinguish failure kinds with different pills, and make the taxonomy load-bearing.**
  Causes that imply different fixes need different labels. A single generic "failed" pill
  throws away the reason the board is useful.
- **State progress as a fraction with a bar**, never a bare percentage.
- Bold the one load-bearing number inside each note.
- Pull every figure in a single pass so rows cannot silently disagree about as-of time.

## Design tokens

Cool-biased neutrals, one accent, and semantic state colors **kept separate from the accent** —
good/warning/critical must not be the brand hue. Type is system stacks used deliberately: sans
for labels, monospace with `tabular-nums` for every figure. Do not link a webfont; the Artifact
CSP blocks font CDNs and the page would silently fall back.

```css
:root{
  --bg:#FBFBFD; --panel:#FFFFFF; --ink:#14171D; --ink-2:#3D4552; --muted:#6B7484;
  --line:#E3E6EC; --line-2:#EFF1F5;
  --accent:#2D7D8A;                                /* instrument teal, not a brand hue */
  --live:#1F8F63; --done:#2D6E8A;                  /* semantic, independent of accent  */
  --warn:#B4682F; --crit:#9E3A57; --neutral:#6B7484;
  --band:#DCE6E8;                                  /* the healthy region of the gauge   */
  --shadow:0 1px 2px rgba(16,20,28,.05), 0 4px 14px rgba(16,20,28,.04);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0F1216; --panel:#161A20; --ink:#E7EBF1; --ink-2:#B4BCC9; --muted:#818B9B;
    --line:#252B34; --line-2:#1D222A;
    --accent:#5FB3BF;
    --live:#42B98A; --done:#5FA3C4; --warn:#D68B4F; --crit:#C96A85; --neutral:#818B9B;
    --band:#232C31; --shadow:none;
  }
}
/* The viewer's theme toggle stamps data-theme on :root and MUST beat the media query in
   BOTH directions, so repeat each palette verbatim below. Style components through the
   tokens only — never place component rules inside the media query, or the toggle cannot
   override them. */
:root[data-theme="dark"]{ /* identical to the dark block above */ }
:root[data-theme="light"]{ /* identical to the light block above */ }
```

Type: `ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` for prose;
`ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace` with
`font-variant-numeric: tabular-nums` for figures. Body 15px/1.55; H1 27px/1.2 at weight 640 and
`letter-spacing:-.015em`; section labels 12px uppercase at `letter-spacing:.12em`.

## The gauge — the component that carries the page

A fixed-width track with the healthy band shaded and one tick per row, colored by which side of
the band it falls on. `left:` is the value as a percentage of the axis, so rescale if the
quantity is not already 0–1.

```html
<div class="gwrap">
  <div class="gauge">
    <div class="track"></div>
    <div class="safe"></div>                        <!-- healthy band -->
    <div class="tick ok" style="left:36.6%"></div>  <!-- .ok | .lo | .hi -->
  </div><span class="gval mono">0.366</span>
</div>
```

```css
.gauge{position:relative;width:132px;height:16px;flex:none}
.gauge .track{position:absolute;inset:6px 0 auto;height:4px;border-radius:2px;background:var(--line)}
.gauge .safe {position:absolute;top:6px;height:4px;left:25%;width:50%;border-radius:2px;background:var(--band)}
.gauge .tick {position:absolute;top:2px;width:2px;height:12px;border-radius:1px;background:var(--ink)}
.gauge .tick.lo{background:var(--warn)}     /* below the band */
.gauge .tick.hi{background:var(--crit)}     /* above the band */
.gauge .tick.ok{background:var(--live)}     /* inside         */
.gval{font-size:12px;color:var(--ink-2);margin-left:7px}
.gwrap{display:flex;align-items:center}
```

Set `.safe`'s `left` and `width` from the actual bounds the owning policy screens on, and pick
each tick's class by comparing to those same bounds in code. Never place a tick by eye.

## Supporting components

```css
/* state pill — one class per failure KIND, never one generic "failed" */
.pill{display:inline-block;font-size:10.5px;font-weight:640;letter-spacing:.05em;
      text-transform:uppercase;padding:2.5px 7px;border-radius:4px;white-space:nowrap}
.p-live{color:var(--live);background:color-mix(in srgb,var(--live) 13%,transparent)}
.p-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 14%,transparent)}

/* progress — fraction plus bar, never a bare percentage */
.prog{display:flex;align-items:center;gap:8px;white-space:nowrap}
.prog .bar{width:52px;height:4px;border-radius:2px;background:var(--line);overflow:hidden;flex:none}
.prog .fill{height:100%;background:var(--accent);display:block}

/* wide content scrolls inside its own container so the page body never scrolls sideways */
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
     border-radius:8px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;min-width:900px}
```

## Publishing

Write the page to the session scratchpad, then call `Artifact` with `file_path`, a
one-sentence `description`, and a `favicon`. Put a `<title>` in the HTML. Do **not** emit
`<!DOCTYPE>`, `<html>`, `<head>`, or `<body>` — the content is wrapped at publish time.

Keep the favicon stable across redeploys; readers find the tab by its icon. To update,
republish **the same file path** from the same conversation and the URL is preserved; from a
different conversation, pass the previous URL as `url` or a new one is minted.

## Failure modes

- **Building before the decisive quantity is known.** The result is a wide table nobody reads.
- **Mixing as-of times.** Pull every figure in one pass and print the timestamp, or rows
  silently disagree.
- **One generic "failed" pill.** Collapses the taxonomy that makes the board worth rendering.
- **Component rules inside the dark-mode media query.** The viewer's toggle can then no longer
  override them. Redefine tokens only.
- **A summary strip of only flattering counts.** That is a poster, not an instrument.
- **A gauge on a quantity the campaign does not actually gate on.** The axis has to be the one
  a decision turns on, or it is decoration with tick marks.
