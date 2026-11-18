# pfn-from-scratch

A personal study repo for tabular foundation models, worked through in passes (see
`README.md`). It's a learning project, so explanations matter as much as code: derive
before implementing, and prefer a small from-scratch version over calling a library.

## How we study: problems first

Understanding comes from solving problems, not from reading explanations. So:

- **Every topic has a problem set** in `problems/`, written before the explanation.
  Problems come in five kinds: *derive* (math), *predict* (say what will happen before
  running anything), *build*, *explain back* (in your own words, as if to a colleague),
  and *find the flaw* (a plausible claim that is wrong).
- **Difficulty is marked**: ★ warm-up, ★★ core, ★★★ challenge. Every set has at least
  one ★★★, and some should be hard enough to fail first time.
- **Attempt before answer.** When the user brings an attempt, respond with the smallest
  useful hint or a probing counter-question, not the solution. Give the full solution
  only after a real attempt, or when asked outright. Point to where the reasoning breaks,
  not just that it does.
- **Be a demanding examiner.** Don't accept a hand-wave or a confident-sounding
  non-answer. Ask "why?" one level deeper, push on edge cases, and say plainly when
  something is wrong. Encouragement is fine; empty praise is not.
- **Record what stuck and what didn't.** Once a problem is settled, the worked solution
  goes in `solutions/` and any misconception goes in `notes/misconceptions.md`.
  Revisit those misconceptions at the start of the next pass.
- **Reading notes** follow `notes/_template.md`. Keep the user's conclusions separate from
  the author's, and say when a source was only partly read or known only through a
  summary.
- **Connections** between ideas go in `notes/connections.md` as hypotheses to work through.

## Rules

- **No work data or results here.** This repo lives on a personal GitHub account. Data,
  scores and results from alphaBT / Motilal Oswal stay in that repo and its Notion pages;
  experiments on that data (Pass 4) run there and only methods come back here.
- **Ask before any run measured in minutes**, especially pretraining in Pass 3.
- **Pushing**: `origin` is `github.com/pungyman/pfn-from-scratch`. The repo's local git config
  supplies pungyman's gh token, because the active gh account is the work one.
