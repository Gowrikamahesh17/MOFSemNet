# Adding content to the web console

Everything the console shows lives in `content/*.yaml`. You should rarely, if ever, need to
touch `backend/` or `frontend/` code just to add a new result, paper, or pipeline stage — that's
the point of keeping content separate from layout.

After editing any `content/*.yaml` file while the backend is running, either restart it or
`curl -X POST http://localhost:8010/api/content/reload` to pick up the change without a restart.
Every file is validated against a schema (`backend/app/models/schemas.py`) and cross-checked
(e.g. a node's paper reference must exist) at load time — a typo fails loudly in the terminal,
not silently in the browser.

## Add a new architecture node (pipeline stage, graph type, comparison method)

Edit `content/architecture.yaml`:

```yaml
nodes:
  - id: my_new_stage          # unique, used as the React key and in edges
    label: "Step 7\nMy New Stage"   # \n makes a two-line card title
    group: pipeline             # "pipeline" | "graph" | add your own — only affects the
                               # small badge shown on "graph" nodes right now
    status: planned              # done | next | planned — drives every status color in the app
    summary: "One sentence, shown on the card and in the hover tooltip."
    description: "Longer paragraph, shown in the detail panel on click."
    why_chosen: "Why this approach, not an alternative. Omit (or null) if not decided yet."
    studied:
      - "A bullet point of what you read/learned to build this."
    references: [p4, p9]         # paper ids from content/papers.yaml — must already exist
    metrics: {}                  # {} until the stage has real, measured numbers. Never invent one.

edges:
  - { from: step6, to: my_new_stage, animated: false }
```

That's it — the diagram relayouts itself (dagre, left-to-right, layered by the edges you just
added). No frontend code changes, no manual x/y coordinates.

## Add a new paper

Edit `content/papers.yaml`:

```yaml
papers:
  - id: p99                        # unique, referenced from architecture.yaml's `references`
    citation_key: "[18]"
    cluster: mof_fusion             # must match an existing cluster id above, or add one
    title: "..."
    authors: "..."
    year: 2026
    venue: "..."
    url: "https://..."
    summary: "1-2 sentences."
    has_local_pdf: true              # false if you're not bundling the PDF (see papers/README.md
                                    # on why some papers deliberately aren't)
    local_pdf: "my-paper.pdf"        # filename under papers/, only when has_local_pdf is true
```

To add a new cluster, add an entry to `clusters:` first (id, label, narrative — the narrative is
what `/studied` shows).

## Add a new step's results

Create `content/results/step7.yaml` (filename must match the step id used in
`GET /api/results/<id>` — see `backend/app/services/content_loader.py`'s `_VALID_STEP_IDS` if you
need a step 7+; currently only `step1`-`step4` are wired up, since Steps 5/6 have no results yet):

```yaml
title: "Step 7 — My New Stage"
status: done
summary: "One paragraph."
stats:
  - label: "Some metric"
    value: "0.42"
    emphasis: true          # highlights the tile — use for the one number that matters most
finding: "Optional: a dark callout box for the most important takeaway."
# Optional, for a Step-3-style headline number:
headline: "0.21"
headline_label: "What this number is"
# Optional, for a Step-4-style results table:
columns: ["Col A", "Col B"]
rows:
  - ["value", "value"]
```

A result card can also show images and a gallery: `figures` is a list of `{src, caption, alt}` (files under
`frontend/public/`, e.g. `/step6/shap_top_features_chemberta.png`), and `gallery_url` (with `gallery_title`) points to a JSON file in
`frontend/public/` that the Step 6 card renders as fragment cards (see `scripts/run_step6_explainability.py`, which writes it).

A step with no `results/<id>.yaml` file returns 404, not a placeholder — `/implemented` simply
skips it rather than showing an invented number.

## Add a new page / nav section

1. Add a route in `frontend/src/App.tsx` and a link in `frontend/src/components/layout/NavBar.tsx`.
2. If it needs its own content file, follow the pattern in any existing page: add a Pydantic
   model to `backend/app/models/schemas.py`, a loader function in
   `backend/app/services/content_loader.py`, a route in a new `backend/app/api/routes_*.py`
   (registered in `backend/app/main.py`), a matching TypeScript type in
   `frontend/src/types/content.ts`, and a fetch wrapper in `frontend/src/api/client.ts`.

This is the only place genuine code changes are expected — a new *section* is a product
decision, not something that should be purely data-driven. Everything *within* an existing
section (a new node, paper, or result) should only ever need a YAML edit.
