# Deploying the web console

## Local / laptop (for a meeting, no server needed)

```bash
docker compose up --build
```

Open `http://localhost:8000`. First build takes a few minutes (npm install + Docker layers);
subsequent builds are fast (layer cache). The ChemBERTa model and dataset embeddings load lazily
on the *first* live-demo request (a few seconds), not at container startup, so the container
itself starts in under a second.

## University server

1. **Copy the repo** (or `git clone`) to the server, including `data/raw/MOFCSD.csv`,
   `data/embeddings/*.npz` (Step 2's cached embeddings — without these, the live demo on
   `/implemented` falls back to embedding the whole dataset, which takes minutes, not seconds),
   and `content/*.yaml`.
2. **Set `MOFSEMNET_CORS_ORIGINS`** if you're reverse-proxying the API and frontend under
   different hostnames/ports; leave unset for the default single-container same-origin setup.
3. **Build and run**:
   ```bash
   docker compose up --build -d
   ```
4. **Reverse proxy** (nginx/Caddy) to `:8000` if you want a real domain/HTTPS — the container
   itself only serves plain HTTP on `:8000`. A minimal nginx `location` block:
   ```nginx
   location / {
       proxy_pass http://127.0.0.1:8000;
       proxy_set_header Host $host;
   }
   ```
5. **GPU vs CPU**: `MOFSEMNET_DEMO_MODEL` picks which embedder backs the live demo. Device
   selection (CUDA/MPS/CPU) is automatic — `src/common/runtime.py`'s `get_device()` picks
   whatever's available on the server, same as the rest of the ML pipeline. No server-specific
   code path to maintain.

## Updating content without redeploying

`content/`, `data/`, and `outputs/` are volume-mounted (see `docker-compose.yml`), not baked into
the image. To update a result, add a paper, or add an architecture node:

```bash
# edit content/*.yaml locally or on the server, then:
curl -X POST http://localhost:8000/api/content/reload
```

No rebuild, no restart. A code change (new page, new route) does need a rebuild:
`docker compose up --build -d`.

## What's NOT handled yet

- No authentication — the console is meant to be shown in a meeting or shared with the
  supervisor directly, not exposed publicly as-is. Put it behind the university VPN or a
  reverse-proxy auth layer if it needs to be reachable from the open internet.
- No automatic HTTPS — terminate TLS at the reverse proxy (step 4 above), the container itself
  doesn't.
- The live demo endpoint has no rate limiting. Fine for a supervisor meeting; worth adding
  (e.g. `slowapi`) before sharing the link more broadly.
