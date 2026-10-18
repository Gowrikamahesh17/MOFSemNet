"""FastAPI app entrypoint. Run locally with:
    uvicorn app.main:app --reload --app-dir backend

In production this same app also serves the built React frontend as static files (see the
bottom of this file) — one process, one URL, matching the prior Case Study's web console.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response
from starlette.types import Scope

from app.api import (
    routes_architecture,
    routes_content,
    routes_papers,
    routes_plan,
    routes_predict,
    routes_progress,
    routes_results,
)
from app.core.config import settings

app = FastAPI(
    title="MOFSemNet Console",
    description="Live progress and architecture console for the MOFSemNet thesis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(routes_architecture.router)
app.include_router(routes_papers.router)
app.include_router(routes_content.router)
app.include_router(routes_plan.router)
app.include_router(routes_progress.router)
app.include_router(routes_results.router)
app.include_router(routes_predict.router)


@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


class SPAStaticFiles(StaticFiles):
    """Plain StaticFiles *raises* HTTPException(404) on a path that isn't an actual file on
    disk — e.g. /architecture, which only exists as a client-side React Router route, not a
    file in frontend/dist (verified directly: Starlette's own get_response does
    `raise HTTPException(status_code=404)`, not return a 404 response, so a naive
    status-code check after calling super() never fires — it has to be a try/except).
    Falls back to index.html for anything that isn't a real static asset and isn't already
    an /api/* route (those are matched by FastAPI before this mount is ever reached), so the
    SPA's own router handles it instead of the browser seeing a raw 404 JSON body.

    Must catch `starlette.exceptions.HTTPException`, not `fastapi.HTTPException` — verified
    directly that the latter is a *subclass* of the former, so catching it would silently
    miss every exception staticfiles.py itself raises (it imports from starlette, not
    fastapi), and this fallback would never fire. First attempt at this fix used the wrong
    import and failed exactly that way before being caught by testing against the real
    build rather than trusting the code.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            # An unmatched /api/* path (a genuine wrong-endpoint bug, not a client-side
            # route) must stay a real 404 — without this check, a 404 here always fell
            # back to index.html, which masked a real typo'd API call as an HTTP 200.
            if path.startswith("api/") or path == "api":
                raise
            return FileResponse(Path(self.directory) / "index.html")


# --- Production: serve the built frontend as static files ---
# frontend/dist only exists after `npm run build`; in local dev (`npm run dev`, Vite on
# :5173) this block is skipped and CORS above handles the cross-origin requests instead.
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", SPAStaticFiles(directory=_frontend_dist, html=True), name="frontend")
