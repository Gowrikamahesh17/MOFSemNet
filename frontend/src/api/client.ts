import type {
  ArchitectureData,
  EmbeddingDemoResponse,
  PapersData,
  PlanData,
  ProgressData,
  StepResult,
} from '../types/content';

// Empty string in production (same-origin, FastAPI serves the built frontend); the Vite
// dev server proxies /api to the backend (see vite.config.ts), so this stays empty there
// too. Only set VITE_API_BASE_URL when the backend genuinely lives on a different origin.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function getArchitecture(): Promise<ArchitectureData> {
  return getJSON<ArchitectureData>('/api/architecture');
}

export function getPapers(): Promise<PapersData> {
  return getJSON<PapersData>('/api/papers');
}

export function paperPdfUrl(paperId: string): string {
  return `${API_BASE}/api/papers/${paperId}/pdf`;
}

export function getPlan(): Promise<PlanData> {
  return getJSON<PlanData>('/api/plan');
}

export function getProgress(): Promise<ProgressData> {
  return getJSON<ProgressData>('/api/progress');
}

export async function getStepResult(stepId: string): Promise<StepResult | null> {
  const response = await fetch(`${API_BASE}/api/results/${stepId}`);
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`/api/results/${stepId} failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<StepResult>;
}

export async function embedDemo(smiles: string): Promise<EmbeddingDemoResponse> {
  const response = await fetch(`${API_BASE}/api/demo/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<EmbeddingDemoResponse>;
}
