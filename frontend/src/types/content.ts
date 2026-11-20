// Mirrors backend/app/models/schemas.py exactly. If the backend response shape changes,
// update both — there's no code generation step here, so this is the one place to check
// when a field goes missing on the frontend.

export type NodeStatus = 'done' | 'next' | 'planned';

export interface ArchitectureNode {
  id: string;
  label: string;
  group: string;
  status: NodeStatus;
  summary: string;
  description: string;
  why_chosen: string | null;
  studied: string[];
  references: string[];
  metrics: Record<string, string>;
}

export interface ArchitectureEdge {
  from: string;
  to: string;
  animated: boolean;
}

export interface ArchitectureData {
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
}

export interface PaperCluster {
  id: string;
  label: string;
  narrative: string;
}

export interface Paper {
  id: string;
  citation_key: string;
  cluster: string;
  title: string;
  authors: string;
  year: number;
  venue: string;
  url: string;
  summary: string;
  has_local_pdf: boolean;
  local_pdf: string | null;
}

export interface PapersData {
  clusters: PaperCluster[];
  papers: Paper[];
}

export interface ResearchQuestion {
  id: string;
  text: string;
  answer?: string | null;
}

export interface PlanData {
  context: string;
  thesis_statement: string;
  research_questions: ResearchQuestion[];
  guardrails: string[];
  negative_result_framing: string;
}

export interface WorkPackage {
  id: string;
  label: string;
  window: string;
  status: NodeStatus;
}

export interface Milestone {
  date: string;
  label: string;
}

export interface ProgressData {
  work_packages: WorkPackage[];
  milestones: Milestone[];
  ahead_of_schedule_note: string;
}

export interface ResultStat {
  label: string;
  value: string;
  emphasis: boolean;
}

export interface EmbeddingNeighbor {
  refcode: string;
  linker_smiles: string;
  metal: string;
  pld_value: number;
  cosine_similarity: number;
}

export interface MetalOption {
  metal: string;
  mofs: number;
}

export interface EmbeddingDemoResponse {
  smiles: string;
  metal_filter: string | null;
  model: string;
  embedding_dim: number;
  embedding_norm: number;
  neighbors: EmbeddingNeighbor[];
}

export interface StepResult {
  title: string;
  status: NodeStatus;
  summary: string;
  stats: ResultStat[];
  finding: string | null;
  headline: string | null;
  headline_label: string | null;
  retention_shown: string | null;
  columns: string[];
  rows: string[][];
}

export interface GraphNode {
  smiles: string;
  mofs: number;
  mean_pld: number;
  x: number;
  y: number;
}

export interface GraphNeighbourhood {
  seed: number;
  nodes: GraphNode[];
  // [node index, node index, number of MOF-level edges collapsed into this link]
  fingerprint: [number, number, number][];
  semantic: [number, number, number][];
  shared_edges: number;
}

export interface GraphComparisonStats {
  mofs: number;
  distinct_linkers: number;
  mof_edges: Record<string, number>;
  mof_jaccard: number;
  same_linker_edge_share: Record<string, number>;
  linker_pair_edges: Record<string, number>;
  linker_pair_jaccard: number;
  components: Record<string, number>;
  mean_abs_pld_gap_between_neighbours: Record<string, number>;
  mean_abs_pld_gap_random_pair: number;
}

export interface GraphComparison {
  model: string;
  k: number;
  stats: GraphComparisonStats;
  neighbourhoods: GraphNeighbourhood[];
}
