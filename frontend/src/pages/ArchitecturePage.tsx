import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getArchitecture, getPapers } from '../api/client';
import { FlowCanvas } from '../components/architecture/FlowCanvas';
import { Legend } from '../components/architecture/Legend';
import { NodeDetailPanel } from '../components/architecture/NodeDetailPanel';
import type { ArchitectureData, ArchitectureNode, Paper, PapersData } from '../types/content';
import styles from './ArchitecturePage.module.css';

export function ArchitecturePage() {
  const [architecture, setArchitecture] = useState<ArchitectureData | null>(null);
  const [papers, setPapers] = useState<PapersData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ArchitectureNode | null>(null);
  const [searchParams] = useSearchParams();
  const requestedNodeId = searchParams.get('node');

  useEffect(() => {
    Promise.all([getArchitecture(), getPapers()])
      .then(([architectureData, papersData]) => {
        setArchitecture(architectureData);
        setPapers(papersData);
        const requested = requestedNodeId && architectureData.nodes.find((node) => node.id === requestedNodeId);
        setSelected(requested || architectureData.nodes.find((node) => node.status === 'next') || architectureData.nodes[0]);
      })
      .catch((err: Error) => setError(err.message));
    // Only re-run if the deep-linked node id changes; architecture/papers fetchers are stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestedNodeId]);

  const paperMap = useMemo(() => {
    const map = new Map<string, Paper>();
    for (const paper of papers?.papers ?? []) {
      map.set(paper.id, paper);
    }
    return map;
  }, [papers]);

  return (
    <div className={styles.wrap}>
      <div className={`page-head ${styles.headWrap}`}>
        <p className="eyebrow">Architecture</p>
        <h1>How the six steps fit together</h1>
        <p className="lede">
          Hover a stage to trace what it connects to. Click one for what it does, why it was chosen, what was
          studied for it, and the papers that informed it.
        </p>
      </div>

      <div className={styles.legendWrap}>
        <Legend />
      </div>

      {error && <p className={styles.error}>Couldn&rsquo;t load the architecture data: {error}</p>}

      {!error && !architecture && <p className={styles.loading}>Loading architecture&hellip;</p>}

      {architecture && (
        <div className={styles.explorer}>
          <FlowCanvas data={architecture} onSelectNode={setSelected} />
          {selected && <NodeDetailPanel node={selected} papers={paperMap} onClose={() => setSelected(null)} />}
        </div>
      )}
    </div>
  );
}
