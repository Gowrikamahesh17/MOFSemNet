import { useState } from 'react';
import { getGraphComparison, graphStructureUrl } from '../../api/client';
import { useFetch } from '../../api/useFetch';
import type { GraphNeighbourhood } from '../../types/content';
import { ErrorState, LoadingState } from '../common/AsyncState';
import { FindingCallout } from '../common/FindingCallout';
import { StatTile } from '../common/StatTile';
import styles from './GraphComparison.module.css';

const WIDTH = 400;
const HEIGHT = 300;
const PAD = 16;

type Kind = 'fingerprint' | 'semantic';

function edgeKey(a: number, b: number): string {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

function degree(hood: GraphNeighbourhood, kind: Kind, index: number): number {
  return hood[kind].filter(([a, b]) => a === index || b === index).length;
}

interface PanelProps {
  hood: GraphNeighbourhood;
  kind: Kind;
  other: Kind;
  title: string;
  tag: string;
  hovered: number | null;
  onHover: (index: number | null) => void;
}

function NetworkPanel({ hood, kind, other, title, tag, hovered, onHover }: PanelProps) {
  const otherKeys = new Set(hood[other].map(([a, b]) => edgeKey(a, b)));
  const x = (n: number) => PAD + hood.nodes[n].x * (WIDTH - 2 * PAD);
  const y = (n: number) => PAD + hood.nodes[n].y * (HEIGHT - 2 * PAD);

  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <h4>{title}</h4>
        <span className={styles.panelTag}>{tag}</span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${title} neighbourhood`}>
        {hood[kind].map(([a, b]) => {
          const shared = otherKeys.has(edgeKey(a, b));
          return (
            <line
              key={edgeKey(a, b)}
              x1={x(a)}
              y1={y(a)}
              x2={x(b)}
              y2={y(b)}
              className={shared ? styles.edgeShared : kind === 'fingerprint' ? styles.edgeFingerprint : styles.edgeSemantic}
            />
          );
        })}
        {hood.nodes.map((node, index) => (
          <circle
            // eslint-disable-next-line react/no-array-index-key
            key={index}
            cx={x(index)}
            cy={y(index)}
            r={3 + Math.min(7, Math.sqrt(node.mofs) * 0.7)}
            className={`${styles.node} ${index === hood.seed ? styles.seed : ''} ${hovered === index ? styles.hot : ''}`}
            onMouseEnter={() => onHover(index)}
            onMouseLeave={() => onHover(null)}
          />
        ))}
      </svg>
    </div>
  );
}

const MODELS = [
  { id: 'chemberta', label: 'ChemBERTa', note: 'primary' },
  { id: 'molt5', label: 'MolT5', note: 'secondary' },
];

export function GraphComparison() {
  const [model, setModel] = useState('chemberta');
  const { data, error, loading } = useFetch(() => getGraphComparison(model), [model]);
  const [active, setActive] = useState(0);
  const [hovered, setHovered] = useState<number | null>(null);

  if (loading) return <LoadingState label="Loading graph comparison…" />;
  if (error || !data) return <ErrorState message={error ?? 'no data'} />;

  const { stats } = data;
  const modelLabel = MODELS.find((m) => m.id === model)?.label ?? model;
  const hood = data.neighbourhoods[active];
  const linkerUnion =
    stats.linker_pair_edges.fingerprint + stats.linker_pair_edges.semantic - stats.linker_pair_edges.shared;
  const fp = hood.fingerprint.length;
  const sem = hood.semantic.length;
  const shared = hood.shared_edges;
  const node = hovered !== null ? hood.nodes[hovered] : null;
  const pct = (value: number) => `${Math.round(value * 100)}%`;

  return (
    <div className={`card ${styles.card}`}>
      <div className={styles.header}>
        <span className={styles.realTag}>Real graphs</span>
        <h3 className={styles.title}>Fingerprint graph vs semantic graph</h3>
      </div>
      <p className={styles.description}>
        The same linkers, connected two ways: by Morgan-fingerprint similarity (the Case Study graph) and by {modelLabel}
        embedding similarity (the new graph). Drawn at the linker level, one dot per distinct linker, because {pct(stats.same_linker_edge_share.fingerprint)} of
        MOF-level edges just join MOFs that share the identical linker. Both panels use the same positions.
      </p>

      <div className={styles.tabs} role="tablist" aria-label="Language model">
        {MODELS.map((m) => (
          <button
            key={m.id}
            type="button"
            role="tab"
            aria-selected={m.id === model}
            className={`${styles.tab} ${m.id === model ? styles.tabOn : ''}`}
            onClick={() => {
              setModel(m.id);
              setActive(0);
              setHovered(null);
            }}
          >
            {m.label} <span className={styles.modelNote}>{m.note}</span>
          </button>
        ))}
      </div>

      <div className={styles.statGrid}>
        <StatTile label="Edge overlap (all MOF edges)" value={stats.mof_jaccard.toFixed(2)} />
        <StatTile label="Edge overlap (different linkers only)" value={stats.linker_pair_jaccard.toFixed(2)} emphasis />
        <StatTile
          label="Same-linker edges (fingerprint / semantic)"
          value={`${pct(stats.same_linker_edge_share.fingerprint)} / ${pct(stats.same_linker_edge_share.semantic)}`}
        />
        <StatTile label="Connected pieces (fingerprint / semantic)" value={`${stats.components.fingerprint} / ${stats.components.semantic}`} />
      </div>

      <div className={styles.tabs} role="tablist" aria-label="Linker neighbourhood">
        {data.neighbourhoods.map((_, index) => (
          <button
            // eslint-disable-next-line react/no-array-index-key
            key={index}
            type="button"
            role="tab"
            aria-selected={index === active}
            className={`${styles.tab} ${index === active ? styles.tabOn : ''}`}
            onClick={() => {
              setActive(index);
              setHovered(null);
            }}
          >
            Neighbourhood {index + 1}
          </button>
        ))}
      </div>

      <div className={styles.panels}>
        <NetworkPanel hood={hood} kind="fingerprint" other="semantic" title="Fingerprint graph" tag="Case Study" hovered={hovered} onHover={setHovered} />
        <NetworkPanel hood={hood} kind="semantic" other="fingerprint" title="Semantic graph" tag={modelLabel} hovered={hovered} onHover={setHovered} />
      </div>

      <div className={styles.foot}>
        <ul className={styles.legend}>
          <li><i className={styles.swShared} />In both graphs ({shared})</li>
          <li><i className={styles.swFingerprint} />Fingerprint only ({fp - shared})</li>
          <li><i className={styles.swSemantic} />Semantic only ({sem - shared})</li>
          <li className={styles.legendNote}>
            {hood.nodes.length} linkers · dot size = MOFs using it · outlined dot = starting linker · hover a dot
          </li>
        </ul>
        <div className={styles.info}>
          {node && hovered !== null ? (
            <>
              <img src={graphStructureUrl(node.smiles)} alt="Linker structure" className={styles.structure} />
              <div>
                <p>
                  <strong>
                    {node.mofs} MOF{node.mofs === 1 ? '' : 's'}
                  </strong>{' '}
                  · mean PLD {node.mean_pld.toFixed(2)} Å
                </p>
                <p>
                  {degree(hood, 'fingerprint', hovered)} links in fingerprint graph · {degree(hood, 'semantic', hovered)} in semantic graph
                </p>
                <p className={styles.smiles}>{node.smiles}</p>
              </div>
            </>
          ) : (
            <p className={styles.hint}>Hover a linker to see its structure.</p>
          )}
        </div>
      </div>

      <FindingCallout>
        Different, but not yet shown to be better. Only {stats.linker_pair_edges.shared.toLocaleString()} of{' '}
        {linkerUnion.toLocaleString()} links between different linkers appear in both graphs, so the semantic graph really does pick other
        neighbours. But connected MOFs differ in PLD by about the same amount in both graphs
        ({stats.mean_abs_pld_gap_between_neighbours.fingerprint.toFixed(2)} Å fingerprint vs{' '}
        {stats.mean_abs_pld_gap_between_neighbours.semantic.toFixed(2)} Å semantic; random pairs{' '}
        {stats.mean_abs_pld_gap_random_pair.toFixed(2)} Å). That is a graph statistic, not a model result &mdash; Step 5 decides whether the new
        edges help prediction.
      </FindingCallout>
    </div>
  );
}
