import { Link } from 'react-router-dom';
import type { ArchitectureNode, Paper } from '../../types/content';
import styles from './NodeDetailPanel.module.css';

interface NodeDetailPanelProps {
  node: ArchitectureNode;
  papers: Map<string, Paper>;
  onClose: () => void;
}

const STATUS_LABEL: Record<ArchitectureNode['status'], string> = {
  done: 'Done',
  next: 'Next up',
  planned: 'Planned',
};

export function NodeDetailPanel({ node, papers, onClose }: NodeDetailPanelProps) {
  const hasMetrics = Object.keys(node.metrics).length > 0;
  const linkedPapers = node.references.map((id) => papers.get(id)).filter((p): p is Paper => Boolean(p));

  return (
    <aside className={styles.panel} aria-label={`Details for ${node.label.replace('\n', ' ')}`}>
      <div className={styles.header}>
        <span className={`status-pill ${node.status}`}>{STATUS_LABEL[node.status]}</span>
        <button type="button" className={styles.close} onClick={onClose} aria-label="Close details">
          &times;
        </button>
      </div>

      <h2 className={styles.title}>{node.label.split('\n').join(' ')}</h2>
      <p className={styles.description}>{node.description}</p>

      {node.why_chosen && (
        <section className={styles.section}>
          <h3 className={styles.sectionLabel}>Why this approach</h3>
          <p className={styles.sectionBody}>{node.why_chosen}</p>
        </section>
      )}

      {node.studied.length > 0 && (
        <section className={styles.section}>
          <h3 className={styles.sectionLabel}>What I studied for it</h3>
          <ul className={styles.list}>
            {node.studied.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      )}

      {hasMetrics && (
        <section className={styles.section}>
          <h3 className={styles.sectionLabel}>Measured so far</h3>
          <dl className={styles.metrics}>
            {Object.entries(node.metrics).map(([label, value]) => (
              <div key={label} className={styles.metricRow}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {linkedPapers.length > 0 && (
        <section className={styles.section}>
          <h3 className={styles.sectionLabel}>Related references</h3>
          <ul className={styles.refList}>
            {linkedPapers.map((paper) => (
              <li key={paper.id}>
                <Link to={`/references#${paper.id}`} className={styles.refLink}>
                  <span className={styles.refKey}>{paper.citation_key}</span>
                  {paper.title}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      {node.status !== 'done' && !node.why_chosen && (
        <p className={styles.plannedNote}>
          Not started yet &mdash; design notes and references will fill in once this stage begins.
        </p>
      )}
    </aside>
  );
}
