import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { getArchitecture, getPapers, paperPdfUrl } from '../api/client';
import { useFetch } from '../api/useFetch';
import { ErrorState, LoadingState } from '../components/common/AsyncState';
import styles from './ReferencesPage.module.css';

export function ReferencesPage() {
  const papersState = useFetch(getPapers);
  const architectureState = useFetch(getArchitecture);

  const nodesByPaper = useMemo(() => {
    const map = new Map<string, { id: string; label: string }[]>();
    for (const node of architectureState.data?.nodes ?? []) {
      for (const paperId of node.references) {
        const list = map.get(paperId) ?? [];
        list.push({ id: node.id, label: node.label.split('\n').join(' ') });
        map.set(paperId, list);
      }
    }
    return map;
  }, [architectureState.data]);

  const loading = papersState.loading || architectureState.loading;
  const error = papersState.error ?? architectureState.error;

  return (
    <div className="page">
      <div className="page-head">
        <p className="eyebrow">References</p>
        <h1>Every paper, and what it informed</h1>
        <p className="lede">
          All 11 cited papers. Each one links to the architecture stage(s) it actually shaped &mdash; not a flat
          bibliography.
        </p>
      </div>

      {loading && <LoadingState />}
      {error && <ErrorState message={error} />}

      <div className={styles.list}>
        {papersState.data?.papers.map((paper) => {
          const linkedNodes = nodesByPaper.get(paper.id) ?? [];
          return (
            <article key={paper.id} id={paper.id} className={`card ${styles.paper}`}>
              <div className={styles.topRow}>
                <span className={styles.key}>{paper.citation_key}</span>
                <span className={styles.year}>{paper.year}</span>
              </div>
              <h2 className={styles.title}>{paper.title}</h2>
              <p className={styles.authors}>
                {paper.authors} &middot; <span className={styles.venue}>{paper.venue}</span>
              </p>
              <p className={styles.summary}>{paper.summary}</p>
              <div className={styles.footer}>
                <div className={styles.links}>
                  <a href={paper.url} target="_blank" rel="noreferrer" className={styles.externalLink}>
                    View source &rarr;
                  </a>
                  {paper.has_local_pdf && (
                    <a href={paperPdfUrl(paper.id)} target="_blank" rel="noreferrer" className={styles.externalLink}>
                      PDF &rarr;
                    </a>
                  )}
                </div>
                {linkedNodes.length > 0 && (
                  <div className={styles.informed}>
                    <span className={styles.informedLabel}>Informed:</span>
                    {linkedNodes.map((node) => (
                      <Link key={node.id} to={`/architecture?node=${node.id}`} className={styles.nodeChip}>
                        {node.label}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
