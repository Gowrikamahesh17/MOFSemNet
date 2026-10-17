import { Link } from 'react-router-dom';
import { getPapers } from '../api/client';
import { useFetch } from '../api/useFetch';
import { ErrorState, LoadingState } from '../components/common/AsyncState';
import styles from './StudiedPage.module.css';

export function StudiedPage() {
  const { data, error, loading } = useFetch(getPapers);

  return (
    <div className="page">
      <div className="page-head">
        <p className="eyebrow">What I Studied</p>
        <h1>The reading that shaped each choice</h1>
        <p className="lede">
          Not a list &mdash; the story of what each body of work contributed and where it changed a decision. Full
          citations live on the <Link to="/references">References</Link> page.
        </p>
      </div>

      {loading && <LoadingState />}
      {error && <ErrorState message={error} />}

      {data?.clusters.map((cluster) => {
        const papers = data.papers.filter((paper) => paper.cluster === cluster.id);
        return (
          <section key={cluster.id} className={`card ${styles.cluster}`}>
            <h2 className={styles.clusterLabel}>{cluster.label}</h2>
            <p className={styles.narrative}>{cluster.narrative}</p>
            <div className={styles.chipRow}>
              {papers.map((paper) => (
                <Link key={paper.id} to={`/references#${paper.id}`} className={styles.chip}>
                  {paper.citation_key} {paper.title.length > 40 ? `${paper.title.slice(0, 40)}…` : paper.title}
                </Link>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
