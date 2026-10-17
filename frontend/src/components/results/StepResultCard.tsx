import { FindingCallout } from '../common/FindingCallout';
import { StatTile } from '../common/StatTile';
import type { StepResult } from '../../types/content';
import styles from './StepResultCard.module.css';

export function StepResultCard({ result }: { result: StepResult }) {
  return (
    <div className={`card ${styles.card}`}>
      <div className={styles.header}>
        <span className={`status-pill ${result.status}`}>{result.status === 'done' ? 'Done' : result.status}</span>
        <h3 className={styles.title}>{result.title}</h3>
      </div>
      <p className={styles.summary}>{result.summary}</p>

      {result.headline && (
        <div className={styles.headline}>
          <p className={styles.headlineValue}>{result.headline}</p>
          <p className={styles.headlineLabel}>{result.headline_label}</p>
        </div>
      )}

      {result.stats.length > 0 && (
        <div className={styles.statGrid}>
          {result.stats.map((stat) => (
            <StatTile key={stat.label} label={stat.label} value={stat.value} emphasis={stat.emphasis} />
          ))}
        </div>
      )}

      {result.rows.length > 0 && (
        <div className={styles.tableWrap}>
          {result.retention_shown && <p className={styles.tableCaption}>Retention shown: {result.retention_shown}</p>}
          <table className={styles.table}>
            <thead>
              <tr>
                {result.columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, rowIndex) => (
                // eslint-disable-next-line react/no-array-index-key
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    // eslint-disable-next-line react/no-array-index-key
                    <td key={cellIndex}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result.finding && <FindingCallout>{result.finding}</FindingCallout>}
    </div>
  );
}
