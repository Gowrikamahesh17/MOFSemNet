import { getProgress } from '../api/client';
import { useFetch } from '../api/useFetch';
import { ErrorState, LoadingState } from '../components/common/AsyncState';
import styles from './ProgressPage.module.css';

const STATUS_LABEL: Record<string, string> = { done: 'Done', next: 'Next', planned: 'Not started' };

export function ProgressPage() {
  const { data, error, loading } = useFetch(getProgress);

  return (
    <div className="page">
      <div className="page-head">
        <p className="eyebrow">Progress</p>
        <h1>Against the registered timeline</h1>
        <p className="lede">
          The same work packages from the thesis expos&eacute;&rsquo;s Gantt chart, recolored by actual status.
        </p>
      </div>

      {loading && <LoadingState />}
      {error && <ErrorState message={error} />}

      {data && (
        <>
          <div className={`card ${styles.tableCard}`}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Work package</th>
                  <th>Planned window</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {data.work_packages.map((wp) => (
                  <tr key={wp.id} className={wp.status === 'done' ? styles.doneRow : undefined}>
                    <td>{wp.label}</td>
                    <td className={styles.window}>{wp.window}</td>
                    <td>
                      <span className={`status-pill ${wp.status}`}>{STATUS_LABEL[wp.status]}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className={styles.milestoneBanner}>
            <strong>Ahead of plan.</strong> {data.ahead_of_schedule_note}
          </div>

          <section className={styles.milestonesSection}>
            <h2 className={styles.sectionTitle}>Milestones</h2>
            <ol className={styles.timeline}>
              {data.milestones.map((milestone) => (
                <li key={milestone.date} className={styles.timelineItem}>
                  <span className={styles.timelineDate}>{milestone.date}</span>
                  <span className={styles.timelineLabel}>{milestone.label}</span>
                </li>
              ))}
            </ol>
          </section>
        </>
      )}
    </div>
  );
}
