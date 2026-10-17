import { getPlan } from '../api/client';
import { useFetch } from '../api/useFetch';
import { ErrorState, LoadingState } from '../components/common/AsyncState';
import { FindingCallout } from '../components/common/FindingCallout';
import styles from './PlanPage.module.css';

export function PlanPage() {
  const { data, error, loading } = useFetch(getPlan);

  return (
    <div className="page">
      <div className="page-head">
        <p className="eyebrow">My Plan</p>
        <h1>What this thesis tests, and where it stops</h1>
      </div>

      {loading && <LoadingState />}
      {error && <ErrorState message={error} />}

      {data && (
        <>
          <section className={styles.section}>
            <p className={styles.body}>{data.context}</p>
          </section>

          <section className={`card ${styles.statementCard}`}>
            <p className={styles.statementLabel}>The question</p>
            <p className={styles.statement}>{data.thesis_statement}</p>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Research questions</h2>
            <div className={styles.questionGrid}>
              {data.research_questions.map((rq, index) => (
                <div key={rq.id} className={`card ${styles.questionCard}`}>
                  <span className={styles.questionNumber}>RQ{index + 1}</span>
                  <p>{rq.text}</p>
                </div>
              ))}
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Scope guardrails</h2>
            <ul className={styles.guardrailList}>
              {data.guardrails.map((guardrail) => (
                <li key={guardrail} className={styles.guardrailItem}>
                  {guardrail}
                </li>
              ))}
            </ul>
          </section>

          <FindingCallout>{data.negative_result_framing}</FindingCallout>
        </>
      )}
    </div>
  );
}
