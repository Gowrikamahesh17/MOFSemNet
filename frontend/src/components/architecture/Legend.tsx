import styles from './Legend.module.css';

export function Legend() {
  return (
    <div className={styles.legend}>
      <span className={styles.item}>
        <span className={`${styles.swatch} ${styles.done}`} />
        Done
      </span>
      <span className={styles.item}>
        <span className={`${styles.swatch} ${styles.next}`} />
        Next up
      </span>
      <span className={styles.item}>
        <span className={`${styles.swatch} ${styles.planned}`} />
        Planned
      </span>
      <span className={styles.hint}>Hover a stage to trace its connections &middot; click for detail</span>
    </div>
  );
}
