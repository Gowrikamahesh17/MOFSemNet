import styles from './StatTile.module.css';

interface StatTileProps {
  label: string;
  value: string;
  emphasis?: boolean;
}

export function StatTile({ label, value, emphasis }: StatTileProps) {
  return (
    <div className={`${styles.tile} ${emphasis ? styles.emphasis : ''}`}>
      <p className={styles.label}>{label}</p>
      <p className={styles.value}>{value}</p>
    </div>
  );
}
