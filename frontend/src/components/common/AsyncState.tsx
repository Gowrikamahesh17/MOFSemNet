import styles from './AsyncState.module.css';

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return <p className={styles.loading}>{label}</p>;
}

export function ErrorState({ message }: { message: string }) {
  return <p className={styles.error}>Couldn&rsquo;t load this page&rsquo;s data: {message}</p>;
}
