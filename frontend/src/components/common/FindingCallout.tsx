import type { ReactNode } from 'react';
import styles from './FindingCallout.module.css';

export function FindingCallout({ children }: { children: ReactNode }) {
  return <div className={styles.callout}>{children}</div>;
}
