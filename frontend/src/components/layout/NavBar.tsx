import { NavLink } from 'react-router-dom';
import styles from './NavBar.module.css';

const SECTIONS = [
  { to: '/plan', label: 'Plan' },
  { to: '/architecture', label: 'Architecture' },
  { to: '/implemented', label: 'Implemented' },
  { to: '/studied', label: 'Studied' },
  { to: '/references', label: 'References' },
  { to: '/progress', label: 'Progress' },
];

export function NavBar() {
  return (
    <header className={styles.bar}>
      <div className={styles.inner}>
        <NavLink to="/plan" className={styles.brand}>
          <span className={styles.brandMark}>MS</span>
          <span className={styles.brandText}>MOFSemNet</span>
        </NavLink>
        <nav className={styles.nav} aria-label="Main sections">
          {SECTIONS.map((section) => (
            <NavLink
              key={section.to}
              to={section.to}
              className={({ isActive }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)}
            >
              {section.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
