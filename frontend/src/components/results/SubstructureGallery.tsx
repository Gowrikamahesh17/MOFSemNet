import { useEffect, useState } from 'react';
import styles from './SubstructureGallery.module.css';

interface Item {
  rank: number;
  feature: string;
  fragment_smiles: string;
  radius: number | null;
  n_mofs_with_bit: number;
  n_distinct_fragments: number;
  fragment_share: number;
  mean_abs_shap: number;
  share_of_total_shap: number;
  mean_shap_when_present: number;
  direction: string;
  svg: string;
}

interface Payload {
  note: string;
  items: Item[];
}

export function SubstructureGallery({ url, title }: { url: string; title: string | null }) {
  const [data, setData] = useState<Payload | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: Payload) => !cancelled && setData(d))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [url]);

  if (failed) return <p className={styles.note}>The substructure data could not be loaded.</p>;
  if (!data) return null;

  return (
    <section className={styles.section}>
      <h4 className={styles.title}>{title ?? 'Most important fingerprint bits, as chemical fragments'}</h4>
      <p className={styles.note}>{data.note}</p>
      <div className={styles.grid}>
        {data.items.map((item) => (
          <article key={item.feature} className={styles.card}>
            {item.svg && <img src={item.svg} alt={`Linker with fragment ${item.fragment_smiles} highlighted`} loading="lazy" />}
            <p className={styles.head}>
              #{item.rank} <span className={styles.mono}>{item.feature}</span>
            </p>
            <p className={styles.frag}>
              <span className={styles.mono}>{item.fragment_smiles}</span> (radius {item.radius})
            </p>
            <p className={styles.meta}>
              {item.direction}
              {Math.abs(item.mean_shap_when_present) >= 0.1 && <> ({item.mean_shap_when_present > 0 ? '+' : ''}{item.mean_shap_when_present.toFixed(2)} &Aring;)</>}
              <br />
              {(item.share_of_total_shap * 100).toFixed(1)}% of the fingerprint model&rsquo;s reliance
              <br />
              in {item.n_mofs_with_bit.toLocaleString()} MOFs; main fragment in {(item.fragment_share * 100).toFixed(0)}% of linkers
              {item.n_distinct_fragments > 1 && <> ({item.n_distinct_fragments} fragments share this bit)</>}
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}
