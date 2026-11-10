import { useState } from 'react';
import { embedDemo, getDemoMetals } from '../../api/client';
import { useFetch } from '../../api/useFetch';
import type { EmbeddingDemoResponse } from '../../types/content';
import styles from './EmbeddingDemo.module.css';

const EXAMPLES = [
  { label: 'Benzoic acid', smiles: 'c1ccc(cc1)C(=O)O' },
  { label: 'Terephthalic acid', smiles: 'c1cc(ccc1C(=O)O)C(=O)O' },
  { label: 'Imidazole', smiles: 'c1cnc[nH]1' },
];

export function EmbeddingDemo() {
  const [smiles, setSmiles] = useState('');
  const [metal, setMetal] = useState('');
  const { data: metals } = useFetch(getDemoMetals, []);
  const [result, setResult] = useState<EmbeddingDemoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function runDemo(input: string) {
    setLoading(true);
    setError(null);
    try {
      const response = await embedDemo(input, metal || null);
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (smiles.trim()) runDemo(smiles);
  }

  return (
    <div className={`card ${styles.card}`}>
      <div className={styles.header}>
        <span className={styles.liveTag}>Live model</span>
        <h3 className={styles.title}>Try the real ChemBERTa embedder</h3>
      </div>
      <p className={styles.description}>
        Type any linker SMILES. It&rsquo;s embedded live with the real, trained ChemBERTa model, then compared by
        cosine similarity against the real cached embeddings for all 14,296 MOFs in the dataset &mdash; the nearest
        matches below are genuine MOFs, not a canned example.
      </p>

      <form onSubmit={handleSubmit} className={styles.form}>
        <input
          type="text"
          value={smiles}
          onChange={(event) => setSmiles(event.target.value)}
          placeholder="e.g. c1ccc(cc1)C(=O)O"
          className={styles.input}
          spellCheck={false}
          aria-label="Linker SMILES"
        />
        <select
          value={metal}
          onChange={(event) => setMetal(event.target.value)}
          className={styles.select}
          aria-label="Filter nearest MOFs by metal"
        >
          <option value="">Any metal{metals ? ` (${metals.length})` : ''}</option>
          {metals?.map((option) => (
            <option key={option.metal} value={option.metal}>
              {option.metal} &middot; {option.mofs.toLocaleString()} MOFs
            </option>
          ))}
        </select>
        <button type="submit" className={styles.submit} disabled={loading || !smiles.trim()}>
          {loading ? 'Embedding…' : 'Embed it'}
        </button>
      </form>

      <div className={styles.examples}>
        <span className={styles.examplesLabel}>Try:</span>
        {EXAMPLES.map((example) => (
          <button
            key={example.smiles}
            type="button"
            className={styles.exampleChip}
            onClick={() => setSmiles(example.smiles)}
            disabled={loading}
          >
            {example.label}
          </button>
        ))}
      </div>

      <p className={styles.hint}>
        The embedding reads the linker only; metal is a separate one-hot code. Picking a metal just limits which real MOFs
        can appear as nearest matches.
      </p>

      {error && <p className={styles.error}>{error}</p>}

      {result && (
        <div className={styles.result}>
          <div className={styles.resultMeta}>
            <span>
              Model: <strong>{result.model}</strong>
            </span>
            <span>
              Embedding dim: <strong>{result.embedding_dim}</strong>
            </span>
            <span>
              Vector norm: <strong>{result.embedding_norm}</strong>
            </span>
          </div>
          <p className={styles.resultLabel}>
            Nearest real MOFs by cosine similarity{result.metal_filter ? ` · metal: ${result.metal_filter}` : ''}
          </p>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Refcode</th>
                <th>Linker SMILES</th>
                <th>Metal</th>
                <th>PLD (Å)</th>
                <th>Similarity</th>
              </tr>
            </thead>
            <tbody>
              {result.neighbors.map((neighbor) => (
                <tr key={neighbor.refcode}>
                  <td className={styles.mono}>{neighbor.refcode}</td>
                  <td className={styles.mono}>{neighbor.linker_smiles}</td>
                  <td>{neighbor.metal}</td>
                  <td className={styles.mono}>{neighbor.pld_value.toFixed(2)}</td>
                  <td className={styles.mono}>{neighbor.cosine_similarity.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
