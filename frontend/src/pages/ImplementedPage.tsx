import { Fragment } from 'react';
import { getStepResult } from '../api/client';
import { useFetch } from '../api/useFetch';
import { ErrorState, LoadingState } from '../components/common/AsyncState';
import { EmbeddingDemo } from '../components/demo/EmbeddingDemo';
import { GraphComparison } from '../components/graphs/GraphComparison';
import { StepResultCard } from '../components/results/StepResultCard';

const STEP_IDS = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6'];

export function ImplementedPage() {
  const { data, error, loading } = useFetch(
    () => Promise.all(STEP_IDS.map((id) => getStepResult(id))),
    [],
  );

  return (
    <div className="page">
      <div className="page-head">
        <p className="eyebrow">What I&rsquo;ve Implemented</p>
        <h1>Six steps, run on the real 14,296-MOF dataset</h1>
        <p className="lede">
          Precomputed results below &mdash; graph construction and sparsification take 20&ndash;90 seconds each, not
          demo-safe live. The embedding step is fast enough to run for real, right now. Steps 5 and 6 are the
          results: every choice that was tried, and what it scored, is in the{' '}
          <a href="/decision_tree.html" target="_blank" rel="noreferrer">interactive decision tree</a>.
        </p>
      </div>

      <EmbeddingDemo />

      {loading && <LoadingState />}
      {error && <ErrorState message={error} />}

      {data?.map((result, index) =>
        result ? (
          <Fragment key={STEP_IDS[index]}>
            <StepResultCard result={result} />
            {STEP_IDS[index] === 'step3' && <GraphComparison />}
          </Fragment>
        ) : null,
      )}
    </div>
  );
}
