import { Handle, Position, type NodeProps } from 'reactflow';
import type { ArchitectureNode } from '../../types/content';
import styles from './PipelineNode.module.css';

export interface PipelineNodeData extends ArchitectureNode {
  uiState: 'default' | 'highlighted' | 'dimmed';
}

const STATUS_LABEL: Record<ArchitectureNode['status'], string> = {
  done: 'Done',
  next: 'Next',
  planned: 'Planned',
};

export function PipelineNode({ data }: NodeProps<PipelineNodeData>) {
  const [titleLine, ...rest] = data.label.split('\n');

  return (
    <div
      className={[
        styles.node,
        styles[data.status],
        data.uiState === 'dimmed' ? styles.dimmed : '',
        data.uiState === 'highlighted' ? styles.highlighted : '',
      ].join(' ')}
    >
      <Handle type="target" position={Position.Left} className={styles.handle} />
      <div className={styles.topRow}>
        <span className={`status-pill ${data.status}`}>{STATUS_LABEL[data.status]}</span>
        {data.group === 'graph' && <span className={styles.groupTag}>graph type</span>}
      </div>
      <p className={styles.title}>
        {titleLine}
        {rest.length > 0 && <span className={styles.titleSub}>{rest.join(' ')}</span>}
      </p>
      <p className={styles.summary}>{data.summary}</p>
      <Handle type="source" position={Position.Right} className={styles.handle} />
    </div>
  );
}
