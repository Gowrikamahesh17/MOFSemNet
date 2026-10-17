import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
  type NodeMouseHandler,
} from 'reactflow';
import 'reactflow/dist/style.css';
import type { ArchitectureData, ArchitectureNode } from '../../types/content';
import { PipelineNode, type PipelineNodeData } from './PipelineNode';
import { useDagreLayout } from './useDagreLayout';
import styles from './FlowCanvas.module.css';

const nodeTypes = { pipelineNode: PipelineNode };

interface FlowCanvasProps {
  data: ArchitectureData;
  onSelectNode: (node: ArchitectureNode) => void;
}

interface TooltipState {
  summary: string;
  x: number;
  y: number;
}

/**
 * The `fitView` boolean prop only runs once, synchronously on mount — before the flex
 * container around the canvas has necessarily been measured, which left the diagram
 * zoomed/panned to whatever the viewport happened to default to. Fitting imperatively
 * inside a layout effect + rAF, keyed on the node count, is the reliable fix: it runs
 * after the browser has actually painted the container at its final size.
 */
function useFitViewOnReady(nodeCount: number) {
  const { fitView } = useReactFlow();

  useEffect(() => {
    if (nodeCount === 0) return;
    const frame = requestAnimationFrame(() => {
      fitView({ padding: 0.12, duration: 300 });
    });
    return () => cancelAnimationFrame(frame);
  }, [nodeCount, fitView]);
}

function FlowCanvasInner({ data, onSelectNode }: FlowCanvasProps) {
  const { nodes: baseNodes, edges: baseEdges } = useDagreLayout(data);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  useFitViewOnReady(baseNodes.length);

  const connected = useMemo(() => {
    if (!hoveredId) return null;
    const ids = new Set<string>([hoveredId]);
    for (const edge of data.edges) {
      if (edge.from === hoveredId) ids.add(edge.to);
      if (edge.to === hoveredId) ids.add(edge.from);
    }
    return ids;
  }, [hoveredId, data.edges]);

  const nodes: Node<PipelineNodeData>[] = useMemo(
    () =>
      baseNodes.map((node) => ({
        ...node,
        data: {
          ...(node.data as ArchitectureNode),
          uiState: !connected ? 'default' : connected.has(node.id) ? 'highlighted' : 'dimmed',
        },
      })),
    [baseNodes, connected],
  );

  const edges: Edge[] = useMemo(
    () =>
      baseEdges.map((edge) => {
        const isConnected = connected && connected.has(edge.source) && connected.has(edge.target);
        return {
          ...edge,
          style: {
            stroke: isConnected ? 'var(--color-accent)' : 'var(--color-border-strong)',
            strokeWidth: isConnected ? 2.5 : 1.75,
            opacity: connected && !isConnected ? 0.3 : 1,
          },
        };
      }),
    [baseEdges, connected],
  );

  const handleMouseEnter: NodeMouseHandler = useCallback((event, node) => {
    setHoveredId(node.id);
    const data = node.data as PipelineNodeData;
    setTooltip({ summary: data.summary, x: event.clientX, y: event.clientY });
  }, []);

  const handleMouseMove: NodeMouseHandler = useCallback((event) => {
    setTooltip((current) => (current ? { ...current, x: event.clientX, y: event.clientY } : current));
  }, []);

  const handleMouseLeave: NodeMouseHandler = useCallback(() => {
    setHoveredId(null);
    setTooltip(null);
  }, []);

  const handleClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      onSelectNode(node.data as ArchitectureNode);
    },
    [onSelectNode],
  );

  return (
    <div className={styles.canvas}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeMouseEnter={handleMouseEnter}
        onNodeMouseMove={handleMouseMove}
        onNodeMouseLeave={handleMouseLeave}
        onNodeClick={handleClick}
        proOptions={{ hideAttribution: true }}
        minZoom={0.15}
        maxZoom={1.5}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-border)" />
        <Controls showInteractive={false} />
      </ReactFlow>
      {tooltip && (
        <div className={styles.tooltip} style={{ left: tooltip.x + 16, top: tooltip.y + 16 }}>
          {tooltip.summary}
        </div>
      )}
    </div>
  );
}

export function FlowCanvas(props: FlowCanvasProps) {
  return (
    <ReactFlowProvider>
      <FlowCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
