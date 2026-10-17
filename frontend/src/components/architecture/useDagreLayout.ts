import dagre from '@dagrejs/dagre';
import { useMemo } from 'react';
import type { Edge, Node } from 'reactflow';
import { Position } from 'reactflow';
import type { ArchitectureData } from '../../types/content';

const NODE_WIDTH = 240;
const NODE_HEIGHT = 92;

/**
 * Lays out architecture.yaml's nodes/edges with dagre (layered, left-to-right) so adding a
 * node to the content file repositions the whole diagram automatically — no layout code to
 * touch when a new pipeline stage, graph type, or comparison method is added.
 */
export function useDagreLayout(data: ArchitectureData | undefined): { nodes: Node[]; edges: Edge[] } {
  return useMemo(() => {
    if (!data) {
      return { nodes: [], edges: [] };
    }

    const graph = new dagre.graphlib.Graph();
    graph.setDefaultEdgeLabel(() => ({}));
    graph.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 70, marginx: 24, marginy: 24 });

    for (const node of data.nodes) {
      graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
    }
    for (const edge of data.edges) {
      graph.setEdge(edge.from, edge.to);
    }

    dagre.layout(graph);

    const nodes: Node[] = data.nodes.map((node) => {
      const position = graph.node(node.id);
      return {
        id: node.id,
        type: 'pipelineNode',
        position: { x: position.x - NODE_WIDTH / 2, y: position.y - NODE_HEIGHT / 2 },
        // Explicit width/height, not just position: fitView (and dagre's own bounds) use
        // these immediately instead of waiting for a DOM ResizeObserver measurement pass,
        // which otherwise races fitView on first load and fits only whichever nodes had
        // already been measured by the time it ran.
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        data: node,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
    });

    const edges: Edge[] = data.edges.map((edge) => ({
      id: `${edge.from}-${edge.to}`,
      source: edge.from,
      target: edge.to,
      animated: edge.animated,
      style: { stroke: 'var(--color-border-strong)', strokeWidth: 1.75 },
    }));

    return { nodes, edges };
  }, [data]);
}
