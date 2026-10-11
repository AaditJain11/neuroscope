import React, { useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
} from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { ModelView } from '../types';

interface CanvasProps {
  modelView: ModelView | null;
  onSelectNode: (nodeId: string) => void;
  selectedNodeId: string | null;
}

export const NetworkCanvas: React.FC<CanvasProps> = ({
  modelView,
  onSelectNode,
  selectedNodeId,
}) => {
  const { nodes, edges } = useMemo(() => {
    if (!modelView) return { nodes: [], edges: [] };

    const flowNodes: Node[] = modelView.nodes.map((node, index) => {
      const isSelected = node.id === selectedNodeId;
      return {
        id: node.id,
        position: { x: 250, y: index * 120 + 50 },
        data: {
          label: (
            <div className="flex flex-col text-left">
              <span className="text-xs uppercase tracking-wide text-brand-amber font-mono">
                {node.type}
              </span>
              <span className="font-semibold text-sm text-zinc-100">
                {node.label}
              </span>
              {node.attrs?.shape && (
                <span className="text-[11px] text-zinc-400 mt-1 font-mono">
                  Shape: {JSON.stringify(node.attrs.shape)}
                </span>
              )}
            </div>
          ),
        },
        style: {
          background: isSelected ? '#18181c' : '#121215',
          border: isSelected ? '2px solid #f59e0b' : '1px solid #27272a',
          borderRadius: '8px',
          padding: '12px',
          width: 220,
          boxShadow: isSelected
            ? '0 0 15px rgba(245, 158, 11, 0.25)'
            : '0 2px 5px rgba(0,0,0,0.4)',
        },
      };
    });

    const flowEdges: Edge[] = modelView.edges.map((edge, idx) => ({
      id: `e-${edge.source}-${edge.target}-${idx}`,
      source: edge.source,
      target: edge.target,
      animated: true,
      style: { stroke: '#f59e0b', strokeWidth: 1.5 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#f59e0b',
      },
    }));

    return { nodes: flowNodes, edges: flowEdges };
  }, [modelView, selectedNodeId]);

  return (
    <div className="w-full h-full bg-surface-950">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={(_, node) => onSelectNode(node.id)}
        fitView
      >
        <Background color="#27272a" gap={20} size={1} />
        <Controls className="!bg-surface-850 !border-surface-800 !text-zinc-200 fill-zinc-200" />
      </ReactFlow>
    </div>
  );
};