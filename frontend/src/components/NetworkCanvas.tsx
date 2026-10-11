import React, { useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  Position,
  Handle,
} from '@xyflow/react';
import type { Node, Edge, NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { ModelView } from '../types';

// Custom circular neuron component with left and right connection handles
const NeuronNode = ({ data, selected }: NodeProps) => {
  return (
    <div
      className={`w-7 h-7 rounded-full flex items-center justify-center font-mono text-[10px] transition-all ${
        selected
          ? 'bg-brand-amber text-zinc-950 font-bold ring-4 ring-brand-amber/40 border border-white'
          : 'bg-surface-850 text-zinc-400 border border-zinc-600 hover:border-zinc-300'
      }`}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!w-1.5 !h-1.5 !bg-zinc-500 !border-0"
      />
      <span>{data.index as number}</span>
      <Handle
        type="source"
        position={Position.Right}
        className="!w-1.5 !h-1.5 !bg-zinc-500 !border-0"
      />
    </div>
  );
};

const nodeTypes = {
  neuron: NeuronNode,
};

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
  const [viewMode, setViewMode] = useState<'neuron' | 'layer'>('neuron');
  const [hoveredWeight, setHoveredWeight] = useState<{
    from: string;
    to: string;
    weight: number;
  } | null>(null);

  const { nodes, edges } = useMemo(() => {
    if (!modelView) return { nodes: [], edges: [] };

    // --- High-Level Layer Block View ---
    if (viewMode === 'layer') {
      const flowNodes: Node[] = modelView.nodes.map((node, index) => {
        const isSelected = node.id === selectedNodeId;
        return {
          id: node.id,
          position: { x: 300, y: index * 140 + 60 },
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
            width: 240,
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
    }

    // --- Individual Neuron View (TensorFlow Playground style) ---
    const neuronNodes: Node[] = [];
    const synapseEdges: Edge[] = [];

    const layers = modelView.nodes;
    const X_SPACING = 320;
    const Y_SPACING = 36;

    const columnData: {
      name: string;
      count: number;
      layerIdx: number;
      kernelWeights?: number[][];
    }[] = [];

    // Derive input neuron count
    let inputCount = 4;
    const inputShape = modelView.meta.input_shapes?.[0];
    if (Array.isArray(inputShape) && inputShape.length > 1 && inputShape[1] !== null) {
      inputCount = Number(inputShape[1]);
    } else if (layers.length > 0 && layers[0].attrs?.weights) {
      // Deduce input count from kernel matrix rows if available
      const firstKernelKey = Object.keys(layers[0].attrs.weights).find((k) =>
        k.endsWith('kernel:0') || k.endsWith('kernel')
      );
      if (firstKernelKey && layers[0].attrs.weights[firstKernelKey]?.shape?.[0]) {
        inputCount = layers[0].attrs.weights[firstKernelKey].shape[0];
      }
    }

    columnData.push({
      name: 'Input Layer',
      count: inputCount,
      layerIdx: -1,
    });

    layers.forEach((layer, lIdx) => {
      let count = 1;

      // Extract neuron count from units, shape, or kernel shape
      if (layer.attrs?.units) {
        count = Number(layer.attrs.units);
      } else if (Array.isArray(layer.attrs?.shape) && layer.attrs.shape.length >= 2) {
        const lastDim = layer.attrs.shape[layer.attrs.shape.length - 1];
        if (lastDim !== null && !isNaN(Number(lastDim))) {
          count = Number(lastDim);
        }
      }

      // Check kernel weights
      let kernel: number[][] | undefined;
      if (layer.attrs?.weights) {
        const kernelKey = Object.keys(layer.attrs.weights).find((k) =>
          k.endsWith('kernel:0') || k.endsWith('kernel')
        );
        if (kernelKey) {
          const wInfo = layer.attrs.weights[kernelKey];
          if (wInfo.shape && wInfo.shape[1]) {
            count = Number(wInfo.shape[1]);
          }
          if (wInfo.values) {
            kernel = wInfo.values;
          }
        }
      }

      columnData.push({
        name: layer.label,
        count: count,
        layerIdx: lIdx,
        kernelWeights: kernel,
      });
    });

    const maxNeurons = Math.max(...columnData.map((c) => c.count));
    const maxHeight = maxNeurons * Y_SPACING;

    // Place column headers and neuron circles
    columnData.forEach((col, colIdx) => {
      const colX = colIdx * X_SPACING + 80;
      const colOffset = (maxHeight - col.count * Y_SPACING) / 2;

      // Column Header
      neuronNodes.push({
        id: `header_${colIdx}`,
        position: { x: colX - 45, y: 15 },
        selectable: false,
        draggable: false,
        data: {
          label: (
            <div className="text-center w-32 select-none">
              <div className="text-xs font-mono font-bold text-brand-amber truncate">
                {col.name}
              </div>
              <div className="text-[11px] text-zinc-500 font-mono">
                {col.count} {col.count === 1 ? 'neuron' : 'neurons'}
              </div>
            </div>
          ),
        },
        style: {
          background: 'transparent',
          border: 'none',
        },
      });

      // Individual neurons
      for (let nIdx = 0; nIdx < col.count; nIdx++) {
        const nodeId = `col_${colIdx}_neuron_${nIdx}`;
        const neuronY = colOffset + nIdx * Y_SPACING + 70;

        neuronNodes.push({
          id: nodeId,
          type: 'neuron',
          position: { x: colX, y: neuronY },
          data: { index: nIdx },
        });
      }
    });

    // Generate straight synapse edges connecting adjacent columns
    for (let c = 0; c < columnData.length - 1; c++) {
      const srcCol = columnData[c];
      const targetCol = columnData[c + 1];
      const weights = targetCol.kernelWeights;

      for (let i = 0; i < srcCol.count; i++) {
        for (let j = 0; j < targetCol.count; j++) {
          const srcId = `col_${c}_neuron_${i}`;
          const targetId = `col_${c + 1}_neuron_${j}`;

          let wVal = 0.0;
          if (weights && weights[i] && weights[i][j] !== undefined) {
            wVal = weights[i][j];
          }

          // Thickness and opacity scaled by weight magnitude
          const absW = Math.abs(wVal);
          const strokeWidth = Math.min(Math.max(absW * 2.2, 0.7), 4.5);
          const strokeColor = wVal >= 0 ? '#38bdf8' : '#f59e0b';
          const opacity = Math.min(Math.max(absW * 1.2, 0.25), 0.9);

          synapseEdges.push({
            id: `edge_${srcId}_${targetId}`,
            source: srcId,
            target: targetId,
            type: 'straight',
            animated: false,
            style: {
              stroke: strokeColor,
              strokeWidth: strokeWidth,
              opacity: opacity,
            },
            data: { weight: wVal },
          });
        }
      }
    }

    return { nodes: neuronNodes, edges: synapseEdges };
  }, [modelView, viewMode, selectedNodeId]);

  return (
    <div className="w-full h-full bg-surface-950 relative">
      {/* View Switcher Toggle */}
      <div className="absolute top-4 right-4 z-20 flex space-x-1.5 bg-surface-900 border border-surface-800 p-1 rounded-lg">
        <button
          onClick={() => setViewMode('neuron')}
          className={`px-3 py-1.5 text-xs font-medium rounded transition ${
            viewMode === 'neuron'
              ? 'bg-brand-amber text-zinc-950 font-bold shadow'
              : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Neuron View (Playground)
        </button>
        <button
          onClick={() => setViewMode('layer')}
          className={`px-3 py-1.5 text-xs font-medium rounded transition ${
            viewMode === 'layer'
              ? 'bg-brand-amber text-zinc-950 font-bold shadow'
              : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Layer Block View
        </button>
      </div>

      {/* Hover Weight Tooltip Banner */}
      {hoveredWeight && (
        <div className="absolute bottom-6 left-6 z-20 bg-surface-900/95 border border-brand-amber/50 px-3.5 py-2 rounded-lg shadow-xl text-xs font-mono backdrop-blur">
          <span className="text-zinc-400">Synapse Weight: </span>
          <span
            className={
              hoveredWeight.weight >= 0
                ? 'text-sky-400 font-bold'
                : 'text-amber-400 font-bold'
            }
          >
            {hoveredWeight.weight.toFixed(5)}
          </span>
        </div>
      )}

      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => onSelectNode(node.id)}
        onEdgeMouseEnter={(_, edge) => {
          if (edge.data && typeof edge.data.weight === 'number') {
            setHoveredWeight({
              from: edge.source,
              to: edge.target,
              weight: edge.data.weight,
            });
          }
        }}
        onEdgeMouseLeave={() => setHoveredWeight(null)}
        fitView
      >
        <Background color="#27272a" gap={20} size={1} />
        <Controls className="!bg-surface-850 !border-surface-800 !text-zinc-200 fill-zinc-200" />
      </ReactFlow>
    </div>
  );
};