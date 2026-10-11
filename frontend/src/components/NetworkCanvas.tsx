import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import {
  Layers,
  Network,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Edit3,
  Check,
  X,
  ChevronUp,
  ChevronDown,
  Grid,
  Plus,
} from 'lucide-react';
import type { ModelView } from '../types';

interface CanvasProps {
  modelView: ModelView | null;
  modelId: string | null;
  modelVersion: number;
  onSelectNode: (nodeId: string) => void;
  selectedNodeId: string | null;
  onUpdateWeight?: (layerName: string, srcIdx: number, targetIdx: number, newWeight: number) => Promise<void>;
  onInspectHeatmap?: (layerName: string) => void;
}

interface CustomRange {
  start: number;
  end: number;
}

interface LayerColumn {
  name: string;
  type: string;
  count: number;
  colIdx: number;
  nodeId: string;
  x: number;
  visibleIndices: number[];
  hiddenCount: number;
}

const DEFAULT_PAGE_SIZE = 32;
const Y_SPACING = 30;
const NEURON_RADIUS = 11;
const COL_WIDTH = 320;

export const NetworkCanvas: React.FC<CanvasProps> = ({
  modelView,
  modelId,
  modelVersion,
  onSelectNode,
  selectedNodeId,
  onUpdateWeight,
  onInspectHeatmap,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [viewMode, setViewMode] = useState<'neuron' | 'layer'>('neuron');
  const [pageSize, setPageSize] = useState<number>(DEFAULT_PAGE_SIZE);

  // Per-layer custom ranges [start, end]
  const [customRanges, setCustomRanges] = useState<Record<number, CustomRange>>({});

  // Active popover for "+N nodes" card
  const [activePopoverCol, setActivePopoverCol] = useState<number | null>(null);
  const [customRangeInput, setCustomRangeInput] = useState<string>('');

  // Cache weight slices
  const [weightCache, setWeightCache] = useState<Record<string, number[][]>>({});

  // Camera Pan & Zoom
  const [camera, setCamera] = useState({ x: 80, y: 80, scale: 0.9 });
  const isDragging = useRef(false);
  const lastMousePos = useRef({ x: 0, y: 0 });

  // Hover state
  const [hoveredSynapse, setHoveredSynapse] = useState<{
    srcCol: number;
    srcIdx: number;
    targetCol: number;
    targetIdx: number;
    weight: number;
    x: number;
    y: number;
    layerName: string;
  } | null>(null);

  const [hoveredNode, setHoveredNode] = useState<{
    colIdx: number;
    idx: number;
    name: string;
    nodeId: string;
    x: number;
    y: number;
  } | null>(null);

  // Edit Synapse Modal
  const [editingSynapse, setEditingSynapse] = useState<{
    layerName: string;
    srcIdx: number;
    targetIdx: number;
    weight: number;
  } | null>(null);
  const [tempWeight, setTempWeight] = useState<string>('');
  const [isUpdating, setIsUpdating] = useState<boolean>(false);

  useEffect(() => {
    setWeightCache({});
  }, [modelVersion, modelView?.name]);

  const columns = useMemo<LayerColumn[]>(() => {
    if (!modelView) return [];

    let inputCount = 784;
    const inputShape = modelView.meta.input_shapes?.[0];
    if (Array.isArray(inputShape) && inputShape.length > 1) {
      const dims = inputShape.filter((d) => d !== null);
      if (dims.length > 0) inputCount = dims.reduce((a, b) => a * b, 1);
    }

    const rawLayers = modelView.nodes;
    const meaningfulLayers: typeof rawLayers = [];

    rawLayers.forEach((l) => {
      if (l.type === 'Flatten') {
        if (l.attrs?.units) inputCount = Number(l.attrs.units);
        return;
      }
      meaningfulLayers.push(l);
    });

    const getVisibleIndices = (colIdx: number, total: number): { list: number[]; hidden: number } => {
      const custom = customRanges[colIdx];
      if (custom) {
        const s = Math.max(0, Math.min(custom.start, total - 1));
        const e = Math.max(s, Math.min(custom.end, total - 1));
        const list: number[] = [];
        for (let i = s; i <= e; i++) list.push(i);
        return { list, hidden: Math.max(0, total - list.length) };
      }
      const count = Math.min(total, pageSize);
      const list = Array.from({ length: count }, (_, i) => i);
      return { list, hidden: Math.max(0, total - count) };
    };

    const inData = getVisibleIndices(0, inputCount);
    const cols: LayerColumn[] = [
      {
        name: 'Input Layer',
        type: 'Input',
        count: inputCount,
        colIdx: 0,
        nodeId: 'input_layer',
        x: 0,
        visibleIndices: inData.list,
        hiddenCount: inData.hidden,
      },
    ];

    let currentX = COL_WIDTH;

    meaningfulLayers.forEach((layer, idx) => {
      let count = 1;
      if (layer.attrs?.units) {
        count = Number(layer.attrs.units);
      } else if (Array.isArray(layer.attrs?.shape) && layer.attrs.shape.length >= 2) {
        const lastDim = layer.attrs.shape[layer.attrs.shape.length - 1];
        if (lastDim !== null && !isNaN(Number(lastDim))) count = Number(lastDim);
      }

      const layerData = getVisibleIndices(idx + 1, count);
      cols.push({
        name: layer.label,
        type: layer.type,
        count,
        colIdx: idx + 1,
        nodeId: layer.id,
        x: currentX,
        visibleIndices: layerData.list,
        hiddenCount: layerData.hidden,
      });

      currentX += COL_WIDTH;
    });

    return cols;
  }, [modelView, customRanges, pageSize]);

  // Fetch weight slices on demand
  useEffect(() => {
    if (!modelId || columns.length < 2) return;

    for (let c = 0; c < columns.length - 1; c++) {
      const srcCol = columns[c];
      const targetCol = columns[c + 1];

      const sStart = srcCol.visibleIndices[0] ?? 0;
      const sEnd = srcCol.visibleIndices[srcCol.visibleIndices.length - 1] ?? 0;
      const tStart = targetCol.visibleIndices[0] ?? 0;
      const tEnd = targetCol.visibleIndices[targetCol.visibleIndices.length - 1] ?? 0;

      const cacheKey = `${targetCol.name}:${sStart}-${sEnd}:${tStart}-${tEnd}`;

      if (!weightCache[cacheKey]) {
        fetch(
          `http://127.0.0.1:8000/api/models/${modelId}/tensors/${encodeURIComponent(
            targetCol.name
          )}?src_start=${sStart}&src_end=${sEnd}&target_start=${tStart}&target_end=${tEnd}&v=${modelVersion}`
        )
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data && data.values) {
              setWeightCache((prev) => ({
                ...prev,
                [cacheKey]: data.values,
              }));
            }
          })
          .catch(() => {});
      }
    }
  }, [modelId, columns, weightCache, modelVersion]);

  const getNodeY = useCallback((localIdx: number) => {
    return 130 + localIdx * Y_SPACING;
  }, []);

  const getWeightFast = useCallback(
    (targetLayerName: string, srcIdx: number, targetIdx: number, srcCol: LayerColumn, targetCol: LayerColumn): number => {
      const sStart = srcCol.visibleIndices[0] ?? 0;
      const sEnd = srcCol.visibleIndices[srcCol.visibleIndices.length - 1] ?? 0;
      const tStart = targetCol.visibleIndices[0] ?? 0;
      const tEnd = targetCol.visibleIndices[targetCol.visibleIndices.length - 1] ?? 0;

      const cacheKey = `${targetLayerName}:${sStart}-${sEnd}:${tStart}-${tEnd}`;
      const matrix = weightCache[cacheKey];

      if (matrix) {
        const r = srcIdx - sStart;
        const c = targetIdx - tStart;
        if (matrix[r] && matrix[r][c] !== undefined) {
          return matrix[r][c];
        }
      }
      return 0.02;
    },
    [weightCache]
  );

  // Main Canvas Render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, width, height);

    ctx.translate(camera.x, camera.y);
    ctx.scale(camera.scale, camera.scale);

    // Grid dots
    ctx.fillStyle = '#1c1c20';
    const gridSize = 32;
    const startX = Math.floor((-camera.x / camera.scale) / gridSize) * gridSize;
    const startY = Math.floor((-camera.y / camera.scale) / gridSize) * gridSize;
    const endX = startX + width / camera.scale + gridSize;
    const endY = startY + height / camera.scale + gridSize;

    for (let x = startX; x < endX; x += gridSize) {
      for (let y = startY; y < endY; y += gridSize) {
        ctx.fillRect(x, y, 1.2, 1.2);
      }
    }

    if (viewMode === 'layer') {
      columns.forEach((col, idx) => {
        const x = idx * 260;
        const y = 100;
        const w = 210;
        const h = 120;
        const isSelected = selectedNodeId === col.nodeId;

        ctx.fillStyle = isSelected ? '#27272a' : '#18181b';
        ctx.strokeStyle = isSelected ? '#f59e0b' : '#3f3f46';
        ctx.lineWidth = isSelected ? 2 : 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 8);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(col.type.toUpperCase(), x + 16, y + 26);

        ctx.fillStyle = '#fafafa';
        ctx.font = 'bold 13px sans-serif';
        ctx.fillText(col.name, x + 16, y + 50);

        ctx.fillStyle = '#a1a1aa';
        ctx.font = '11px monospace';
        ctx.fillText(`${col.count.toLocaleString()} Neurons`, x + 16, y + 80);

        if (idx < columns.length - 1) {
          ctx.strokeStyle = '#f59e0b';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x + w, y + h / 2);
          ctx.lineTo(x + 260, y + h / 2);
          ctx.stroke();
        }
      });
      ctx.restore();
      return;
    }

    // 1. Draw Synapses
    for (let c = 0; c < columns.length - 1; c++) {
      const srcCol = columns[c];
      const targetCol = columns[c + 1];

      for (let i = 0; i < srcCol.visibleIndices.length; i++) {
        const actualSrc = srcCol.visibleIndices[i];
        const sy = getNodeY(i);

        for (let j = 0; j < targetCol.visibleIndices.length; j++) {
          const actualTarget = targetCol.visibleIndices[j];
          const ty = getNodeY(j);

          const w = getWeightFast(targetCol.name, actualSrc, actualTarget, srcCol, targetCol);
          const isHovered =
            hoveredSynapse &&
            hoveredSynapse.srcCol === c &&
            hoveredSynapse.srcIdx === actualSrc &&
            hoveredSynapse.targetCol === c + 1 &&
            hoveredSynapse.targetIdx === actualTarget;

          const absW = Math.abs(w);
          const strokeWidth = isHovered ? 4.5 : Math.min(Math.max(absW * 3.2, 0.8), 2.5);

          ctx.lineWidth = strokeWidth;
          if (isHovered) {
            ctx.strokeStyle = '#ffffff';
          } else {
            ctx.strokeStyle = w >= 0 ? 'rgba(56, 189, 248, 0.45)' : 'rgba(245, 158, 11, 0.45)';
          }

          ctx.beginPath();
          ctx.moveTo(srcCol.x + NEURON_RADIUS, sy);
          ctx.lineTo(targetCol.x - NEURON_RADIUS, ty);
          ctx.stroke();
        }
      }
    }

    // 2. Draw Column Neurons & Headers
    columns.forEach((col, cIdx) => {
      const isColSelected = selectedNodeId === col.nodeId;

      ctx.fillStyle = isColSelected ? '#38bdf8' : '#f59e0b';
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(col.name, col.x, 32);

      const s = col.visibleIndices[0] ?? 0;
      const e = col.visibleIndices[col.visibleIndices.length - 1] ?? 0;

      ctx.fillStyle = '#a1a1aa';
      ctx.font = '11px monospace';
      ctx.fillText(`Showing ${s}–${e} of ${col.count}`, col.x, 50);

      // Neurons
      for (let i = 0; i < col.visibleIndices.length; i++) {
        const actualIdx = col.visibleIndices[i];
        const ny = getNodeY(i);

        const isHighlighted =
          (hoveredSynapse &&
            ((hoveredSynapse.srcCol === cIdx && hoveredSynapse.srcIdx === actualIdx) ||
              (hoveredSynapse.targetCol === cIdx && hoveredSynapse.targetIdx === actualIdx))) ||
          (hoveredNode && hoveredNode.colIdx === cIdx && hoveredNode.idx === actualIdx);

        ctx.beginPath();
        ctx.arc(col.x, ny, NEURON_RADIUS, 0, Math.PI * 2);

        if (isHighlighted) {
          ctx.fillStyle = '#38bdf8';
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2.5;
          ctx.shadowColor = '#38bdf8';
          ctx.shadowBlur = 10;
        } else {
          ctx.shadowBlur = 0;
          ctx.fillStyle = '#18181b';
          ctx.strokeStyle = '#52525b';
          ctx.lineWidth = 1.8;
        }

        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;

        ctx.fillStyle = isHighlighted ? '#09090b' : '#e4e4e7';
        ctx.font = 'bold 9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(actualIdx), col.x, ny);
      }
    });

    ctx.restore();
  }, [columns, camera, viewMode, hoveredSynapse, hoveredNode, selectedNodeId, getNodeY, getWeightFast]);

  // Pan handlers
  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDragging.current = true;
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging.current) {
      const dx = e.clientX - lastMousePos.current.x;
      const dy = e.clientY - lastMousePos.current.y;
      setCamera((prev) => ({
        ...prev,
        x: prev.x + dx,
        y: prev.y + dy,
      }));
      lastMousePos.current = { x: e.clientX, y: e.clientY };
      return;
    }

    if (viewMode === 'layer') return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mx = (e.clientX - rect.left - camera.x) / camera.scale;
    const my = (e.clientY - rect.top - camera.y) / camera.scale;

    // Check Node Hover
    let foundNode = null;
    for (const col of columns) {
      if (Math.abs(col.x - mx) <= NEURON_RADIUS + 4) {
        for (let i = 0; i < col.visibleIndices.length; i++) {
          const ny = getNodeY(i);
          if (Math.abs(ny - my) <= NEURON_RADIUS + 4) {
            foundNode = {
              colIdx: col.colIdx,
              idx: col.visibleIndices[i],
              name: col.name,
              nodeId: col.nodeId,
              x: e.clientX,
              y: e.clientY,
            };
            break;
          }
        }
      }
      if (foundNode) break;
    }
    setHoveredNode(foundNode);

    // Check Synapse Hover
    let foundSynapse = null;
    for (let c = 0; c < columns.length - 1; c++) {
      const srcCol = columns[c];
      const targetCol = columns[c + 1];

      const x1 = srcCol.x + NEURON_RADIUS;
      const x2 = targetCol.x - NEURON_RADIUS;

      if (mx < x1 - 5 || mx > x2 + 5) continue;

      for (let i = 0; i < srcCol.visibleIndices.length; i++) {
        const y1 = getNodeY(i);
        const actualSrc = srcCol.visibleIndices[i];

        for (let j = 0; j < targetCol.visibleIndices.length; j++) {
          const y2 = getNodeY(j);
          const actualTarget = targetCol.visibleIndices[j];

          const dx = x2 - x1;
          const dy = y2 - y1;
          const lenSq = dx * dx + dy * dy;
          const t = Math.max(0, Math.min(1, ((mx - x1) * dx + (my - y1) * dy) / lenSq));
          const px = x1 + t * dx;
          const py = y1 + t * dy;
          const dist = Math.hypot(mx - px, my - py);

          if (dist < 6) {
            const w = getWeightFast(targetCol.name, actualSrc, actualTarget, srcCol, targetCol);
            foundSynapse = {
              srcCol: c,
              srcIdx: actualSrc,
              targetCol: c + 1,
              targetIdx: actualTarget,
              weight: w,
              x: e.clientX,
              y: e.clientY,
              layerName: targetCol.name,
            };
            break;
          }
        }
        if (foundSynapse) break;
      }
      if (foundSynapse) break;
    }

    setHoveredSynapse(foundSynapse);
  };

  const onMouseUp = () => {
    isDragging.current = false;
  };

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    const newScale = Math.max(0.15, Math.min(3.0, camera.scale * zoomFactor));

    setCamera((prev) => ({
      scale: newScale,
      x: mouseX - (mouseX - prev.x) * (newScale / prev.scale),
      y: mouseY - (mouseY - prev.y) * (newScale / prev.scale),
    }));
  };

  const handlePageChange = (colIdx: number, direction: 'next' | 'prev') => {
    const col = columns[colIdx];
    if (!col) return;
    const currentStart = col.visibleIndices[0] ?? 0;
    const step = col.visibleIndices.length;
    let nextStart = currentStart;

    if (direction === 'next') {
      if (currentStart + step < col.count) nextStart = currentStart + step;
    } else {
      if (currentStart - step >= 0) nextStart = currentStart - step;
      else nextStart = 0;
    }

    const nextEnd = Math.min(col.count - 1, nextStart + step - 1);
    setCustomRanges((prev) => ({
      ...prev,
      [colIdx]: { start: nextStart, end: nextEnd },
    }));
  };

  const handleSetSpan = (colIdx: number, spanSize: number) => {
    const col = columns[colIdx];
    if (!col) return;
    const currentStart = col.visibleIndices[0] ?? 0;
    const nextEnd = Math.min(col.count - 1, currentStart + spanSize - 1);
    setCustomRanges((prev) => ({
      ...prev,
      [colIdx]: { start: currentStart, end: nextEnd },
    }));
    setActivePopoverCol(null);
  };

  const handleApplyCustomInput = (colIdx: number) => {
    const col = columns[colIdx];
    if (!col) return;

    const parts = customRangeInput.split('-').map((s) => parseInt(s.trim(), 10));
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      const s = Math.max(0, Math.min(parts[0], col.count - 1));
      const e = Math.max(s, Math.min(parts[1], col.count - 1));
      setCustomRanges((prev) => ({
        ...prev,
        [colIdx]: { start: s, end: e },
      }));
    } else if (parts.length === 1 && !isNaN(parts[0])) {
      const s = Math.max(0, Math.min(parts[0], col.count - 1));
      const e = Math.min(col.count - 1, s + pageSize - 1);
      setCustomRanges((prev) => ({
        ...prev,
        [colIdx]: { start: s, end: e },
      }));
    }
    setActivePopoverCol(null);
  };

  const handleSaveWeight = async () => {
    if (!editingSynapse || !onUpdateWeight) return;
    const val = parseFloat(tempWeight);
    if (isNaN(val)) return;

    setIsUpdating(true);
    try {
      await onUpdateWeight(
        editingSynapse.layerName,
        editingSynapse.srcIdx,
        editingSynapse.targetIdx,
        val
      );

      // Instantly update weightCache
      setWeightCache((prev) => {
        const next = { ...prev };
        Object.keys(next).forEach((k) => {
          if (k.startsWith(`${editingSynapse.layerName}:`)) {
            const parts = k.split(':');
            const [sS, sE] = parts[1].split('-').map(Number);
            const [tS, tE] = parts[2].split('-').map(Number);
            if (
              editingSynapse.srcIdx >= sS &&
              editingSynapse.srcIdx <= sE &&
              editingSynapse.targetIdx >= tS &&
              editingSynapse.targetIdx <= tE
            ) {
              const r = editingSynapse.srcIdx - sS;
              const c = editingSynapse.targetIdx - tS;
              if (next[k][r]) {
                const copy = next[k].map((row) => [...row]);
                copy[r][c] = val;
                next[k] = copy;
              }
            }
          }
        });
        return next;
      });

      setHoveredSynapse((prev) => (prev ? { ...prev, weight: val } : null));
      setEditingSynapse(null);
    } catch (err) {
      console.error(err);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div ref={containerRef} className="w-full h-full bg-zinc-950 relative overflow-hidden select-none">
      {/* Top Floating Controls */}
      <div className="absolute top-4 right-4 z-20 flex items-center space-x-2 bg-zinc-900/95 backdrop-blur border border-zinc-700 p-1.5 rounded-lg shadow-xl">
        <button
          onClick={() => setCamera((prev) => ({ ...prev, scale: Math.min(3.0, prev.scale * 1.2) }))}
          className="p-1.5 rounded hover:bg-zinc-800 text-zinc-300 hover:text-white"
          title="Zoom In"
        >
          <ZoomIn size={14} />
        </button>
        <button
          onClick={() => setCamera((prev) => ({ ...prev, scale: Math.max(0.15, prev.scale * 0.8) }))}
          className="p-1.5 rounded hover:bg-zinc-800 text-zinc-300 hover:text-white"
          title="Zoom Out"
        >
          <ZoomOut size={14} />
        </button>
        <button
          onClick={() => setCamera({ x: 80, y: 80, scale: 0.9 })}
          className="p-1.5 rounded hover:bg-zinc-800 text-zinc-300 hover:text-white"
          title="Reset View"
        >
          <RotateCcw size={14} />
        </button>

        <div className="h-4 w-px bg-zinc-700" />

        <button
          onClick={() => setViewMode('neuron')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold rounded transition ${
            viewMode === 'neuron'
              ? 'bg-amber-500 text-zinc-950 shadow'
              : 'text-zinc-400 hover:text-zinc-100'
          }`}
        >
          <Network size={14} />
          <span>Neuron Playground</span>
        </button>
        <button
          onClick={() => setViewMode('layer')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold rounded transition ${
            viewMode === 'layer'
              ? 'bg-amber-500 text-zinc-950 shadow'
              : 'text-zinc-400 hover:text-zinc-100'
          }`}
        >
          <Layers size={14} />
          <span>Layer Blocks</span>
        </button>
      </div>

      {/* Layer Navigation & Centered "+N nodes" Expansion Box */}
      {viewMode === 'neuron' && (
        <div className="absolute inset-0 pointer-events-none z-10">
          {columns.map((col) => {
            const screenX = col.x * camera.scale + camera.x;
            const s = col.visibleIndices[0] ?? 0;
            const e = col.visibleIndices[col.visibleIndices.length - 1] ?? 0;
            const isPopoverOpen = activePopoverCol === col.colIdx;

            const midVisibleIdx = Math.floor(col.visibleIndices.length / 2);
            const centerWorldY = getNodeY(midVisibleIdx);
            const screenCenterY = centerWorldY * camera.scale + camera.y;

            return (
              <React.Fragment key={col.colIdx}>
                {/* Header Title Pill */}
                <div
                  className="absolute pointer-events-auto transform -translate-x-1/2 flex items-center space-x-1 bg-zinc-900/90 border border-zinc-700 rounded-md px-2 py-0.5 shadow-xl"
                  style={{ left: `${screenX}px`, top: '70px' }}
                >
                  <button
                    disabled={s <= 0}
                    onClick={() => handlePageChange(col.colIdx, 'prev')}
                    className="p-0.5 rounded hover:bg-zinc-800 text-amber-400 disabled:opacity-20"
                    title="Previous page"
                  >
                    <ChevronUp size={12} />
                  </button>

                  <div
                    onClick={() => onSelectNode(col.nodeId)}
                    className="cursor-pointer text-[10px] font-mono font-bold text-zinc-300 hover:text-amber-400 px-1"
                    title="Click to view details in Inspector side panel"
                  >
                    {s}–{e}
                  </div>

                  <button
                    disabled={e >= col.count - 1}
                    onClick={() => handlePageChange(col.colIdx, 'next')}
                    className="p-0.5 rounded hover:bg-zinc-800 text-amber-400 disabled:opacity-20"
                    title="Next page"
                  >
                    <ChevronDown size={12} />
                  </button>

                  {col.colIdx > 0 && onInspectHeatmap && (
                    <button
                      onClick={(ev) => {
                        ev.stopPropagation();
                        onInspectHeatmap(col.name);
                      }}
                      className="p-0.5 text-zinc-400 hover:text-amber-400 ml-0.5"
                      title="View Heatmap Matrix"
                    >
                      <Grid size={11} />
                    </button>
                  )}
                </div>

                {/* Centered "+N nodes" Box with 8, 16, 32, 64 Presets */}
                {col.hiddenCount > 0 && (
                  <div
                    className="absolute pointer-events-auto transform -translate-x-1/2 -translate-y-1/2"
                    style={{ left: `${screenX}px`, top: `${screenCenterY}px` }}
                  >
                    <button
                      onClick={() => {
                        setActivePopoverCol(isPopoverOpen ? null : col.colIdx);
                        setCustomRangeInput(`${s}-${e}`);
                      }}
                      className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-zinc-900/95 hover:bg-zinc-800 border-2 border-dashed border-amber-500/70 hover:border-amber-400 text-xs font-mono font-bold text-amber-400 shadow-2xl backdrop-blur transition-all hover:scale-105 active:scale-95"
                      title="Click to choose preset or custom range"
                    >
                      <Plus size={13} className="text-amber-400" />
                      <span>+{col.hiddenCount} nodes</span>
                    </button>

                    {/* Popover Menu: 8 | 16 | 32 | 64 + Custom Input */}
                    {isPopoverOpen && (
                      <div className="absolute top-10 left-1/2 transform -translate-x-1/2 w-56 bg-zinc-900 border border-zinc-700 rounded-xl p-3 shadow-2xl space-y-2.5 z-50 text-left">
                        <div className="flex justify-between items-center text-[10px] uppercase font-bold text-zinc-400 border-b border-zinc-800 pb-1">
                          <span>Choose Node Window</span>
                          <button onClick={() => setActivePopoverCol(null)} className="text-zinc-500 hover:text-zinc-200">
                            <X size={12} />
                          </button>
                        </div>

                        {/* Presets now include 8 */}
                        <div className="flex space-x-1.5">
                          {[8, 16, 32, 64].map((span) => (
                            <button
                              key={span}
                              onClick={() => handleSetSpan(col.colIdx, span)}
                              className="flex-1 py-1.5 rounded-md bg-zinc-800 hover:bg-amber-500 hover:text-zinc-950 text-xs font-mono font-bold transition text-center text-zinc-200"
                            >
                              {span}
                            </button>
                          ))}
                        </div>

                        <div className="text-[10px] uppercase font-bold text-zinc-400 pt-1 border-t border-zinc-800">
                          Or Custom Range:
                        </div>
                        <div className="flex items-center space-x-1.5">
                          <input
                            type="text"
                            placeholder="e.g. 7-39"
                            value={customRangeInput}
                            onChange={(ev) => setCustomRangeInput(ev.target.value)}
                            onKeyDown={(ev) => ev.key === 'Enter' && handleApplyCustomInput(col.colIdx)}
                            className="w-full bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs font-mono text-zinc-100 focus:outline-none focus:border-amber-400"
                          />
                          <button
                            onClick={() => handleApplyCustomInput(col.colIdx)}
                            className="p-1.5 bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold rounded"
                          >
                            <Check size={12} />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      )}

      {/* Floating Synapse Tooltip */}
      {hoveredSynapse && (
        <div
          className="pointer-events-none fixed z-50 transform -translate-x-1/2 -translate-y-12 bg-zinc-900/95 border border-amber-400 px-3.5 py-1.5 rounded-md shadow-2xl backdrop-blur text-xs font-mono flex items-center space-x-2"
          style={{
            left: `${hoveredSynapse.x}px`,
            top: `${hoveredSynapse.y}px`,
          }}
        >
          <span className="text-zinc-400">
            Neuron {hoveredSynapse.srcIdx} → {hoveredSynapse.targetIdx}:
          </span>
          <span
            className={`font-black ${
              hoveredSynapse.weight >= 0 ? 'text-sky-400' : 'text-amber-400'
            }`}
          >
            {hoveredSynapse.weight.toFixed(5)}
          </span>
          <span className="text-[10px] text-zinc-500 font-sans">(Click to edit)</span>
        </div>
      )}

      {/* Floating Node Tooltip */}
      {hoveredNode && !hoveredSynapse && (
        <div
          className="pointer-events-none fixed z-50 transform -translate-x-1/2 -translate-y-10 bg-zinc-900/95 border border-sky-400 px-3 py-1 rounded shadow-xl backdrop-blur text-xs font-mono flex items-center space-x-1.5"
          style={{
            left: `${hoveredNode.x}px`,
            top: `${hoveredNode.y}px`,
          }}
        >
          <span className="text-zinc-400">{hoveredNode.name}:</span>
          <span className="font-bold text-sky-400">Neuron #{hoveredNode.idx}</span>
          <span className="text-[9px] text-zinc-500">(Click to inspect)</span>
        </div>
      )}

      {/* Edit Synapse Weight Modal */}
      {editingSynapse && (
        <div className="absolute inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center">
          <div className="bg-zinc-900 border border-zinc-700 p-5 rounded-xl shadow-2xl w-84 space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-bold flex items-center space-x-2 text-zinc-100">
                <Edit3 size={16} className="text-amber-400" />
                <span>Edit Synapse Weight</span>
              </span>
              <button onClick={() => setEditingSynapse(null)} className="text-zinc-400 hover:text-zinc-100">
                <X size={18} />
              </button>
            </div>

            <div className="text-xs text-zinc-400 font-mono space-y-1 bg-zinc-950 p-3 rounded border border-zinc-800">
              <div>Layer: <span className="text-amber-400">{editingSynapse.layerName}</span></div>
              <div>Connection: Neuron {editingSynapse.srcIdx} → Neuron {editingSynapse.targetIdx}</div>
            </div>

            <div>
              <label className="text-xs text-zinc-300 block mb-1">New Weight Value:</label>
              <input
                type="number"
                step="0.001"
                autoFocus
                value={tempWeight}
                onChange={(e) => setTempWeight(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSaveWeight()}
                className="w-full bg-zinc-950 border border-zinc-700 rounded p-2 text-sm font-mono text-zinc-100 focus:outline-none focus:border-amber-400"
              />
            </div>

            <div className="flex justify-end space-x-2 pt-1">
              <button onClick={() => setEditingSynapse(null)} className="px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-100">
                Cancel
              </button>
              <button
                disabled={isUpdating}
                onClick={handleSaveWeight}
                className="flex items-center space-x-1.5 px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold rounded text-xs transition"
              >
                <Check size={14} />
                <span>{isUpdating ? 'Saving...' : 'Apply'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* GPU Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onWheel={onWheel}
        onClick={() => {
          if (hoveredNode) {
            onSelectNode(hoveredNode.nodeId);
          } else if (hoveredSynapse) {
            setEditingSynapse({
              layerName: hoveredSynapse.layerName,
              srcIdx: hoveredSynapse.srcIdx,
              targetIdx: hoveredSynapse.targetIdx,
              weight: hoveredSynapse.weight,
            });
            setTempWeight(String(hoveredSynapse.weight));
          }
        }}
      />
    </div>
  );
};