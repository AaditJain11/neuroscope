import React, { useState } from 'react';
import {
  Upload,
  Cpu,
  Layers,
  Activity,
  Play,
  Sliders,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Download,
  RotateCcw,
  Grid,
  TrendingUp,
  TrendingDown,
  Gauge,
  Activity as Pulse,
  ChevronRight,
  ChevronLeft,
} from 'lucide-react';
import type { ModelView } from './types';
import { NetworkCanvas } from './components/NetworkCanvas';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [modelId, setModelId] = useState<string | null>(null);
  const [modelView, setModelView] = useState<ModelView | null>(null);
  const [modelVersion, setModelVersion] = useState<number>(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'network' | 'inference' | 'activations' | 'training'>('network');
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Model Metrics State
  const [currentMetrics, setCurrentMetrics] = useState<{
    accuracy: number;
    loss: number;
    baselineAccuracy: number;
    baselineLoss: number;
  } | null>(null);

  // Mutation Impact state
  const [lastMutationImpact, setLastMutationImpact] = useState<{
    deltaAcc: number;
    deltaLoss: number;
    outputShift: number;
    currentAcc: number;
    currentLoss: number;
    layer: string;
    connection: string;
    oldWeight: number;
    newWeight: number;
  } | null>(null);

  // Inference state
  const [inputVector, setInputVector] = useState<string>('0.5, -1.2, 0.3, 2.1');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [activationResult, setActivationResult] = useState<any>(null);

  // Training parameters
  const [trainEpochs, setTrainEpochs] = useState<number>(2);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [isRetraining, setIsRetraining] = useState<boolean>(false);
  const [trainingMetrics, setTrainingMetrics] = useState<any>(null);

  // Heatmap state
  const [heatmapData, setHeatmapData] = useState<{
    layerName: string;
    shape: number[];
    grid: number[][];
    min: number;
    max: number;
  } | null>(null);

  const refreshModelView = async (id: string) => {
    const inspectRes = await fetch(`${API_BASE}/api/models/${id}?t=${Date.now()}`);
    if (!inspectRes.ok) throw new Error('Failed to fetch model architecture');
    const view: ModelView = await inspectRes.json();
    setModelView(view);

    const bAcc = (view.meta as any).baseline_accuracy ?? 0;
    const bLoss = (view.meta as any).baseline_loss ?? 0;

    setCurrentMetrics((prev) => ({
      accuracy: prev?.accuracy ?? bAcc,
      loss: prev?.loss ?? bLoss,
      baselineAccuracy: bAcc,
      baselineLoss: bLoss,
    }));
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setErrorMsg(null);
    setPredictionResult(null);
    setActivationResult(null);
    setTrainingMetrics(null);
    setHeatmapData(null);
    setLastMutationImpact(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadRes = await fetch(`${API_BASE}/api/models/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed with HTTP ${uploadRes.status}`);
      }

      const uploadData = await uploadRes.json();
      const newModelId = uploadData.model_id;
      setModelId(newModelId);

      await refreshModelView(newModelId);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Error uploading file');
    } finally {
      setLoading(false);
      e.target.value = '';
    }
  };

  const handleNodeSelect = (nodeId: string) => {
    setSelectedNodeId(nodeId);
    setIsInspectorOpen(true);
  };

  const handleInspectHeatmap = async (layerName: string) => {
    if (!modelId) return;
    try {
      const res = await fetch(`${API_BASE}/api/models/${modelId}/heatmap/${encodeURIComponent(layerName)}`);
      if (res.ok) {
        const data = await res.json();
        setHeatmapData({
          layerName,
          shape: data.shape,
          grid: data.grid,
          min: data.min,
          max: data.max,
        });
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleUpdateWeight = async (
    layerName: string,
    srcIdx: number,
    targetIdx: number,
    newWeight: number
  ) => {
    if (!modelId) return;
    try {
      const res = await fetch(`${API_BASE}/api/models/${modelId}/weights`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layer_name: layerName,
          src_idx: srcIdx,
          target_idx: targetIdx,
          weight_value: newWeight,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed updating weight');
      }

      const resData = await res.json();
      if (resData.impact) {
        setLastMutationImpact({
          deltaAcc: resData.impact.delta_accuracy,
          deltaLoss: resData.impact.delta_loss,
          outputShift: resData.impact.output_shift,
          currentAcc: resData.impact.current_accuracy,
          currentLoss: resData.impact.current_loss,
          layer: layerName,
          connection: `${srcIdx} → ${targetIdx}`,
          oldWeight: resData.impact.old_weight,
          newWeight: resData.impact.new_weight,
        });
        setCurrentMetrics((prev) => ({
          ...prev!,
          accuracy: resData.impact.current_accuracy,
          loss: resData.impact.current_loss,
        }));
      }

      setModelVersion((v) => v + 1);
      await refreshModelView(modelId);
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };

  const handleRetrain = async () => {
    if (!modelId) return;
    setIsRetraining(true);
    try {
      const res = await fetch(`${API_BASE}/api/models/${modelId}/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          epochs: trainEpochs,
          learning_rate: learningRate,
        }),
      });
      if (!res.ok) throw new Error('Retraining failed');
      const data = await res.json();
      setTrainingMetrics(data.metrics);
      setCurrentMetrics((prev) => ({
        ...prev!,
        accuracy: data.metrics.after_accuracy,
        loss: data.metrics.after_loss,
      }));
      setModelVersion((v) => v + 1);
      await refreshModelView(modelId);
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      setIsRetraining(false);
    }
  };

  const handleExport = () => {
    if (!modelId) return;
    window.open(`${API_BASE}/api/models/${modelId}/export`, '_blank');
  };

  const runInference = async () => {
    if (!modelId) return;
    try {
      const parsedNumbers = inputVector.split(',').map((s) => parseFloat(s.trim()));
      const res = await fetch(`${API_BASE}/api/models/${modelId}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [parsedNumbers] }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Inference error');
      setPredictionResult(data.predictions);
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };

  const runActivations = async () => {
    if (!modelId) return;
    try {
      const parsedNumbers = inputVector.split(',').map((s) => parseFloat(s.trim()));
      const res = await fetch(`${API_BASE}/api/models/${modelId}/activations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [parsedNumbers] }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Activation error');
      setActivationResult(data.activations);
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };

  const selectedNode = modelView?.nodes.find((n) => n.id === selectedNodeId);

  return (
    <div className="flex flex-col h-screen w-screen bg-zinc-950 text-zinc-100 select-none">
      <header className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-zinc-900 z-10">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/40 flex items-center justify-center text-amber-500">
            <Cpu size={18} />
          </div>
          <span className="font-bold text-lg tracking-tight">NeuroScope</span>
          <span className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded">v0.1.0</span>
        </div>

        {/* Live Top Metrics Display */}
        {currentMetrics && (
          <div className="flex items-center space-x-4 bg-zinc-950 border border-zinc-800 px-3.5 py-1.5 rounded-lg font-mono text-xs shadow-inner">
            <div className="flex items-center space-x-1.5">
              <Gauge size={13} className="text-zinc-400" />
              <span className="text-zinc-400">Val Acc:</span>
              <span className="font-bold text-emerald-400">
                {(currentMetrics.accuracy * 100).toFixed(2)}%
              </span>
              {currentMetrics.accuracy !== currentMetrics.baselineAccuracy && (
                <span
                  className={`text-[10px] flex items-center ${
                    currentMetrics.accuracy >= currentMetrics.baselineAccuracy
                      ? 'text-emerald-400'
                      : 'text-rose-400'
                  }`}
                >
                  {currentMetrics.accuracy >= currentMetrics.baselineAccuracy ? (
                    <TrendingUp size={10} className="mr-0.5" />
                  ) : (
                    <TrendingDown size={10} className="mr-0.5" />
                  )}
                  {((currentMetrics.accuracy - currentMetrics.baselineAccuracy) * 100).toFixed(2)}%
                </span>
              )}
            </div>

            <div className="h-3 w-px bg-zinc-800" />

            <div className="flex items-center space-x-1.5">
              <span className="text-zinc-400">Loss:</span>
              <span className="font-bold text-amber-400">
                {currentMetrics.loss.toFixed(4)}
              </span>
            </div>
          </div>
        )}

        <div className="flex items-center space-x-3">
          {modelId && (
            <button
              onClick={handleExport}
              className="flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-semibold bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 transition"
              title="Download updated .keras model file"
            >
              <Download size={14} className="text-sky-400" />
              <span>Export .keras</span>
            </button>
          )}

          <label className="flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-semibold bg-amber-500 hover:bg-amber-400 text-zinc-950 cursor-pointer transition shadow">
            {loading ? (
              <Loader2 size={14} className="animate-spin text-zinc-950" />
            ) : (
              <Upload size={14} className="text-zinc-950" />
            )}
            <span>{loading ? 'Parsing...' : 'Upload Model'}</span>
            <input
              type="file"
              className="hidden"
              accept=".keras,.h5,.json"
              onChange={handleFileUpload}
              disabled={loading}
            />
          </label>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden relative">
        <aside className="w-80 border-r border-zinc-800 bg-zinc-900 flex flex-col p-4 overflow-y-auto space-y-4">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">Model Info</h2>
            {modelView ? (
              <div className="bg-zinc-800/80 p-3 rounded-lg border border-zinc-700 space-y-2 text-sm font-mono">
                <div className="truncate text-zinc-200 font-bold">{modelView.name}</div>
                <div className="flex justify-between text-xs text-zinc-400">
                  <span>Architecture</span>
                  <span className="text-amber-400 uppercase font-bold">{modelView.kind}</span>
                </div>
                <div className="flex justify-between text-xs text-zinc-400">
                  <span>Total Params</span>
                  <span>{modelView.meta.total_params?.toLocaleString() ?? 'N/A'}</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-zinc-500">Upload a model to inspect architecture.</p>
            )}
          </div>

          {/* Live Weight Mutation Impact Badge */}
          {lastMutationImpact && (
            <div className="bg-zinc-950 border border-amber-500/40 p-3 rounded-lg space-y-2 text-xs font-mono shadow-xl">
              <div className="flex justify-between items-center text-amber-400 font-bold">
                <span>Last Edit Impact</span>
                <span className="text-[10px] text-zinc-500">{lastMutationImpact.connection}</span>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>Weight change:</span>
                <span className="text-zinc-200">{lastMutationImpact.oldWeight} → {lastMutationImpact.newWeight}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400">Prediction Shift:</span>
                <span className="text-sky-400 font-bold flex items-center">
                  <Pulse size={11} className="mr-1" />
                  {lastMutationImpact.outputShift.toFixed(5)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Accuracy Δ:</span>
                <span
                  className={`font-black flex items-center ${
                    lastMutationImpact.deltaAcc >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {lastMutationImpact.deltaAcc >= 0 ? '+' : ''}
                  {(lastMutationImpact.deltaAcc * 100).toFixed(3)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Loss Δ:</span>
                <span
                  className={`font-black ${
                    lastMutationImpact.deltaLoss <= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {lastMutationImpact.deltaLoss >= 0 ? '+' : ''}
                  {lastMutationImpact.deltaLoss.toFixed(4)}
                </span>
              </div>
            </div>
          )}

          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">Capabilities</h2>
            {modelView ? (
              <div className="space-y-1.5">
                {Object.entries(modelView.capabilities).map(([name, cap]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between text-xs p-2 rounded bg-zinc-800/70 border border-zinc-700/80"
                  >
                    <span className="capitalize">{name.replace('_', ' ')}</span>
                    <span className="flex items-center space-x-1">
                      {cap.status === 'supported' && (
                        <CheckCircle2 size={13} className="text-emerald-400" />
                      )}
                      {cap.status === 'partial' && (
                        <AlertCircle size={13} className="text-amber-400" />
                      )}
                      {cap.status === 'unsupported' && (
                        <XCircle size={13} className="text-zinc-500" />
                      )}
                      <span className="text-[10px] uppercase text-zinc-300 font-semibold">{cap.status}</span>
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-zinc-500">Awaiting upload...</p>
            )}
          </div>

          {/* 2D Heatmap Minimap */}
          {heatmapData && (
            <div className="border border-zinc-700 bg-zinc-950 p-3 rounded-lg space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono text-amber-400 font-bold flex items-center space-x-1">
                  <Grid size={12} />
                  <span>{heatmapData.layerName}</span>
                </span>
                <span className="text-[10px] font-mono text-zinc-400">
                  {heatmapData.shape[0]} × {heatmapData.shape[1]}
                </span>
              </div>

              <div className="grid grid-cols-32 gap-px bg-zinc-900 border border-zinc-800 p-1 rounded">
                {heatmapData.grid.map((row, rIdx) => (
                  <div key={rIdx} className="flex">
                    {row.map((val, cIdx) => {
                      const range = heatmapData.max - heatmapData.min || 1;
                      const norm = (val - heatmapData.min) / range;
                      const r = Math.round(56 + norm * (245 - 56));
                      const g = Math.round(189 - norm * 90);
                      const b = Math.round(248 - norm * 230);
                      return (
                        <div
                          key={cIdx}
                          style={{ backgroundColor: `rgb(${r},${g},${b})` }}
                          className="w-1.5 h-1.5 hover:scale-125 transition-transform"
                          title={`[${rIdx * (heatmapData.shape[0] / 32)}, ${cIdx * (heatmapData.shape[1] / 32)}]: ${val.toFixed(4)}`}
                        />
                      );
                    })}
                  </div>
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-zinc-500 font-mono">
                <span>Min: {heatmapData.min.toFixed(3)}</span>
                <span>Max: {heatmapData.max.toFixed(3)}</span>
              </div>
            </div>
          )}
        </aside>

        <main className="flex-1 flex flex-col bg-zinc-950 overflow-hidden relative">
          <div className="flex border-b border-zinc-800 bg-zinc-900 px-4 space-x-1">
            <button
              onClick={() => setActiveTab('network')}
              className={`flex items-center space-x-2 px-4 py-2.5 text-xs font-medium border-b-2 transition ${
                activeTab === 'network'
                  ? 'border-amber-500 text-amber-500 bg-zinc-800/50'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Layers size={14} />
              <span>Network</span>
            </button>
            <button
              onClick={() => setActiveTab('inference')}
              className={`flex items-center space-x-2 px-4 py-2.5 text-xs font-medium border-b-2 transition ${
                activeTab === 'inference'
                  ? 'border-amber-500 text-amber-500 bg-zinc-800/50'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Play size={14} />
              <span>Forward Pass</span>
            </button>
            <button
              onClick={() => setActiveTab('activations')}
              className={`flex items-center space-x-2 px-4 py-2.5 text-xs font-medium border-b-2 transition ${
                activeTab === 'activations'
                  ? 'border-amber-500 text-amber-500 bg-zinc-800/50'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Activity size={14} />
              <span>Activations</span>
            </button>
            <button
              onClick={() => setActiveTab('training')}
              className={`flex items-center space-x-2 px-4 py-2.5 text-xs font-medium border-b-2 transition ${
                activeTab === 'training'
                  ? 'border-amber-500 text-amber-500 bg-zinc-800/50'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <RotateCcw size={14} />
              <span>Training & Metrics</span>
            </button>
          </div>

          <div className="flex-1 relative overflow-hidden">
            {errorMsg && (
              <div className="absolute top-4 left-4 right-4 z-50 bg-red-950 border border-red-500 text-red-200 px-4 py-3 rounded text-xs flex justify-between items-center shadow-2xl">
                <span><strong>Error:</strong> {errorMsg}</span>
                <button onClick={() => setErrorMsg(null)} className="font-bold ml-4 text-base leading-none">×</button>
              </div>
            )}

            {activeTab === 'network' && (
              <NetworkCanvas
                modelView={modelView}
                modelId={modelId}
                modelVersion={modelVersion}
                onSelectNode={handleNodeSelect}
                selectedNodeId={selectedNodeId}
                onUpdateWeight={handleUpdateWeight}
                onInspectHeatmap={handleInspectHeatmap}
              />
            )}

            {activeTab === 'training' && (
              <div className="p-6 max-w-2xl space-y-5 overflow-y-auto h-full">
                <div>
                  <h3 className="text-sm font-semibold mb-1">Fine-Tuning & Model Training</h3>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    Train and adapt weights with real-time accuracy and loss delta tracking against the benchmark validation split.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4 bg-zinc-900 border border-zinc-800 p-4 rounded-xl">
                  <div>
                    <label className="text-xs text-zinc-400 block mb-1">Epochs:</label>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={trainEpochs}
                      onChange={(e) => setTrainEpochs(parseInt(e.target.value) || 1)}
                      className="w-full bg-zinc-950 border border-zinc-700 rounded p-2 text-xs font-mono text-zinc-100"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-zinc-400 block mb-1">Learning Rate:</label>
                    <input
                      type="number"
                      step={0.0001}
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                      className="w-full bg-zinc-950 border border-zinc-700 rounded p-2 text-xs font-mono text-zinc-100"
                    />
                  </div>
                </div>

                <button
                  disabled={isRetraining || !modelId}
                  onClick={handleRetrain}
                  className="flex items-center space-x-2 px-4 py-2.5 bg-amber-500 text-zinc-950 font-bold rounded-lg text-xs hover:bg-amber-400 transition disabled:opacity-50 shadow-lg"
                >
                  {isRetraining ? (
                    <Loader2 size={15} className="animate-spin text-zinc-950" />
                  ) : (
                    <RotateCcw size={15} />
                  )}
                  <span>{isRetraining ? 'Training in progress...' : `Train Model (${trainEpochs} Epochs)`}</span>
                </button>

                {trainingMetrics && (
                  <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 font-mono text-xs space-y-3">
                    <span className="text-zinc-400 uppercase font-bold block border-b border-zinc-800 pb-2">
                      Training Run Results
                    </span>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <span className="text-zinc-500 block text-[10px]">BEFORE ACCURACY</span>
                        <span className="text-zinc-300 font-bold">{(trainingMetrics.before_accuracy * 100).toFixed(2)}%</span>
                      </div>
                      <div>
                        <span className="text-zinc-500 block text-[10px]">AFTER ACCURACY</span>
                        <span className="text-emerald-400 font-bold">{(trainingMetrics.after_accuracy * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <span className="text-zinc-500 block text-[10px]">BEFORE LOSS</span>
                        <span className="text-zinc-300 font-bold">{trainingMetrics.before_loss.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="text-zinc-500 block text-[10px]">AFTER LOSS</span>
                        <span className="text-amber-400 font-bold">{trainingMetrics.after_loss.toFixed(4)}</span>
                      </div>
                    </div>
                    <div className="pt-2 border-t border-zinc-800 flex justify-between">
                      <span className="text-zinc-400">Total Accuracy Gain:</span>
                      <span
                        className={`font-black ${
                          trainingMetrics.delta_accuracy >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {trainingMetrics.delta_accuracy >= 0 ? '+' : ''}
                        {(trainingMetrics.delta_accuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'inference' && (
              <div className="p-6 max-w-2xl space-y-4">
                <h3 className="text-sm font-semibold">Run Forward Pass Inference</h3>
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Input Tensor (Comma-separated floats):</label>
                  <input
                    type="text"
                    value={inputVector}
                    onChange={(e) => setInputVector(e.target.value)}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded p-2 text-sm font-mono text-zinc-100"
                  />
                </div>
                <button
                  onClick={runInference}
                  className="px-4 py-2 bg-amber-500 text-zinc-950 font-semibold rounded text-xs hover:bg-amber-400 transition"
                >
                  Predict
                </button>
                {predictionResult && (
                  <div className="mt-4 p-4 bg-zinc-900 border border-zinc-800 rounded">
                    <span className="text-xs text-zinc-400 uppercase font-mono block mb-1">Predictions</span>
                    <pre className="text-sm font-mono text-emerald-400">
                      {JSON.stringify(predictionResult, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'activations' && (
              <div className="p-6 max-w-3xl space-y-4 overflow-y-auto h-full">
                <h3 className="text-sm font-semibold">Intermediate Layer Activations</h3>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={inputVector}
                    onChange={(e) => setInputVector(e.target.value)}
                    className="flex-1 bg-zinc-800 border border-zinc-700 rounded p-2 text-sm font-mono text-zinc-100"
                  />
                  <button
                    onClick={runActivations}
                    className="px-4 py-2 bg-amber-500 text-zinc-950 font-semibold rounded text-xs hover:bg-amber-400 transition"
                  >
                    Trace
                  </button>
                </div>
                {activationResult && (
                  <div className="space-y-3 mt-4">
                    {Object.entries(activationResult).map(([layerName, values]: any) => (
                      <div key={layerName} className="p-3 bg-zinc-900 border border-zinc-800 rounded">
                        <span className="text-xs text-amber-400 font-mono font-bold">{layerName}</span>
                        <pre className="text-xs font-mono text-zinc-300 mt-1 overflow-x-auto">
                          {JSON.stringify(values, null, 2)}
                        </pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </main>

        {/* Clickable Inspector Toggle Tab Button */}
        <button
          onClick={() => setIsInspectorOpen(!isInspectorOpen)}
          className={`absolute top-1/2 -translate-y-1/2 z-30 flex items-center justify-center p-1.5 rounded-l-md bg-zinc-900 border-l border-y border-zinc-700 text-zinc-300 hover:text-amber-400 hover:bg-zinc-800 shadow-2xl transition-all ${
            isInspectorOpen ? 'right-80' : 'right-0'
          }`}
          title={isInspectorOpen ? 'Collapse Inspector' : 'Open Inspector'}
        >
          {isInspectorOpen ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>

        {/* Collapsible Inspector Side Panel */}
        {isInspectorOpen && (
          <aside className="w-80 border-l border-zinc-800 bg-zinc-900 flex flex-col p-4 overflow-y-auto z-20">
            <div className="flex items-center justify-between mb-3 border-b border-zinc-800 pb-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center space-x-1.5">
                <Sliders size={14} />
                <span>Inspector</span>
              </h2>
              <button
                onClick={() => setIsInspectorOpen(false)}
                className="text-zinc-500 hover:text-zinc-200"
                title="Close sidebar"
              >
                <X size={14} />
              </button>
            </div>

            {selectedNode ? (
              <div className="space-y-4 text-xs">
                <div className="bg-zinc-800/80 p-3 rounded border border-zinc-700">
                  <span className="text-zinc-400 uppercase text-[10px]">Node ID</span>
                  <div className="font-mono text-zinc-200 mt-0.5">{selectedNode.id}</div>
                </div>
                <div className="bg-zinc-800/80 p-3 rounded border border-zinc-700">
                  <span className="text-zinc-400 uppercase text-[10px]">Type</span>
                  <div className="font-mono text-amber-400 mt-0.5 font-bold">{selectedNode.type}</div>
                </div>
                <div className="bg-zinc-800/80 p-3 rounded border border-zinc-700">
                  <span className="text-zinc-400 uppercase text-[10px]">Shape</span>
                  <div className="font-mono text-zinc-200 mt-0.5">
                    {JSON.stringify(selectedNode.attrs?.shape ?? 'N/A')}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-10 space-y-2">
                <Sliders size={24} className="mx-auto text-zinc-600" />
                <p className="text-xs text-zinc-500">
                  Click any neuron node or layer pill on the canvas to inspect its parameters.
                </p>
              </div>
            )}
          </aside>
        )}
      </div>
    </div>
  );
}