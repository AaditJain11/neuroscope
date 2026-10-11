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
} from 'lucide-react';
import type { ModelView } from './types';
import { NetworkCanvas } from './components/NetworkCanvas';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [modelId, setModelId] = useState<string | null>(null);
  const [modelView, setModelView] = useState<ModelView | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'network' | 'inference' | 'activations'>('network');
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [inputVector, setInputVector] = useState<string>('0.5, -1.2, 0.3, 2.1');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [activationResult, setActivationResult] = useState<any>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setErrorMsg(null);
    setPredictionResult(null);
    setActivationResult(null);

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

      const inspectRes = await fetch(`${API_BASE}/api/models/${newModelId}`);
      if (!inspectRes.ok) {
        throw new Error('Failed to fetch model architecture from backend');
      }

      const view: ModelView = await inspectRes.json();
      setModelView(view);
      setSelectedNodeId(view.nodes[0]?.id || null);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Error uploading file');
    } finally {
      setLoading(false);
      // Reset input value so re-selecting same file triggers onChange
      e.target.value = '';
    }
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
    <div className="flex flex-col h-screen w-screen bg-surface-950 text-zinc-100 select-none">
      <header className="flex items-center justify-between px-6 py-3 border-b border-surface-800 bg-surface-900 z-10">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-brand-amber/10 border border-brand-amber/40 flex items-center justify-center text-brand-amber">
            <Cpu size={18} />
          </div>
          <span className="font-bold text-lg tracking-tight">NeuroScope</span>
          <span className="text-xs bg-surface-800 text-zinc-400 px-2 py-0.5 rounded">v0.1.0</span>
        </div>

        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2 px-3 py-1.5 rounded-md text-sm font-medium bg-surface-850 hover:bg-surface-800 border border-surface-700 cursor-pointer transition">
            {loading ? (
              <Loader2 size={15} className="animate-spin text-brand-amber" />
            ) : (
              <Upload size={15} className="text-brand-amber" />
            )}
            <span>{loading ? 'Uploading & Parsing...' : 'Upload (.keras, .h5, JSON)'}</span>
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

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 border-r border-surface-800 bg-surface-900 flex flex-col p-4 overflow-y-auto space-y-5">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">Model Info</h2>
            {modelView ? (
              <div className="bg-surface-850 p-3 rounded-lg border border-surface-800 space-y-2 text-sm font-mono">
                <div className="truncate text-zinc-200 font-bold">{modelView.name}</div>
                <div className="flex justify-between text-xs text-zinc-400">
                  <span>Architecture</span>
                  <span className="text-brand-amber uppercase">{modelView.kind}</span>
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

          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">Capabilities</h2>
            {modelView ? (
              <div className="space-y-1.5">
                {Object.entries(modelView.capabilities).map(([name, cap]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between text-xs p-2 rounded bg-surface-850 border border-surface-800"
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
                      <span className="text-[10px] uppercase text-zinc-400">{cap.status}</span>
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-zinc-500">Awaiting upload...</p>
            )}
          </div>
        </aside>

        <main className="flex-1 flex flex-col bg-surface-950 overflow-hidden">
          <div className="flex border-b border-surface-800 bg-surface-900 px-4 space-x-1">
            <button
              onClick={() => setActiveTab('network')}
              className={`flex items-center space-x-2 px-4 py-2.5 text-xs font-medium border-b-2 transition ${
                activeTab === 'network'
                  ? 'border-brand-amber text-brand-amber bg-surface-850/50'
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
                  ? 'border-brand-amber text-brand-amber bg-surface-850/50'
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
                  ? 'border-brand-amber text-brand-amber bg-surface-850/50'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Activity size={14} />
              <span>Activations</span>
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
                onSelectNode={(id) => setSelectedNodeId(id)}
                selectedNodeId={selectedNodeId}
              />
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
                    className="w-full bg-surface-850 border border-surface-700 rounded p-2 text-sm font-mono text-zinc-100"
                  />
                </div>
                <button
                  onClick={runInference}
                  className="px-4 py-2 bg-brand-amber text-zinc-950 font-semibold rounded text-xs hover:bg-brand-accent transition"
                >
                  Predict
                </button>
                {predictionResult && (
                  <div className="mt-4 p-4 bg-surface-900 border border-surface-800 rounded">
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
                    className="flex-1 bg-surface-850 border border-surface-700 rounded p-2 text-sm font-mono text-zinc-100"
                  />
                  <button
                    onClick={runActivations}
                    className="px-4 py-2 bg-brand-amber text-zinc-950 font-semibold rounded text-xs hover:bg-brand-accent transition"
                  >
                    Trace
                  </button>
                </div>
                {activationResult && (
                  <div className="space-y-3 mt-4">
                    {Object.entries(activationResult).map(([layerName, values]: any) => (
                      <div key={layerName} className="p-3 bg-surface-900 border border-surface-800 rounded">
                        <span className="text-xs text-brand-amber font-mono font-bold">{layerName}</span>
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

        <aside className="w-80 border-l border-surface-800 bg-surface-900 flex flex-col p-4 overflow-y-auto">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-3 flex items-center space-x-1.5">
            <Sliders size={14} />
            <span>Inspector</span>
          </h2>
          {selectedNode ? (
            <div className="space-y-4 text-xs">
              <div className="bg-surface-850 p-3 rounded border border-surface-800">
                <span className="text-zinc-500 uppercase text-[10px]">Node ID</span>
                <div className="font-mono text-zinc-200 mt-0.5">{selectedNode.id}</div>
              </div>
              <div className="bg-surface-850 p-3 rounded border border-surface-800">
                <span className="text-zinc-500 uppercase text-[10px]">Type</span>
                <div className="font-mono text-brand-amber mt-0.5 font-bold">{selectedNode.type}</div>
              </div>
              <div className="bg-surface-850 p-3 rounded border border-surface-800">
                <span className="text-zinc-500 uppercase text-[10px]">Shape</span>
                <div className="font-mono text-zinc-200 mt-0.5">
                  {JSON.stringify(selectedNode.attrs?.shape ?? 'N/A')}
                </div>
              </div>
              {selectedNode.attrs?.weights && Object.keys(selectedNode.attrs.weights).length > 0 && (
                <div className="bg-surface-850 p-3 rounded border border-surface-800">
                  <span className="text-zinc-500 uppercase text-[10px]">Weights</span>
                  <pre className="text-[11px] font-mono text-zinc-300 mt-1">
                    {JSON.stringify(selectedNode.attrs.weights, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-zinc-500">Select a node from the network canvas.</p>
          )}
        </aside>
      </div>
    </div>
  );
}