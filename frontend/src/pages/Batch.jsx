import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Layers, Zap, CheckCircle2, AlertTriangle, Clock, BarChart2, Trash2, Download } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { saveScanForUser } from '../lib/scans';

const PATHOLOGIES = [
  'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
  'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
  'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
  'Pneumonia', 'Pneumothorax'
];

const STATUS_CONFIG = {
  pending: { label: 'Queued', color: 'text-slate-400', bg: 'bg-slate-400/10', icon: Clock },
  processing: { label: 'Processing', color: 'text-cyan-400', bg: 'bg-cyan-400/10', icon: Zap },
  done: { label: 'Complete', color: 'text-emerald-400', bg: 'bg-emerald-400/10', icon: CheckCircle2 },
  error: { label: 'Error', color: 'text-red-400', bg: 'bg-red-400/10', icon: AlertTriangle },
};

export default function Batch() {
  const { user } = useAuth();
  const fileInputRef = useRef();
  const [files, setFiles] = useState([]);
  const [threshold, setThreshold] = useState(0.4);
  const [running, setRunning] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [stats, setStats] = useState({ processed: 0, anomalies: 0 });

  const addFiles = useCallback((newFiles) => {
    const fileList = Array.from(newFiles).filter(f => f.type.startsWith('image/')).slice(0, 50 - files.length);
    const entries = fileList.map(f => ({
      id: `${f.name}-${Date.now()}-${Math.random()}`,
      file: f,
      name: f.name,
      url: URL.createObjectURL(f),
      status: 'pending',
      topFinding: null,
    }));
    setFiles(prev => [...prev, ...entries]);
  }, [files.length]);

  const removeFile = (id) => setFiles(prev => prev.filter(f => f.id !== id));
  const clearAll = () => { setFiles([]); setStats({ processed: 0, anomalies: 0 }); setRunning(false); };

  const runBatch = async () => {
    if (files.length === 0 || running) return;
    setRunning(true);
    setStats({ processed: 0, anomalies: 0 });
    let processed = 0;
    let anomalies = 0;

    for (let i = 0; i < files.length; i++) {
      // Mark processing
      setFiles(prev => prev.map((f, idx) => idx === i ? { ...f, status: 'processing' } : f));
      
      // Simulate AI delay
      await new Promise(r => setTimeout(r, 800 + Math.random() * 1200));
      
      // Generate results
      const topScore = Math.random();
      const topName = PATHOLOGIES[Math.floor(Math.random() * PATHOLOGIES.length)];
      const hasAnomaly = topScore >= threshold;
      if (hasAnomaly) anomalies++;
      processed++;

      setFiles(prev => prev.map((f, idx) => idx === i ? {
        ...f,
        status: 'done',
        topFinding: { name: topName, score: topScore },
        hasAnomaly,
      } : f));
      setStats({ processed, anomalies });

      // Persist
      saveScanForUser(user.email, {
        title: `[Batch] ${files[i].name}`,
        status: 'completed',
        findings: hasAnomaly ? [`${topName}: ${Math.round(topScore * 100)}%`] : [],
        mode: 'batch',
      });
    }
    setRunning(false);
  };

  const doneCount = files.filter(f => f.status === 'done').length;
  const anomalyCount = files.filter(f => f.hasAnomaly).length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6" style={{ fontFamily: "'Inter', sans-serif" }}>
      
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <span className="p-2 bg-violet-500/10 rounded-lg border border-violet-500/20">
              <Layers size={20} className="text-violet-400" />
            </span>
            Batch Processing
          </h1>
          <p className="text-slate-500 text-sm mt-1 ml-1">Process up to 50 X-rays simultaneously.</p>
        </div>
        {files.length > 0 && (
          <div className="flex items-center gap-4">
            <div className="flex gap-4 text-center">
              <div>
                <p className="text-xl font-extrabold text-white">{files.length}</p>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Selected</p>
              </div>
              <div>
                <p className="text-xl font-extrabold text-emerald-400">{doneCount}</p>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Done</p>
              </div>
              <div>
                <p className="text-xl font-extrabold text-amber-400">{anomalyCount}</p>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Anomalies</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Drop Zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={e => { e.preventDefault(); setDragActive(false); addFiles(e.dataTransfer.files); }}
        onClick={() => fileInputRef.current?.click()}
        className={`rounded-2xl border-2 border-dashed min-h-[140px] flex flex-col items-center justify-center cursor-pointer transition-all mb-6
          ${dragActive ? 'border-violet-400 bg-violet-500/5' : 'border-white/10 hover:border-violet-500/40 hover:bg-violet-500/3'}`}
      >
        <Upload size={28} className={`mb-2 ${dragActive ? 'text-violet-400' : 'text-slate-600'}`} />
        <p className="text-slate-300 font-semibold text-sm">Drop images here or click to select</p>
        <p className="text-slate-600 text-xs mt-1">Supports up to 50 images (PNG, JPG)</p>
        <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={e => addFiles(e.target.files)} />
      </div>

      {files.length > 0 && (
        <>
          {/* Config + Actions */}
          <div className="flex items-center gap-4 mb-5">
            <div className="flex-1 bg-slate-900 rounded-xl border border-white/5 p-4 flex items-center gap-4">
              <label className="text-xs text-slate-500 whitespace-nowrap">Threshold</label>
              <input type="range" min={0} max={1} step={0.01} value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                className="flex-1 h-1.5 rounded-full appearance-none bg-slate-700 cursor-pointer accent-violet-400"
              />
              <span className="text-xs font-bold text-violet-400 w-8">{threshold.toFixed(2)}</span>
            </div>

            <button onClick={clearAll} disabled={running}
              className="px-4 py-3 bg-slate-900 border border-white/5 rounded-xl text-slate-400 hover:text-red-400 hover:border-red-400/20 transition-all flex items-center gap-2 text-sm font-bold">
              <Trash2 size={15} /> Clear
            </button>

            <button onClick={runBatch} disabled={running || doneCount === files.length}
              className="px-6 py-3 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-bold text-sm transition-all flex items-center gap-2 shadow-lg shadow-violet-500/20 disabled:opacity-40 disabled:cursor-not-allowed">
              <Zap size={15} />
              {running ? 'Processing...' : 'Start Batch'}
            </button>
          </div>

          {/* Files Table */}
          <div className="bg-slate-900 rounded-2xl border border-white/5 overflow-hidden">
            <div className="grid grid-cols-12 px-5 py-3 text-[11px] font-bold text-slate-500 uppercase tracking-widest border-b border-white/5">
              <span className="col-span-1"></span>
              <span className="col-span-4">Filename</span>
              <span className="col-span-3">Top Finding</span>
              <span className="col-span-2">Confidence</span>
              <span className="col-span-2">Status</span>
            </div>
            <div className="divide-y divide-white/5 max-h-[400px] overflow-y-auto">
              <AnimatePresence initial={false}>
                {files.map((f) => {
                  const sc = STATUS_CONFIG[f.status];
                  const Icon = sc.icon;
                  return (
                    <motion.div
                      key={f.id}
                      initial={{ opacity: 0, y: -8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, height: 0 }}
                      className="grid grid-cols-12 items-center px-5 py-3 hover:bg-white/2.5 transition-all"
                    >
                      {/* Thumb */}
                      <div className="col-span-1">
                        <img src={f.url} alt={f.name} className="w-9 h-9 rounded-lg object-cover border border-white/5" />
                      </div>
                      {/* Name */}
                      <div className="col-span-4 min-w-0 pr-2">
                        <p className="text-sm font-semibold text-slate-300 truncate">{f.name}</p>
                      </div>
                      {/* Finding */}
                      <div className="col-span-3">
                        {f.topFinding ? (
                          <span className={`text-xs font-bold ${f.hasAnomaly ? 'text-amber-400' : 'text-slate-500'}`}>
                            {f.topFinding.name.replace('_', ' ')}
                          </span>
                        ) : (
                          <span className="text-xs text-slate-700">—</span>
                        )}
                      </div>
                      {/* Score */}
                      <div className="col-span-2">
                        {f.topFinding ? (
                          <span className={`text-xs font-bold ${f.hasAnomaly ? 'text-amber-400' : 'text-slate-600'}`}>
                            {Math.round(f.topFinding.score * 100)}%
                          </span>
                        ) : (
                          f.status === 'processing' ? (
                            <div className="flex gap-1">
                              {[0,1,2].map(i => (
                                <div key={i} className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                              ))}
                            </div>
                          ) : <span className="text-xs text-slate-700">—</span>
                        )}
                      </div>
                      {/* Status badge */}
                      <div className="col-span-2 flex items-center gap-2">
                        <span className={`flex items-center gap-1 text-[10px] font-bold px-2 py-1 rounded-lg ${sc.bg} ${sc.color}`}>
                          <Icon size={10} />
                          {sc.label}
                        </span>
                        {f.status === 'pending' && (
                          <button onClick={() => removeFile(f.id)} className="text-slate-600 hover:text-red-400 transition-colors">
                            <X size={13} />
                          </button>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          </div>

          {/* Download row */}
          {doneCount > 0 && !running && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="mt-4 flex items-center justify-between bg-slate-900 rounded-xl border border-white/5 px-5 py-3"
            >
              <div className="flex items-center gap-3">
                <BarChart2 size={16} className="text-violet-400" />
                <span className="text-sm font-bold text-slate-300">
                  {doneCount} scans processed · {anomalyCount} anomalies detected
                </span>
              </div>
              <button className="flex items-center gap-2 text-xs font-bold text-violet-400 hover:text-violet-300 transition-colors">
                <Download size={13} /> Export CSV
              </button>
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}
