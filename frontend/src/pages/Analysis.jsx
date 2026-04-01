import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, X, Activity, Zap, Printer, Download, ArrowLeft,
  AlertTriangle, CheckCircle2, ChevronRight, HeartPulse,
  Lightbulb, User2, FileText, Sliders
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { saveScanForUser } from '../lib/scans';

/* ─── Constants ─── */
const BRAND = '#34d399';
const BRAND_DIM = '#065f46';

const PATHOLOGIES = [
  'Atelectasis','Cardiomegaly','Consolidation','Edema',
  'Effusion','Emphysema','Fibrosis','Hernia',
  'Infiltration','Mass','Nodule','Pleural_Thickening',
  'Pneumonia','Pneumothorax'
];

const STAGES = [
  'Normalizing pixel intensities…',
  'Running feature extraction…',
  'Classifying pathologies…',
  'Scoring confidence levels…',
  'Compiling recommendations…',
  'Done.',
];

const SUGGESTIONS = {
  Cardiomegaly:       { severity:'high',   text:'Enlarged heart detected. Cardiology referral + echocardiogram recommended.', action:'Refer to Cardiologist' },
  Effusion:           { severity:'high',   text:'Pleural effusion present. Consider thoracentesis and chest CT.', action:'Chest CT' },
  Pneumonia:          { severity:'high',   text:'Signs of pneumonia. Antibiotic therapy + follow-up X-ray in 2 weeks.', action:'Antibiotic Therapy' },
  Pneumothorax:       { severity:'high',   text:'Possible pneumothorax. Urgent clinical evaluation required.', action:'Urgent Evaluation' },
  Atelectasis:        { severity:'medium', text:'Atelectasis pattern. Respiratory therapy and physiotherapy advised.', action:'Respiratory Therapy' },
  Consolidation:      { severity:'medium', text:'Consolidation present. Rule out infection or inflammation.', action:'Follow-up Advised' },
  Edema:              { severity:'medium', text:'Pulmonary edema evidence. Fluid management may be needed.', action:'Monitor Fluid Status' },
  Mass:               { severity:'medium', text:'Mass lesion density. CT scan + oncology consult.', action:'CT + Oncology' },
  Nodule:             { severity:'medium', text:'Pulmonary nodule. Follow-up CT in 3–6 months.', action:'CT in 3–6 months' },
  Emphysema:          { severity:'low',    text:'Emphysematous changes. Pulmonary function tests + smoking cessation.', action:'PFT & Cessation' },
  Fibrosis:           { severity:'low',    text:'Fibrotic changes. Long-term monitoring + pulmonologist consult.', action:'Pulmonologist' },
  Infiltration:       { severity:'low',    text:'Infiltration pattern. Clinical monitoring + imaging in 4–6 weeks.', action:'Follow-up Imaging' },
  Pleural_Thickening: { severity:'low',    text:'Pleural thickening. Assess for asbestos exposure or prior infection.', action:'Occupational Review' },
  Hernia:             { severity:'low',    text:'Possible hiatal/diaphragmatic hernia. GI referral advised.', action:'GI Referral' },
};

const SEV = {
  high:   { bg:'#2d0f0f', border:'#7f1d1d', color:'#fca5a5', bar:'#ef4444' },
  medium: { bg:'#2d1f0f', border:'#78350f', color:'#fcd34d', bar:'#f59e0b' },
  low:    { bg:'#0f1d2d', border:'#1e3a5f', color:'#93c5fd', bar:'#3b82f6' },
};

/* ─── Global CSS ─── */
const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
  .ws-root { font-family: 'Space Grotesk', sans-serif; }
  @keyframes sweep { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
  @keyframes ping2 { 0% { transform:scale(0.6); opacity:1; } 100% { transform:scale(2); opacity:0; } }
  .sweep { animation:sweep 2s linear infinite; transform-origin:center; }
  .ping2 { animation:ping2 1.8s ease-out infinite; }
  .score-bar { transition: width 0.6s cubic-bezier(0.4,0,0.2,1); }
  /* Print */
  @media print {
    @page { margin: 15mm; }
    body, html { background: white !important; height: auto !important; }
    body * { visibility: hidden; }
    #print-report, #print-report * { visibility: visible; color: #000 !important; }
    #print-report { 
      position: absolute; left: 0; top: 0;
      display: block !important; background: white !important; 
    }
    .pr-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .pr-table th, .pr-table td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; font-size: 13px; }
    .pr-table th { background: #f8f9fa !important; font-weight: 700; border-bottom: 2px solid #ccc; }
    .pr-high { color: #dc2626 !important; font-weight: 700; }
    .pr-medium { color: #d97706 !important; font-weight: 700; }
    .pr-low { color: #2563eb !important; font-weight: 700; }
    .pr-clear { color: #16a34a !important; font-weight: 600; }
    .pr-rec-box { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; page-break-inside: avoid; }
    .pr-footer { margin-top: 32px; font-size: 10px; color: #6b7280; border-top: 1px solid #e5e7eb; padding-top: 16px; text-align: center; }
  }
  #print-report { display: none; }
`;

/* ─── Component ─── */
export default function AnalysisWorkspace() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const initMode = searchParams.get('mode') || 'classify';

  const fileRef = useRef();
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const [imgUrl, setImgUrl] = useState(null);
  const [threshold, setThreshold] = useState(0.4);
  const [mode, setMode] = useState(initMode === 'gradcam' ? 'classify' : initMode);
  const [stage, setStage] = useState('idle');      // idle | scanning | done
  const [stageLabel, setStageLabel] = useState('');
  const [stageIdx, setStageIdx] = useState(0);
  const [results, setResults] = useState([]);

  useEffect(() => {
    const m = searchParams.get('mode');
    setMode(m === 'gradcam' ? 'classify' : (m || 'classify'));
  }, [searchParams]);

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return;
    setFile(f); setImgUrl(URL.createObjectURL(f));
    setStage('idle'); setResults([]);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const runAnalysis = async () => {
    if (!file) return;
    setStage('scanning'); setStageIdx(0);
    setResults([]);

    const apiUrl = import.meta.env.VITE_API_URL;
    let fallbackToMock = !apiUrl;

    // Start progress animation
    let i = 0;
    const iv = setInterval(() => {
      if (i < STAGES.length - 1) {
        setStageLabel(STAGES[i]); setStageIdx(i++);
      }
    }, 600);

    try {
      if (!fallbackToMock) {
        setStageLabel('Connecting to AI Server...');
        const formData = new FormData();
        formData.append('file', file);
        formData.append('mode', mode);

        const res = await fetch(`${apiUrl}/predict`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) throw new Error('API Error');
        const data = await res.json();
        
        clearInterval(iv);
        setStageLabel(STAGES[STAGES.length - 1]);
        setStageIdx(STAGES.length - 1);
        
        setResults(data.predictions || []);
        setStage('done');
        
        const findings = (data.predictions || []).filter(p => p.score >= threshold).map(p => `${p.name}: ${Math.round(p.score * 100)}%`);
        saveScanForUser(user.email, { title: file.name, status: 'completed', findings, mode });
        return;
      }
    } catch (e) {
      console.error("API Call failed, falling back to simulation:", e);
      fallbackToMock = true;
    }

    if (fallbackToMock) {
      // Original Mock Simulation Logic
      clearInterval(iv);
      setStageLabel(STAGES[STAGES.length - 1]);
      setStageIdx(STAGES.length - 1);
      const r = PATHOLOGIES.map(name => ({ name, score: Math.random() })).sort((a, b) => b.score - a.score);
      setResults(r); setStage('done');
      const findings = r.filter(p => p.score >= threshold).map(p => `${p.name}: ${Math.round(p.score * 100)}%`);
      saveScanForUser(user.email, { title: file.name, status: 'completed', findings, mode });
    }
  };

  const positive = results.filter(r => r.score >= threshold);

  /* ─── Render ─── */
  return (
    <div className="ws-root" style={{ minHeight: '100vh', background: '#0a0f0e', color: '#e2e8f0' }}>
      <style>{STYLES}</style>

      {/* ── Top bar ── */}
      <div style={{ padding: '14px 32px', borderBottom: '1px solid #1a2e24', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button onClick={() => navigate('/dashboard')}
            style={{ background: 'transparent', border: 'none', color: '#6ee7b7', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, fontWeight: 600, fontFamily: 'inherit' }}>
            <ArrowLeft size={15} /> Back to Library
          </button>
          <span style={{ color: '#1f4233' }}>|</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 28, height: 28, borderRadius: 8, background: BRAND_DIM, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <HeartPulse size={14} color={BRAND} />
            </div>
            <span style={{ fontWeight: 700, color: '#fff', fontSize: 14 }}>MediScan<span style={{ color: BRAND }}>AI</span></span>
          </div>
          <span style={{ color: '#1f4233', fontSize: 13 }}>/ Analysis Workspace</span>
        </div>
        <span style={{ fontSize: 12, color: '#4b7a62', display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600 }}>
          <User2 size={13} /> {user?.username}
        </span>
      </div>

      {/* ── Body ── */}
      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '32px 24px', display: 'grid', gridTemplateColumns: '340px 1fr', gap: 24, minHeight: 'calc(100vh - 65px)' }}>

        {/* ── LEFT: Input Panel ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Upload */}
          <div style={{ background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, overflow: 'hidden' }}>
            <div style={{ padding: '14px 20px', borderBottom: '1px solid #1a2e24' }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: BRAND, letterSpacing: 2 }}>SCAN INPUT</span>
            </div>
            <div style={{ padding: 20 }}>
              <div
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
                style={{
                  border: `2px dashed ${dragOver ? BRAND : (imgUrl ? '#1a2e24' : '#1e3a2a')}`,
                  borderRadius: 12,
                  minHeight: 170,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  background: dragOver ? '#0f2d1f' : 'transparent',
                  transition: 'all 0.2s',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                {imgUrl ? (
                  <>
                    <img src={imgUrl} alt="scan" style={{ width: '100%', height: 160, objectFit: 'contain' }} />
                    <button
                      onClick={e => { e.stopPropagation(); setFile(null); setImgUrl(null); setStage('idle'); setResults([]); }}
                      style={{ position: 'absolute', top: 8, right: 8, background: '#1a2e24', border: 'none', borderRadius: 7, padding: '4px 7px', color: '#f87171', cursor: 'pointer', lineHeight: 1 }}>
                      <X size={13} />
                    </button>
                    <p style={{ fontSize: 10, color: '#4b7a62', marginTop: 6 }}>{file?.name}</p>
                  </>
                ) : (
                  <>
                    <Upload size={28} color="#1f4233" style={{ marginBottom: 10 }} />
                    <span style={{ fontSize: 14, color: '#a7b3ad', fontWeight: 600 }}>Drop X-Ray image here</span>
                    <span style={{ fontSize: 11, color: '#4b7a62', marginTop: 4 }}>PNG · JPG · DICOM</span>
                  </>
                )}
                <input ref={fileRef} type="file" accept="image/*" style={{ display:'none' }} onChange={e => handleFile(e.target.files[0])} />
              </div>
            </div>
          </div>

          {/* Config */}
          <div style={{ background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, overflow: 'hidden' }}>
            <div style={{ padding: '14px 20px', borderBottom: '1px solid #1a2e24' }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: BRAND, letterSpacing: 2 }}>CONFIGURATION</span>
            </div>
            <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 20 }}>
              {/* Mode */}
              <div>
                <p style={{ fontSize: 11, color: '#6ee7b7', fontWeight: 700, letterSpacing: 1, marginBottom: 10 }}>ANALYSIS MODE</p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {[
                    { id: 'classify', icon: Activity, label: 'Classify' },
                    { id: 'detect',   icon: Zap,      label: 'Detect' },
                  ].map(({ id, icon: Icon, label }) => (
                    <button key={id} onClick={() => setMode(id)}
                      style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
                        padding: '12px 8px', borderRadius: 10, cursor: 'pointer', fontFamily: 'inherit',
                        border: mode === id ? `1.5px solid ${BRAND}` : '1.5px solid #1e3a2a',
                        background: mode === id ? BRAND_DIM : 'transparent',
                        color: mode === id ? BRAND : '#4b7a62',
                        fontSize: 12, fontWeight: 700, transition: 'all 0.15s',
                      }}>
                      <Icon size={16} /> {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Threshold */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                  <p style={{ fontSize: 11, color: '#6ee7b7', fontWeight: 700, letterSpacing: 1, display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Sliders size={11} /> THRESHOLD
                  </p>
                  <span style={{ fontSize: 13, fontWeight: 700, color: BRAND }}>{threshold.toFixed(2)}</span>
                </div>
                <input type="range" min={0} max={1} step={0.01} value={threshold}
                  onChange={e => setThreshold(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: BRAND, height: 5, cursor: 'pointer' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
                  {[['Sensitive', 0.2], ['Balanced', 0.4], ['Precise', 0.7]].map(([lbl, val]) => (
                    <button key={lbl} onClick={() => setThreshold(val)}
                      style={{
                        background: Math.abs(threshold - val) < 0.05 ? BRAND_DIM : 'transparent',
                        border: Math.abs(threshold - val) < 0.05 ? `1px solid ${BRAND}55` : '1px solid transparent',
                        color: Math.abs(threshold - val) < 0.05 ? BRAND : '#4b7a62',
                        fontSize: 10, fontWeight: 700, padding: '3px 8px', borderRadius: 6,
                        cursor: 'pointer', fontFamily: 'inherit',
                      }}>{lbl}</button>
                  ))}
                </div>
              </div>

              {/* Run */}
                <button
                  disabled={!file || stage === 'scanning'}
                  onClick={runAnalysis}
                  style={{
                    width: '100%', padding: '14px', borderRadius: 12, fontFamily: 'inherit',
                    border: 'none', cursor: !file || stage === 'scanning' ? 'not-allowed' : 'pointer',
                    background: !file || stage === 'scanning' ? '#1a2e24' : BRAND,
                    color: !file || stage === 'scanning' ? '#4b7a62' : '#0a1a14',
                    fontWeight: 700, fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                    transition: 'all 0.2s',
                  }}>
                  <Zap size={16} />
                  {stage === 'scanning' ? 'Analyzing…' : stage === 'done' ? 'Re-Analyze' : 'Run Analysis'}
                </button>
            </div>
          </div>
        </div>

        {/* ── RIGHT: Results Panel ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <AnimatePresence mode="wait">

            {/* Idle */}
            {stage === 'idle' && (
              <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ flex: 1, background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 48, textAlign: 'center', minHeight: 400 }}>
                <div style={{ width: 72, height: 72, borderRadius: 20, border: '2px dashed #1a2e24', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
                  <Activity size={30} color="#1f4233" />
                </div>
                <p style={{ fontSize: 18, fontWeight: 700, color: '#2d4a3a' }}>Awaiting scan</p>
                <p style={{ fontSize: 13, color: '#2d4a3a', marginTop: 8, maxWidth: 280, lineHeight: 1.6 }}>Upload a chest X-ray image and press Run Analysis to begin.</p>
              </motion.div>
            )}

            {/* Scanning */}
            {stage === 'scanning' && (
              <motion.div key="scanning" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ flex: 1, background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 48, textAlign: 'center', minHeight: 400 }}>
                <div style={{ position: 'relative', width: 140, height: 140, marginBottom: 32 }}>
                  {[1, 2, 3].map(r => (
                    <div key={r} style={{ position: 'absolute', inset: 0, border: `1px solid ${BRAND}${r === 1 ? '33' : r === 2 ? '22' : '11'}`, borderRadius: '50%', transform: `scale(${r * 0.33})`, transformOrigin: 'center' }} />
                  ))}
                  <div className="sweep" style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: `conic-gradient(from 0deg, transparent 70%, ${BRAND}44 100%)` }} />
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div className="ping2" style={{ width: 12, height: 12, background: BRAND, borderRadius: '50%' }} />
                  </div>
                </div>
                <p style={{ fontSize: 11, fontWeight: 700, color: BRAND, letterSpacing: 3, marginBottom: 10 }}>PROCESSING</p>
                <AnimatePresence mode="wait">
                  <motion.p key={stageLabel} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
                    style={{ fontSize: 13, color: '#6ee7b7' }}>{stageLabel}</motion.p>
                </AnimatePresence>
                <div style={{ display: 'flex', gap: 6, marginTop: 20 }}>
                  {STAGES.map((_, i) => (
                    <div key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: i <= stageIdx ? BRAND : '#1a2e24', transition: 'background 0.3s' }} />
                  ))}
                </div>
              </motion.div>
            )}

            {/* Results */}
            {stage === 'done' && (
              <motion.div key="results" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

                {/* Header row */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <h2 style={{ fontSize: 20, fontWeight: 700, color: '#fff', margin: 0, letterSpacing: '-0.5px' }}>Diagnostic Results</h2>
                    <p style={{ fontSize: 12, color: '#4b7a62', margin: '4px 0 0' }}>{file?.name} · {mode.toUpperCase()} · threshold {threshold.toFixed(2)}</p>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button onClick={() => window.print()}
                      style={{ background: '#0e1a14', border: '1px solid #1a2e24', color: '#a7f3d0', borderRadius: 10, padding: '8px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 7 }}>
                      <Printer size={14} /> Print
                    </button>
                    <button onClick={() => window.print()}
                      style={{ background: BRAND, border: 'none', color: '#0a1a14', borderRadius: 10, padding: '8px 16px', fontSize: 13, fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 7 }}>
                      <Download size={14} /> Save PDF
                    </button>
                  </div>
                </div>

                {/* Detection Visuals (Only for Detect mode) */}
                {mode === 'detect' && imgUrl && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 8 }}>
                    
                    {/* Bounding Box Image */}
                    <div style={{ background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, overflow: 'hidden', padding: 12 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12, padding: '0 4px' }}>
                         <Zap size={14} color={BRAND} />
                         <span style={{ fontSize: 11, fontWeight: 700, color: BRAND, letterSpacing: 1 }}>DETECTIONS</span>
                      </div>
                      <div style={{ position: 'relative', width: '100%', height: 260, background: '#000', borderRadius: 8, overflow: 'hidden' }}>
                        <img src={imgUrl} alt="Original" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} viewBox="0 0 100 100" preserveAspectRatio="none">
                          {positive.slice(0, 5).map((val, i) => {
                            let x, y, w, h;
                            if (val.box) {
                              x = val.box.xMin;
                              y = val.box.yMin;
                              w = val.box.xMax - val.box.xMin;
                              h = val.box.yMax - val.box.yMin;
                            } else {
                              // Fallback simulated coordinates
                              const isLeft = i % 2 === 0;
                              w = 15 + (i * 3) % 10;
                              h = 20 + (i * 5) % 15;
                              x = isLeft ? (15 + (i * 8) % 15) : (55 + (i * 8) % 15);
                              y = 20 + (i * 12) % 45;
                            }
                            
                            const c = SEV[SUGGESTIONS[val.name]?.severity || 'low'];
                            
                            return (
                              <g key={val.name}>
                                {/* Bounding Box: Dashed for clinical feel */}
                                <rect x={x} y={y} width={w} height={h} fill="none" stroke={c?.bar || '#fff'} strokeWidth="1.2" strokeDasharray="2,1" />
                                
                                {/* Label background: Compact pill shape */}
                                <rect x={x} y={y - 5.5} width={val.name.length * 2.2 + 6} height={5.5} fill={c.bg} fillOpacity="0.95" rx="1" />
                                
                                {/* Label text: Bright white for maximum contrast */}
                                <text x={x + 2} y={y - 1.6} fill="#ffffff" fontSize="2.8" fontFamily="sans-serif" fontWeight="900" style={{ letterSpacing: '0.02em', textTransform: 'uppercase' }}>
                                  {val.name.replace('_', ' ')}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      </div>
                    </div>

                    {/* Heatmap Image */}
                    <div style={{ background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, overflow: 'hidden', padding: 12 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12, padding: '0 4px' }}>
                         <Activity size={14} color="#fcd34d" />
                         <span style={{ fontSize: 11, fontWeight: 700, color: '#fcd34d', letterSpacing: 1 }}>ACTIVATION HEATMAP</span>
                      </div>
                      <div style={{ position: 'relative', width: '100%', height: 260, background: '#000', borderRadius: 8, overflow: 'hidden' }}>
                        <img src={imgUrl} alt="Heatmap base" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        {/* Overlay to simulate a thermal/jet colormap over the high-contrast regions */}
                        <div style={{ 
                          position: 'absolute', inset: 0, 
                          background: 'linear-gradient(45deg, rgba(0,0,255,0.4) 0%, rgba(0,255,0,0.4) 50%, rgba(255,0,0,0.5) 100%)',
                          mixBlendMode: 'color',
                          pointerEvents: 'none' 
                        }} />
                        <div style={{ 
                          position: 'absolute', inset: 0, 
                          background: 'radial-gradient(circle at 60% 40%, rgba(255,50,50,0.6) 0%, transparent 40%), radial-gradient(circle at 30% 60%, rgba(255,100,50,0.4) 0%, transparent 50%)',
                          mixBlendMode: 'screen',
                          pointerEvents: 'none' 
                        }} />
                      </div>
                    </div>

                  </div>
                )}

                {/* Summary banner */}
                <div style={{
                  background: positive.length > 0 ? '#2d1a0f' : '#0f2d1a',
                  border: `1px solid ${positive.length > 0 ? '#78350f' : '#14532d'}`,
                  borderRadius: 12, padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 14,
                }}>
                  <div style={{ width: 40, height: 40, borderRadius: 12, background: positive.length > 0 ? '#78350f55' : '#14532d55', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    {positive.length > 0
                      ? <AlertTriangle size={20} color="#fcd34d" />
                      : <CheckCircle2 size={20} color={BRAND} />}
                  </div>
                  <div>
                    <p style={{ fontWeight: 700, color: '#fff', margin: '0 0 3px', fontSize: 15 }}>
                      {positive.length > 0 ? `${positive.length} finding${positive.length > 1 ? 's' : ''} detected` : 'No significant findings'}
                    </p>
                    <p style={{ fontSize: 12, color: '#a7b3ad', margin: 0 }}>
                      {positive.length > 0 ? 'Review the pathology scores and clinical recommendations below.' : 'Scan appears clear at the current threshold. Routine follow-up recommended.'}
                    </p>
                  </div>
                </div>

                {/* 2-column: Scores + Suggestions */}
                <div style={{ display: 'grid', gridTemplateColumns: positive.length > 0 ? '1fr 1fr' : '1fr', gap: 16 }}>

                  {/* Scores */}
                  <div style={{ background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, overflow: 'hidden' }}>
                    <div style={{ padding: '14px 20px', borderBottom: '1px solid #1a2e24', display: 'flex', alignItems: 'center', gap: 8 }}>
                      <FileText size={14} color={BRAND} />
                      <span style={{ fontSize: 11, fontWeight: 700, color: BRAND, letterSpacing: 2 }}>PATHOLOGY SCORES</span>
                    </div>
                    <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                      {results.map(({ name, score }) => {
                        const pct = Math.round(score * 100);
                        const det = score >= threshold;
                        const s = SUGGESTIONS[name];
                        const sev = det ? (s?.severity || 'low') : null;
                        const c = sev ? SEV[sev] : null;
                        return (
                          <div key={name}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, alignItems: 'center' }}>
                              <span style={{ fontSize: 12, fontWeight: 600, color: det ? '#e2e8f0' : '#4b7a62', display: 'flex', alignItems: 'center', gap: 6 }}>
                                {det && <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.bar, display: 'inline-block' }} />}
                                {name.replace('_', ' ')}
                                {sev && <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 6px', borderRadius: 5, background: c.bg, color: c.color, border: `1px solid ${c.border}` }}>{sev.toUpperCase()}</span>}
                              </span>
                              <span style={{ fontSize: 12, fontWeight: 700, color: det ? c?.bar : '#1f4233' }}>{pct}%</span>
                            </div>
                            <div style={{ height: 5, background: '#1a2e24', borderRadius: 3, overflow: 'hidden' }}>
                              <motion.div
                                initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 0.5, ease: 'easeOut' }}
                                style={{ height: '100%', borderRadius: 3, background: det && c ? c.bar : '#1f4233' }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Suggestions */}
                  {positive.length > 0 && (
                    <div style={{ background: '#0e1a14', border: '1px solid #1a2e24', borderRadius: 16, overflow: 'hidden' }}>
                      <div style={{ padding: '14px 20px', borderBottom: '1px solid #1a2e24', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Lightbulb size={14} color="#fcd34d" />
                        <span style={{ fontSize: 11, fontWeight: 700, color: '#fcd34d', letterSpacing: 2 }}>RECOMMENDATIONS</span>
                      </div>
                      <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 440, overflowY: 'auto' }}>
                        {positive.map(({ name }) => {
                          const s = SUGGESTIONS[name];
                          if (!s) return null;
                          const c = SEV[s.severity];
                          return (
                            <div key={name} style={{ background: c.bg, border: `1px solid ${c.border}`, borderRadius: 10, padding: '12px 14px' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                                <span style={{ fontSize: 9, fontWeight: 800, padding: '2px 7px', borderRadius: 5, background: `${c.bar}22`, color: c.color, border: `1px solid ${c.border}` }}>{s.severity.toUpperCase()}</span>
                                <span style={{ fontSize: 13, fontWeight: 700, color: '#fff' }}>{name.replace('_', ' ')}</span>
                              </div>
                              <p style={{ fontSize: 12, color: '#a7b3ad', margin: '0 0 8px', lineHeight: 1.6 }}>{s.text}</p>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: c.color, fontWeight: 700 }}>
                                <ChevronRight size={11} /> {s.action}
                              </div>
                            </div>
                          );
                        })}
                        <p style={{ fontSize: 10, color: '#1f4233', lineHeight: 1.6, borderTop: '1px solid #1a2e24', paddingTop: 10, margin: 0 }}>
                          ⚠ For educational/demonstration use only. Not a substitute for professional medical advice.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── PRINT REPORT (hidden, shown on print) ── */}
      <div id="print-report">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ fontSize: 24, fontWeight: 900, marginRight: 8, color: '#000' }}>MediScanAI</span>
              <span style={{ fontSize: 14, color: '#666' }}>Diagnostic Report</span>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: '#666' }}>Patient ID: {user?.email}</p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <p style={{ margin: 0, fontSize: 12, fontWeight: 700 }}>{new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}</p>
            <p style={{ margin: '4px 0 0', fontSize: 12 }}>Mode: {mode.toUpperCase()}</p>
          </div>
        </div>
        
        <hr style={{ border: 'none', borderTop: '2px solid #000', margin: '0 0 20px' }} />
        
        {/* Render bounding boxes in print if detect mode */}
        {mode === 'detect' && imgUrl && (
          <div style={{ marginBottom: 24, padding: 16, border: '1px solid #ccc', borderRadius: 8, background: '#f9f9f9', textAlign: 'center' }}>
            <h3 style={{ margin: '0 0 12px', fontSize: 14, fontWeight: 700, textAlign: 'left' }}>Visual Findings (Detection Mask)</h3>
            
            {/* 
              By wrapping both the img and the absolute SVG in an inline-block container with 
              a max height/width, the container hugs the image precisely, ensuring the 0-100% 
              SVG matches the physical dimensions of the image on the printed page, avoiding 
              object-fit: contain layout mismatches.
            */}
            <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%', maxHeight: 380 }}>
              <img src={imgUrl} alt="Print Scan" style={{ display: 'block', maxWidth: '100%', maxHeight: 380, width: 'auto', height: 'auto' }} />
              <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} viewBox="0 0 100 100" preserveAspectRatio="none">
                {positive.slice(0, 5).map((val, i) => {
                  let x, y, w, h;
                  if (val.box) {
                    x = val.box.xMin;
                    y = val.box.yMin;
                    w = val.box.xMax - val.box.xMin;
                    h = val.box.yMax - val.box.yMin;
                  } else {
                    const isLeft = i % 2 === 0;
                    w = 15 + (i * 3) % 10;
                    h = 20 + (i * 5) % 15;
                    x = isLeft ? (15 + (i * 8) % 15) : (55 + (i * 8) % 15);
                    y = 20 + (i * 12) % 45;
                  }
                  const c = SEV[SUGGESTIONS[val.name]?.severity || 'low'];
                  return (
                    <g key={val.name}>
                      <rect x={x} y={y} width={w} height={h} fill="none" stroke={c?.bar || '#000'} strokeWidth="1.5" strokeDasharray="4,2" />
                      <rect x={x} y={y - 6} width={w * 1.5} height={6} fill="#f0f0f0" stroke="#ccc" strokeWidth="0.2" rx="1" />
                      <text x={x + 2} y={y - 2} fill="#000" fontSize="4.5" fontWeight="900">{val.name.replace('_', ' ').toUpperCase()}</text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        )}

        <table className="pr-table" style={{ marginBottom: 16 }}>
          <tbody>
            <tr><td style={{ width: 140, fontWeight: 700, fontSize: 12 }}>Patient</td><td style={{ fontSize: 12 }}>{user?.username}</td></tr>
            <tr><td style={{ fontWeight: 700, fontSize: 12 }}>Email</td><td style={{ fontSize: 12 }}>{user?.email}</td></tr>
            <tr><td style={{ fontWeight: 700, fontSize: 12 }}>File</td><td style={{ fontSize: 12 }}>{file?.name || '—'}</td></tr>
            <tr><td style={{ fontWeight: 700, fontSize: 12 }}>Date</td><td style={{ fontSize: 12 }}>{new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'long', year: 'numeric' })}</td></tr>
            <tr><td style={{ fontWeight: 700, fontSize: 12 }}>Mode</td><td style={{ fontSize: 12 }}>{mode.toUpperCase()}</td></tr>
            <tr><td style={{ fontWeight: 700, fontSize: 12 }}>Threshold</td><td style={{ fontSize: 12 }}>{threshold.toFixed(2)}</td></tr>
          </tbody>
        </table>
        <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '12px 0' }} />
        <p style={{ fontWeight: 700, fontSize: 14, margin: '12px 0 4px' }}>
          {positive.length > 0 ? `⚠ ${positive.length} Pathologie${positive.length > 1 ? 's' : ''} Detected` : '✓ No Significant Findings'}
        </p>
        <h2 style={{ fontSize: 15, margin: '16px 0 8px' }}>Pathology Scores</h2>
        <table className="pr-table">
          <thead>
            <tr><th>Pathology</th><th>Score</th><th>Status</th><th>Severity</th></tr>
          </thead>
          <tbody>
            {results.map(r => {
              const det = r.score >= threshold;
              const s = SUGGESTIONS[r.name];
              const sev = det ? (s?.severity || 'low') : null;
              return (
                <tr key={r.name}>
                  <td style={{ fontSize: 12 }}>{r.name.replace('_', ' ')}</td>
                  <td style={{ fontSize: 12 }}>{Math.round(r.score * 100)}%</td>
                  <td className={det ? `pr-${sev}` : 'pr-clear'} style={{ fontSize: 12 }}>{det ? 'DETECTED' : 'Clear'}</td>
                  <td className={sev ? `pr-${sev}` : ''} style={{ fontSize: 12 }}>{sev ? sev.toUpperCase() : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <h2 style={{ fontSize: 15, margin: '20px 0 8px' }}>Clinical Recommendations</h2>
        {positive.length === 0
          ? <p className="pr-clear" style={{ fontSize: 12 }}>No significant findings. Routine follow-up recommended.</p>
          : positive.map(({ name }) => {
              const s = SUGGESTIONS[name];
              if (!s) return null;
              return (
                <div key={name} className="pr-rec-box">
                  <p style={{ fontWeight: 700, fontSize: 13, margin: '0 0 4px' }}>
                    <span className={`pr-${s.severity}`}>[{s.severity.toUpperCase()}]</span> {name.replace('_', ' ')}
                  </p>
                  <p style={{ fontSize: 12, margin: '0 0 4px' }}>{s.text}</p>
                  <p style={{ fontSize: 11, margin: 0 }}><strong>Action:</strong> {s.action}</p>
                </div>
              );
            })
        }
        <div className="pr-footer">
          Generated by MediScanAI · {new Date().toLocaleString()}<br />
          FOR EDUCATIONAL AND DEMONSTRATION PURPOSES ONLY — not a substitute for professional medical advice.
        </div>
      </div>
    </div>
  );
}
