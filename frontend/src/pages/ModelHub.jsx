import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { Activity, Layers, LogOut, ChevronRight, Zap, User2, Shield, HeartPulse } from 'lucide-react';

/* ── unique color tokens ── */
const BRAND = '#34d399';      // emerald-400 — our signature accent
const BRAND_DIM = '#065f46';  // emerald-900

const MODELS = [
  {
    id: 'classify',
    icon: Activity,
    tag: '14 PATHOLOGIES',
    name: 'Chest X-Ray Classifier',
    blurb: 'Deep-learning classification of 14 chest conditions powered by DenseNet-121.',
    metrics: [{ label: 'AUC', val: '0.905' }, { label: 'Sensitivity', val: '92%' }],
    path: '/analysis?mode=classify',
    num: '01',
  },
  {
    id: 'detect',
    icon: Zap,
    tag: 'OBJECT DETECTION',
    name: 'Lesion Detector',
    blurb: 'Draws bounding boxes around lung anomalies in real time using YOLOv8.',
    metrics: [{ label: 'mAP50', val: '0.176' }, { label: 'Precision', val: '88%' }],
    path: '/analysis?mode=detect',
    num: '02',
  },
];

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
  .hub-root { font-family: 'Space Grotesk', sans-serif; }
  .card-line::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    border: 1.5px solid transparent;
    background: linear-gradient(135deg, #34d39940, transparent 60%) border-box;
    -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: destination-out;
    mask-composite: exclude;
    pointer-events: none;
    transition: opacity 0.3s;
    opacity: 0;
  }
  .model-card:hover .card-line::after { opacity: 1; }
  @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
  .float { animation: float 4s ease-in-out infinite; }
`;

const StatBadge = ({ label, val }) => (
  <div style={{ background: '#0f2d1f', border: '1px solid #1a4a34', borderRadius: 8, padding: '4px 10px', display: 'inline-flex', flexDirection: 'column', alignItems: 'center' }}>
    <span style={{ fontSize: 15, fontWeight: 700, color: BRAND, lineHeight: 1.2 }}>{val}</span>
    <span style={{ fontSize: 9, color: '#6ee7b7', letterSpacing: 1, fontWeight: 600 }}>{label}</span>
  </div>
);

export default function ModelHub() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="hub-root" style={{ minHeight: '100vh', background: '#0a0f0e', color: '#e2e8f0', fontFamily: "'Space Grotesk', sans-serif" }}>
      <style>{STYLES}</style>

      {/* ── Nav ── */}
      <nav style={{ borderBottom: '1px solid #1a2e24', padding: '16px 40px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 36, height: 36, borderRadius: 10, background: BRAND_DIM, display: 'flex', alignItems: 'center', justifyContent: 'center', border: `1.5px solid ${BRAND}33` }}>
            <HeartPulse size={18} color={BRAND} />
          </div>
          <span style={{ fontSize: 18, fontWeight: 700, color: '#fff', letterSpacing: '-0.5px' }}>MediScan<span style={{ color: BRAND }}>AI</span></span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ background: '#111d16', border: '1px solid #1a2e24', borderRadius: 10, padding: '6px 14px', fontSize: 13, color: '#a7f3d0', display: 'flex', alignItems: 'center', gap: 7 }}>
            <User2 size={13} /> {user?.username}
          </div>
          <button onClick={() => { logout(); navigate('/auth'); }}
            style={{ background: 'transparent', border: '1px solid #2d1f1f', borderRadius: 10, padding: '6px 14px', fontSize: 13, color: '#f87171', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 7 }}>
            <LogOut size={13} /> Sign out
          </button>
        </div>
      </nav>

      {/* ── Hero ── */}
      <div style={{ padding: '64px 40px 40px', maxWidth: 960, margin: '0 auto', textAlign: 'center' }}>
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="float"
          style={{ width: 72, height: 72, borderRadius: 20, background: BRAND_DIM, border: `2px solid ${BRAND}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px' }}
        >
          <Shield size={32} color={BRAND} />
        </motion.div>
        <motion.h1 initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.1 }}
          style={{ fontSize: 42, fontWeight: 700, letterSpacing: '-1.5px', color: '#fff', margin: '0 0 12px' }}>
          AI Diagnostic<br />
          <span style={{ color: BRAND }}>Model Library</span>
        </motion.h1>
        <motion.p initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}
          style={{ fontSize: 15, color: '#6ee7b7', maxWidth: 480, margin: '0 auto 48px', lineHeight: 1.7 }}>
          Select a model, upload your scan, and receive{' '}
          <span style={{ color: BRAND, fontWeight: 600 }}>evidence-based diagnostic insights</span> in seconds.
        </motion.p>
      </div>

      {/* ── Model Cards ── */}
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 40px 80px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {MODELS.map((m, idx) => {
          const Icon = m.icon;
          return (
            <motion.div
              key={m.id}
              className="model-card"
              initial={{ x: -24, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: idx * 0.1 + 0.3, type: 'spring', stiffness: 300 }}
              whileHover={{ x: 6 }}
              onClick={() => navigate(m.path)}
              style={{
                position: 'relative',
                background: '#0e1a14',
                borderRadius: 20,
                border: '1px solid #1a2e24',
                padding: '36px 28px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                gap: 20,
                cursor: 'pointer',
                transition: 'background 0.2s',
                minHeight: 340,
              }}
              onMouseEnter={e => e.currentTarget.style.background = '#112218'}
              onMouseLeave={e => e.currentTarget.style.background = '#0e1a14'}
            >
              <div className="card-line" style={{ position: 'absolute', inset: 0, borderRadius: 16, pointerEvents: 'none' }} />

              {/* Number + Icon */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                <div style={{ width: 52, height: 52, borderRadius: 16, background: BRAND_DIM, border: `1px solid ${BRAND}33`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Icon size={24} color={BRAND} />
                </div>
                <span style={{ fontSize: 28, fontWeight: 800, color: '#1f4233', letterSpacing: -1 }}>{m.num}</span>
              </div>

              {/* Info */}
              <div style={{ flex: 1 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: BRAND, letterSpacing: 2, display: 'block', marginBottom: 8 }}>{m.tag}</span>
                <h2 style={{ fontSize: 24, fontWeight: 700, color: '#fff', margin: '0 0 10px', letterSpacing: '-0.8px' }}>{m.name}</h2>
                <p style={{ fontSize: 14, color: '#a7b3ad', lineHeight: 1.7, margin: '0 0 20px' }}>{m.blurb}</p>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {m.metrics.map(s => <StatBadge key={s.label} {...s} />)}
                </div>
              </div>

              {/* CTA */}
              <div style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: 16, borderTop: '1px solid #1a2e24' }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: BRAND }}>Launch Model</span>
                <div style={{ width: 36, height: 36, borderRadius: 10, background: BRAND_DIM, border: `1px solid ${BRAND}55`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ChevronRight size={18} color={BRAND} />
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
