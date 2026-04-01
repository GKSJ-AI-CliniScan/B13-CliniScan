import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Lock, User, AlertCircle, Loader2, Eye, EyeOff, UserPlus, LogIn, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { validateCredentials, registerUser } from '../lib/auth';

const Auth = () => {
  const [activeTab, setActiveTab] = useState('login'); // 'login' or 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  
  const BRAND = '#34d399';
  const DARK_BG = '#0a0f0e';
  const CARD_BG = '#0e1a14';
  const BORDER = '#16271e';
  const INPUT_BG = '#1a2e24';
  
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || '/dashboard';

  const resetForm = () => {
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setError('');
    setSuccess('');
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setIsLoading(true);

    try {
      if (activeTab === 'login') {
        const user = await validateCredentials(email, password);
        if (user) {
          login(user, rememberMe);
          navigate(from, { replace: true });
        } else {
          setError('Invalid email or password. Please try again.');
        }
      } else {
        // Sign Up Logic
        if (password !== confirmPassword) {
          throw new Error('Passwords do not match');
        }
        if (password.length < 6) {
          throw new Error('Password must be at least 6 characters');
        }
        
        await registerUser(email, password);
        setSuccess('Registration successful! You can now sign in.');
        setTimeout(() => {
          setActiveTab('login');
          setPassword('');
          setConfirmPassword('');
        }, 1500);
      }
    } catch (err) {
      setError(err.message || 'An error occurred during authentication.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center p-6 min-h-screen" style={{ background: DARK_BG, color: '#e2e8f0', fontFamily: "'Space Grotesk', sans-serif" }}>
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-4" style={{ background: 'linear-gradient(135deg, #34d399 0%, #10b981 100%)', boxShadow: '0 0 20px rgba(52, 211, 153, 0.3)' }}>
             <Lock size={28} color="#064e3b" />
          </div>
          <h1 className="text-4xl font-black tracking-tight mb-2 text-white">
            {activeTab === 'login' ? 'MediScanAI' : 'Create Account'}
          </h1>
          <p style={{ color: '#4b7a62', fontWeight: 500 }}>
            {activeTab === 'login' 
              ? 'Institutional Medical Grade Diagnostics' 
              : 'Join the next generation of clinical analysis'}
          </p>
        </div>

        <div className="p-1 rounded-2xl mb-6 flex gap-1 z-10 relative" style={{ background: '#091410', border: `1px solid ${BORDER}` }}>
          <button
            onClick={() => { setActiveTab('login'); resetForm(); }}
            className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-bold transition-all duration-300 ${
              activeTab === 'login' ? 'bg-emerald-500 text-emerald-950 shadow-lg' : 'text-slate-500 hover:text-slate-300'
            }`}
            style={activeTab === 'login' ? { background: BRAND } : {}}
          >
            <LogIn size={18} />
            Login
          </button>
          <button
            onClick={() => { setActiveTab('signup'); resetForm(); }}
            className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-bold transition-all duration-300 ${
              activeTab === 'signup' ? 'bg-emerald-500 text-emerald-950 shadow-lg' : 'text-slate-500 hover:text-slate-300'
            }`}
            style={activeTab === 'signup' ? { background: BRAND } : {}}
          >
            <UserPlus size={18} />
            Sign Up
          </button>
        </div>

        <div className="p-8 rounded-3xl shadow-2xl relative overflow-hidden" style={{ background: CARD_BG, border: `1px solid ${BORDER}` }}>
          {/* Subtle glow effect */}
          <div style={{ position: 'absolute', top: -100, right: -100, width: 200, height: 200, background: BRAND, filter: 'blur(100px)', opacity: 0.05, pointerEvents: 'none' }} />

          <form onSubmit={handleAuth} className="space-y-6">
            <AnimatePresence mode="wait">
              {error && (
                <motion.div 
                  key="error"
                  initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 10 }}
                  className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-xl flex items-center gap-3 text-sm"
                >
                  <AlertCircle size={18} className="shrink-0" />
                  <p>{error}</p>
                </motion.div>
              )}
              {success && (
                <motion.div 
                  key="success"
                  initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 10 }}
                  className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl flex items-center gap-3 text-sm"
                >
                  <CheckCircle2 size={18} className="shrink-0" />
                  <p>{success}</p>
                </motion.div>
              )}
            </AnimatePresence>

            <div>
              <label className="block text-xs font-black uppercase tracking-widest mb-2 ml-1" style={{ color: '#4b7a62' }}>Email Address</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-500 group-focus-within:text-emerald-400 transition-colors">
                  <User size={18} />
                </div>
                <input
                  type="email" required value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-12 pr-4 py-4 rounded-xl focus:outline-none transition-all placeholder:text-slate-600 text-white"
                  style={{ background: INPUT_BG, border: `1px solid ${BORDER}` }}
                  placeholder="name@hospital.org"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-black uppercase tracking-widest mb-2 ml-1" style={{ color: '#4b7a62' }}>Password</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-500 group-focus-within:text-emerald-400 transition-colors">
                  <Lock size={18} />
                </div>
                <input
                  type={showPassword ? 'text' : 'password'} required value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-12 pr-12 py-4 rounded-xl focus:outline-none transition-all placeholder:text-slate-600 text-white"
                  style={{ background: INPUT_BG, border: `1px solid ${BORDER}` }}
                  placeholder="••••••••"
                />
                <button
                  type="button" onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-4 flex items-center text-slate-500 hover:text-emerald-400 transition-colors"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {activeTab === 'signup' && (
              <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                <label className="block text-xs font-black uppercase tracking-widest mb-2 ml-1" style={{ color: '#4b7a62' }}>Verify Password</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-500 group-focus-within:text-emerald-400 transition-colors">
                    <Lock size={18} />
                  </div>
                  <input
                    type={showPassword ? 'text' : 'password'} required value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full pl-12 pr-4 py-4 rounded-xl focus:outline-none transition-all placeholder:text-slate-600 text-white"
                    style={{ background: INPUT_BG, border: `1px solid ${BORDER}` }}
                    placeholder="••••••••"
                  />
                </div>
              </motion.div>
            )}

            <div className="flex items-center justify-between">
              {activeTab === 'login' ? (
                <>
                  <label className="flex items-center gap-2 cursor-pointer group">
                    <input
                      type="checkbox" checked={rememberMe}
                      onChange={(e) => setRememberMe(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900 cursor-pointer"
                    />
                    <span className="text-sm text-slate-500 group-hover:text-slate-300 transition-colors">Persistent session</span>
                  </label>
                  <a href="#" className="text-sm font-bold text-emerald-400 hover:text-emerald-300 transition-colors">
                    Recovery
                  </a>
                </>
              ) : (
                <div className="text-[10px] text-slate-600 uppercase tracking-tighter font-bold">
                  Institutional credentialing enabled
                </div>
              )}
            </div>

            <button
              type="submit" disabled={isLoading}
              className="w-full py-4 px-4 text-emerald-950 rounded-xl font-black uppercase tracking-widest shadow-xl active:scale-[0.98] transition-all disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2 group"
              style={{ background: BRAND, boxShadow: `0 8px 20px ${BRAND}33` }}
            >
              {isLoading ? (
                <Loader2 className="animate-spin" size={20} />
              ) : (
                <>
                  <span>{activeTab === 'login' ? 'Authenticate' : 'Register'}</span>
                  <LogIn size={18} className="group-hover:translate-x-1 transition-transform" />
                </>
              )}
            </button>
          </form>
        </div>

        <div className="mt-8 text-center">
            <p className="text-[10px] text-slate-700 font-bold uppercase tracking-[0.2em]">
                MediScanAI v2.0.4 · Clinical Environment
            </p>
        </div>
      </motion.div>
    </div>
  );
};

export default Auth;
