import React, { useMemo, useState, useEffect } from 'react';
import { 
  Activity, 
  Users, 
  ShieldCheck, 
  Clock, 
  ArrowUpRight, 
  TrendingUp, 
  Search,
  Bell,
  Mail,
  CheckCircle2,
  AlertCircle,
  Microscope,
  Layers
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { getScansForUser } from '../lib/scans';
import { useNavigate } from 'react-router-dom';

const StatsCard = ({ title, value, change, icon: Icon, color, glow }) => (
  <motion.div 
    whileHover={{ y: -4, scale: 1.01 }}
    transition={{ type: 'spring', stiffness: 300 }}
    className="bg-slate-900 p-6 rounded-2xl border border-white/5 flex items-start justify-between relative overflow-hidden group"
  >
    <div className={`absolute -top-6 -right-6 w-24 h-24 rounded-full opacity-10 blur-xl ${color} transition-opacity group-hover:opacity-20`}></div>
    <div>
      <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">{title}</p>
      <h3 className="text-3xl font-extrabold text-white mb-2">{value}</h3>
      <div className="flex items-center gap-1.5">
        <span className="text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-lg text-xs font-bold flex items-center gap-1">
          <TrendingUp size={10} />{change}
        </span>
      </div>
    </div>
    <div className={`p-3 rounded-xl bg-white/5 border border-white/5 ${glow}`}>
      <Icon size={22} className={color.replace('bg-', 'text-')} />
    </div>
  </motion.div>
);

const ActivityItem = ({ icon: Icon, iconColor, title, subtitle, time, status }) => (
  <div className="flex items-center gap-4 p-3 hover:bg-white/3 rounded-xl transition-all cursor-pointer group">
    <div className={`p-2.5 rounded-xl ${iconColor} bg-opacity-10`}>
      <Icon size={16} className={iconColor.replace('bg-', 'text-')} />
    </div>
    <div className="flex-1 min-w-0">
      <h4 className="text-sm font-bold text-slate-200 group-hover:text-cyan-400 transition-colors truncate">{title}</h4>
      <p className="text-xs text-slate-600 font-medium truncate">{subtitle}</p>
    </div>
    <div className="text-right whitespace-nowrap">
      <p className="text-[10px] font-bold text-slate-600 uppercase tracking-tighter">{time}</p>
      <span className={`text-[9px] font-black uppercase px-1.5 py-0.5 rounded mt-0.5 inline-block
        ${status === 'completed' ? 'bg-emerald-400/10 text-emerald-400' : 
          status === 'processing' ? 'bg-cyan-400/10 text-cyan-400' : 
          'bg-amber-400/10 text-amber-400'}`}
      >{status}</span>
    </div>
  </div>
);

const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [userScans, setUserScans] = useState([]);

  useEffect(() => {
    setUserScans(getScansForUser(user?.email));
  }, [user?.email]);

  const stats = [
    { title: 'Total Scans', value: userScans.length.toString(), change: '+12%', icon: Activity, color: 'bg-cyan-400', glow: 'shadow-lg shadow-cyan-500/0 group-hover:shadow-cyan-500/10' },
    { title: 'Completed', value: userScans.filter(s => s.status === 'completed').length.toString(), change: '+5%', icon: ShieldCheck, color: 'bg-emerald-400', glow: '' },
    { title: 'Accuracy', value: '98.2%', change: '+0.4%', icon: TrendingUp, color: 'bg-violet-400', glow: '' },
    { title: 'Pending', value: userScans.filter(s => s.status === 'pending' || s.status === 'processing').length.toString(), change: '-2%', icon: Clock, color: 'bg-amber-400', glow: '' },
  ];

  return (
    <div className="p-6 bg-slate-950 min-h-full text-slate-100" style={{ fontFamily: "'Inter', sans-serif" }}>
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Hello, <span className="text-cyan-400">{user?.username || 'User'}</span> 👋
            </h1>
            <p className="text-slate-500 text-sm mt-1">Here's an overview of your diagnostic activity.</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="p-2.5 bg-slate-900 border border-white/5 rounded-xl text-slate-500 hover:text-slate-300 transition-all">
              <Search size={18} />
            </button>
            <button className="p-2.5 bg-slate-900 border border-white/5 rounded-xl text-slate-500 hover:text-slate-300 transition-all relative">
              <div className="absolute top-2 right-2 w-1.5 h-1.5 bg-red-500 rounded-full"></div>
              <Bell size={18} />
            </button>
            <button 
              onClick={() => navigate('/analysis')}
              className="flex items-center gap-2 px-4 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-900 rounded-xl font-bold text-sm transition-all shadow-lg shadow-cyan-500/20"
            >
              <Microscope size={16} />
              New Analysis
            </button>
            <button 
              onClick={() => navigate('/batch')}
              className="flex items-center gap-2 px-4 py-2.5 bg-slate-900 border border-violet-500/30 hover:bg-violet-500/10 text-violet-400 rounded-xl font-bold text-sm transition-all"
            >
              <Layers size={16} />
              Batch
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {stats.map((stat, idx) => <StatsCard key={idx} {...stat} />)}
        </div>

        {/* Main */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Recent Activity */}
          <div className="lg:col-span-2 bg-slate-900 rounded-2xl border border-white/5 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-base font-extrabold text-white flex items-center gap-2">
                Recent Scans
                <span className="px-2 py-0.5 bg-slate-800 text-slate-500 text-[10px] font-bold rounded-lg uppercase tracking-wider">{userScans.length}</span>
              </h2>
              <button onClick={() => navigate('/analysis')} className="text-xs font-bold text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1 group">
                New Scan <ArrowUpRight size={13} className="group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </button>
            </div>
            <div className="space-y-1">
              {userScans.length > 0 ? (
                userScans.slice(0, 8).map((scan) => (
                  <ActivityItem 
                    key={scan.id}
                    icon={scan.status === 'completed' ? CheckCircle2 : (scan.status === 'pending' ? AlertCircle : Clock)}
                    iconColor={scan.status === 'completed' ? 'bg-emerald-400' : (scan.status === 'pending' ? 'bg-amber-400' : 'bg-cyan-400')}
                    title={scan.title}
                    subtitle={scan.findings?.length > 0 ? scan.findings[0] : 'No significant findings'}
                    time={scan.time}
                    status={scan.status}
                  />
                ))
              ) : (
                <div className="text-center py-12">
                  <Microscope size={32} className="text-slate-800 mx-auto mb-3" />
                  <p className="text-slate-500 font-semibold">No scans yet</p>
                  <p className="text-slate-600 text-sm mt-1">Run your first analysis</p>
                  <button onClick={() => navigate('/analysis')} className="mt-4 px-4 py-2 bg-cyan-500 text-slate-900 rounded-xl text-sm font-bold hover:bg-cyan-400 transition-all">
                    Start Analysis
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Account Card */}
          <div className="bg-slate-900 rounded-2xl border border-white/5 p-6 h-fit">
            <h2 className="text-base font-extrabold text-white mb-6">Account</h2>
            <div className="flex flex-col items-center text-center mb-6">
              <div className="w-16 h-16 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 rounded-2xl flex items-center justify-center mb-3">
                <Users size={24} />
              </div>
              <h3 className="text-base font-extrabold text-white">{user?.username}</h3>
              <p className="text-xs text-slate-500">{user?.email}</p>
            </div>
            <div className="space-y-2.5">
              <div className="p-3 bg-slate-800/50 rounded-xl flex items-center justify-between border border-white/5">
                <div className="flex items-center gap-2.5">
                  <Mail size={15} className="text-slate-500" />
                  <span className="text-xs font-bold text-slate-400">Email Verified</span>
                </div>
                <CheckCircle2 size={15} className="text-emerald-400" />
              </div>
              <div className="p-3 bg-slate-800/50 rounded-xl flex items-center justify-between border border-white/5">
                <div className="flex items-center gap-2.5">
                  <ShieldCheck size={15} className="text-slate-500" />
                  <span className="text-xs font-bold text-slate-400">Password Security</span>
                </div>
                <span className="px-2 py-0.5 bg-cyan-500/10 text-cyan-400 text-[9px] font-black uppercase rounded-lg">SHA-256</span>
              </div>

              {/* Quick Actions */}
              <div className="pt-2 space-y-2">
                <button onClick={() => navigate('/analysis')} className="w-full flex items-center gap-2 px-3 py-2.5 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 rounded-xl text-xs font-bold hover:bg-cyan-500/20 transition-all">
                  <Microscope size={13} /> Run Analysis
                </button>
                <button onClick={() => navigate('/batch')} className="w-full flex items-center gap-2 px-3 py-2.5 bg-violet-500/10 border border-violet-500/20 text-violet-400 rounded-xl text-xs font-bold hover:bg-violet-500/20 transition-all">
                  <Layers size={13} /> Batch Process
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
