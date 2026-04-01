import React from 'react';
import { LogOut, User, LayoutDashboard, Settings } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Navbar = () => {
  const { user, logout } = useAuth();

  return (
    <nav className="glass sticky top-0 z-50 px-6 py-4 flex items-center justify-between border-b border-slate-200/60 shadow-sm">
      <div className="flex items-center gap-2">
        <div className="bg-brand-600 p-2 rounded-lg text-white">
          <LayoutDashboard size={24} />
        </div>
        <span className="text-xl font-bold bg-gradient-to-r from-brand-600 to-brand-800 bg-clip-text text-transparent">
          CliniScan Admin
        </span>
      </div>

      {user && (
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-2 text-slate-600 font-medium">
            <User size={18} className="text-brand-500" />
            <span>{user.username}</span>
          </div>
          
          <button 
            onClick={logout}
            className="flex items-center gap-2 px-4 py-2 text-slate-700 hover:text-red-600 hover:bg-red-50 rounded-full transition-all duration-200 border border-transparent hover:border-red-100"
          >
            <LogOut size={18} />
            <span className="hidden sm:inline font-medium">Logout</span>
          </button>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
