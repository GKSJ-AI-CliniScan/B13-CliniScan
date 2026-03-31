import React, { useRef } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Challenges from './components/Challenges';
import Solution from './components/Solution';
import Capabilities from './components/Capabilities';
import Abnormalities from './components/Abnormalities';
import UploadSection from './components/UploadSection';
import ResultsPanel from './components/ResultsPanel';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import HistoryTimeline from './components/HistoryTimeline';
import Chatbot from './components/Chatbot';
import Auth from './components/Auth';
import { useAI } from './hooks/useAI';
import { useLanguage } from './context/LanguageContext';

const App: React.FC = () => {
  const { analyzeImage, isAnalyzing, result } = useAI();
  const { t } = useLanguage();
  const [user, setUser] = React.useState<{ name: string, role: string } | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const handleUpload = async (file: File) => {
    await analyzeImage(file);
    // Smooth scroll to results after analysis
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  if (!user) {
    return <Auth onLogin={setUser} />;
  }

  return (
    <div className="min-h-screen bg-white dark:bg-slate-950 font-sans selection:bg-medical-blue/20 selection:text-medical-blue">
      <Navbar />
      
      <main>
        <Hero title={t.hero_title} subtitle={t.hero_subtitle} />
        <Challenges title={t.challenges_title} />
        <Solution title={t.solution_title} />
        <Capabilities title={t.capabilities_title} />
        <Abnormalities title={t.abnormalities_title} />
        
        <UploadSection 
          onUpload={handleUpload} 
          isAnalyzing={isAnalyzing} 
          title={t.upload_title}
          subtitle={t.upload_subtitle}
        />

        <div ref={resultsRef}>
          {result && <ResultsPanel result={result} />}
        </div>

        <AnalyticsDashboard />
        <HistoryTimeline />
      </main>

      <Chatbot />



      <footer className="py-12 border-t dark:border-slate-800 bg-slate-50 dark:bg-slate-900/20">
        <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center space-x-2">
            <div className="p-1.5 bg-medical-blue rounded-lg text-white">
              <Activity size={16} />
            </div>
            <span className="text-xl font-bold tracking-tight text-medical-dark dark:text-medical-light">
              Clini<span className="text-medical-blue">Scan</span>
            </span>
          </div>
          <p className="text-slate-500 dark:text-slate-400 text-sm">
            © 2026 CliniScan AI. All Rights Reserved by Supriya
          </p>
          <div className="flex space-x-6 text-sm font-medium text-slate-500 dark:text-slate-400">
             <a href="#" className="hover:text-medical-blue transition-colors">Privacy Policy</a>
             <a href="#" className="hover:text-medical-blue transition-colors">Terms of Service</a>
          </div>
        </div>
      </footer>
    </div>
  );
};


const Activity: React.FC<{ size?: number, className?: string }> = ({ size = 24, className = "" }) => (
  <svg 
    width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}
  >
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
  </svg>
);

export default App;
