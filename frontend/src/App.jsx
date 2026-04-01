import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Auth from './pages/Auth';
import ModelHub from './pages/ModelHub';
import Analysis from './pages/Analysis';
import Dashboard from './pages/Dashboard';

// Wrap any page in a protected route
const Protected = ({ children }) => <ProtectedRoute>{children}</ProtectedRoute>;

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<Auth />} />
          <Route path="/auth" element={<Auth />} />

          {/* After login → Model Hub home */}
          <Route path="/dashboard" element={<Protected><ModelHub /></Protected>} />

          {/* Analysis workspace with mode param */}
          <Route path="/analysis" element={<Protected><Analysis /></Protected>} />

          {/* Default */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
