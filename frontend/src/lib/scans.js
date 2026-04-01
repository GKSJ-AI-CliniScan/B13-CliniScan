/**
 * Static baseline mock scans, mapped by user email.
 */
const STATIC_SCANS = [
  { id: 's1', userEmail: 'user1@example.com', title: 'Chest X-Ray Analysis', time: '2 days ago', status: 'completed', findings: ['Cardiomegaly: 78%', 'Infiltration: 45%'] },
  { id: 's2', userEmail: 'user1@example.com', title: 'Minor Anomaly Detected', time: '3 days ago', status: 'pending', findings: ['Nodule: 62%'] },
  { id: 's3', userEmail: 'user2@example.com', title: 'Abdominal CT Scan', time: '15 mins ago', status: 'completed', findings: ['Effusion: 83%'] },
  { id: 's4', userEmail: 'user2@example.com', title: 'MRI Processing', time: '3 hours ago', status: 'processing', findings: [] },
  { id: 's5', userEmail: 'admin@cliniscan.com', title: 'System-wide Diagnostics', time: 'Recently', status: 'completed', findings: [] },
  { id: 's6', userEmail: 'admin@cliniscan.com', title: 'Database Integrity Check', time: '5 mins ago', status: 'completed', findings: [] },
];

/**
 * Fetches scan history for a specific user, merging static + dynamic scans from localStorage.
 * @param {string} userEmail
 * @returns {Array}
 */
export const getScansForUser = (userEmail) => {
  if (!userEmail) return [];
  const staticScans = STATIC_SCANS.filter(s => s.userEmail === userEmail);
  const storageKey = `scans_${userEmail}`;
  const dynamicScans = JSON.parse(localStorage.getItem(storageKey) || '[]');
  return [...dynamicScans.reverse(), ...staticScans]; // newest first
};

/**
 * Saves a new scan result for a user to localStorage.
 * @param {string} userEmail
 * @param {{ title: string, status: string, findings: string[] }} scanData
 */
export const saveScanForUser = (userEmail, scanData) => {
  const storageKey = `scans_${userEmail}`;
  const existing = JSON.parse(localStorage.getItem(storageKey) || '[]');
  const newScan = {
    id: `d${Date.now()}`,
    userEmail,
    time: 'Just now',
    ...scanData,
  };
  existing.push(newScan);
  localStorage.setItem(storageKey, JSON.stringify(existing));
  return newScan;
};
