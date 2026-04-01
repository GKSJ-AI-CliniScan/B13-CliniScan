import Papa from 'papaparse';
import CryptoJS from 'crypto-js';

/**
 * Hashes a password using SHA-256.
 * @param {string} password 
 * @returns {string} Hashed password hex string
 */
export const hashPassword = (password) => {
  return CryptoJS.SHA256(password).toString();
};

/**
 * Fetches and parses the users.csv file and merges with locally registered users.
 * @returns {Promise<Array>} List of user objects {email, password_hash, username}
 */
export const fetchUsers = async () => {
  try {
    const response = await fetch('/users.csv');
    const csvData = await response.text();
    
    const localUsers = JSON.parse(localStorage.getItem('registered_users') || '[]');
    
    return new Promise((resolve, reject) => {
      Papa.parse(csvData, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const allUsers = [...results.data, ...localUsers];
          resolve(allUsers);
        },
        error: (error) => {
          reject(error);
        }
      });
    });
  } catch (error) {
    console.error('Error fetching users:', error);
    return JSON.parse(localStorage.getItem('registered_users') || '[]');
  }
};

/**
 * Registers a new user and saves to localStorage.
 * @param {string} email
 * @param {string} password 
 * @param {string} username (optional, defaults to part of email)
 * @returns {Promise<Object>} The new user object
 */
export const registerUser = async (email, password, username) => {
  const users = await fetchUsers();
  
  if (users.some(u => u.email === email)) {
    throw new Error('Email already registered');
  }

  const newUser = { 
    email, 
    password_hash: hashPassword(password),
    username: username || email.split('@')[0]
  };
  
  const localUsers = JSON.parse(localStorage.getItem('registered_users') || '[]');
  localUsers.push(newUser);
  localStorage.setItem('registered_users', JSON.stringify(localUsers));
  
  return newUser;
};

/**
 * Validates credentials against the parsed user list.
 * @param {string} email 
 * @param {string} password 
 * @returns {Promise<Object|null>} User object if found, null otherwise
 */
export const validateCredentials = async (email, password) => {
  const users = await fetchUsers();
  const hashedPassword = hashPassword(password);
  
  const user = users.find(u => u.email === email && u.password_hash === hashedPassword);
  return user || null;
};
