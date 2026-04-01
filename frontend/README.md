# MediScanAI: Clinical Diagnostic Platform (Frontend)

MediScanAI is a highly optimized, React-based clinical interface designed to augment radiological workflows. Built with Vite and Tailwind CSS, it seamlessly integrates with our custom PyTorch backend to deliver real-time X-Ray classification and abnormality localization.

## 🚀 Live Production
**Access the live platform:** [https://mediscanai-tawny.vercel.app](https://mediscanai-tawny.vercel.app)

*(The backend API is hosted independently on Hugging Face Spaces.)*

## ✨ Core Features
- **Dual-Mode Analysis**: Seamlessly switch between global pathology **Classification** and localized **Detection**.
- **Dynamic Bounding Boxes**: Resolution-independent SVG overlays map Faster R-CNN coordinates perfectly onto the uploaded X-Ray.
- **Explainable AI (Heatmap)**: Visual evidence of CNN activation patterns for clinical interpretability.
- **Secure Authentication**: Built-in SHA-256 password hashing and role-based access control.
- **Patient History Dashboard**: Local-persisted scan database tracking historical diagnostic results.
- **Clinical Print Reports**: Specialized `@media print` CSS engine generates structured, scaled PDF reports containing pathology scores, bounding box visuals, and actionable medical recommendations.

## 🛠 Tech Stack
- **Framework**: React 18, Vite
- **Styling**: Tailwind CSS, Framer Motion (Animations)
- **Security**: Crypto-js (SHA-256 Hashing)
- **Deployment**: Vercel Edge Network

## ⚙️ Running Locally
1. Install dependencies: `npm install`
2. Create a `.env` file mapping the API: `VITE_API_URL=http://localhost:7860`
3. Start the dev server: `npm run dev`
