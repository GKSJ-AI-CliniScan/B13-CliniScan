// 1. Core Variable Setup - Ensure these IDs match your index.html exactly
const fileInput = document.getElementById('file-input');
const resultArea = document.getElementById('result-area');
const previewImg = document.getElementById('preview-img');
const predictionsDiv = document.getElementById('predictions');
const boxOverlay = document.getElementById('box-overlay');

// 2. Button Trigger Function
function triggerUpload() {
    console.log("Upload triggered...");
    fileInput.click();
}

// 3. File Selection & Analysis Handler
async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Reset UI for the new Clinical Light Theme
    previewImg.src = URL.createObjectURL(file);
    resultArea.classList.remove('hidden');
    boxOverlay.innerHTML = ''; 
    
    // Loading State
    predictionsDiv.innerHTML = `
        <div class="p-8 bg-slate-50 border border-slate-200 rounded-2xl text-center shadow-sm">
            <div class="inline-block w-8 h-8 border-4 border-blue-600/20 border-t-blue-600 rounded-full animate-spin mb-4"></div>
            <p class="text-slate-900 font-bold tracking-tight">AI ANALYZING RADIOGRAPH...</p>
            <p class="text-slate-500 text-xs mt-1 uppercase tracking-widest">Dual-Engine Inference in Progress</p>
        </div>`;

    resultArea.scrollIntoView({ behavior: 'smooth', block: 'center' });

    const formData = new FormData();
    formData.append('file', file);

    try {
        // ☁️ CLOUD UPDATE: Changed from 'http://127.0.0.1:8000/analyze' to just '/analyze'
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayResults(data); 
        }
    } catch (error) {
        predictionsDiv.innerHTML = `
            <div class="p-6 bg-rose-50 border border-rose-200 rounded-2xl text-center">
                <p class="text-rose-600 font-bold">⚠️ Connection Error</p>
                <p class="text-rose-500 text-xs">Ensure your Python FastAPI backend is running.</p>
            </div>`;
    }
}

// 4. Result Rendering (B-Boxes & Findings)
function displayResults(data) {
    predictionsDiv.innerHTML = '';
    boxOverlay.innerHTML = ''; 
    
    const chiefVerdict = data.chief_verdict;
    const specialistFindings = data.specialist_findings;

    // A. RESNET BANNER
    let chiefHtml = '';
    if (chiefVerdict.diagnosis === 'ABNORMAL') {
        chiefHtml = `
            <div class="mb-6 p-6 rounded-2xl border border-rose-200 bg-white text-center shadow-sm relative overflow-hidden">
                <div class="absolute top-0 left-0 w-full h-1 bg-rose-500"></div>
                <p class="text-[10px] text-rose-600 font-bold uppercase tracking-[0.2em] mb-2">ResNet-18 Global Verdict</p>
                <p class="text-4xl text-rose-600 font-black tracking-tight">ABNORMAL</p>
                <div class="mt-3 flex justify-center items-center gap-2">
                    <span class="text-xs font-bold text-slate-400 uppercase">Confidence Score</span>
                    <span class="px-2 py-0.5 bg-rose-100 text-rose-700 text-xs font-black rounded">${chiefVerdict.confidence}%</span>
                </div>
            </div>`;
    } else {
        chiefHtml = `
            <div class="mb-6 p-6 rounded-2xl border border-emerald-200 bg-white text-center shadow-sm relative overflow-hidden">
                <div class="absolute top-0 left-0 w-full h-1 bg-emerald-500"></div>
                <p class="text-[10px] text-emerald-600 font-bold uppercase tracking-[0.2em] mb-2">ResNet-18 Global Verdict</p>
                <p class="text-4xl text-emerald-600 font-black tracking-tight">NORMAL</p>
                <div class="mt-3 flex justify-center items-center gap-2">
                    <span class="text-xs font-bold text-slate-400 uppercase">Confidence Score</span>
                    <span class="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-black rounded">${chiefVerdict.confidence}%</span>
                </div>
            </div>`;
    }
    predictionsDiv.innerHTML += chiefHtml;

    // B. SPECIALIST BOXES
    predictionsDiv.innerHTML += `<h4 class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4">Localized Specialist Findings</h4>`;

    if (!specialistFindings || specialistFindings.length === 0) {
        predictionsDiv.innerHTML += `
            <div class="p-4 bg-slate-50 border border-slate-200 rounded-xl text-center">
                <p class="text-slate-500 font-medium text-sm">No localized anomalies detected by YOLOv8.</p>
            </div>`;
    } else {
        const imgWidth = previewImg.naturalWidth;
        const imgHeight = previewImg.naturalHeight;

        specialistFindings.forEach(res => {
            predictionsDiv.innerHTML += `
                <div class="flex justify-between items-center p-4 mb-3 bg-white border border-slate-200 rounded-xl shadow-sm">
                    <div><p class="text-[10px] text-blue-600 font-bold uppercase mb-1">Pathology</p>
                    <p class="text-slate-900 font-extrabold text-base">${res.pathology.toUpperCase()}</p></div>
                    <div class="text-right"><p class="text-[10px] text-slate-400 font-bold mb-1">Confidence</p>
                    <p class="text-blue-600 font-black text-lg">${res.confidence}%</p></div>
                </div>`;

            // Draw Box
            const [x1, y1, x2, y2] = res.bounding_box;
            const leftPct = (x1 / imgWidth) * 100;
            const topPct = (y1 / imgHeight) * 100;
            const widthPct = ((x2 - x1) / imgWidth) * 100;
            const heightPct = ((y2 - y1) / imgHeight) * 100;

            const box = document.createElement('div');
            box.style.position = 'absolute';
            box.style.left = `${leftPct}%`;
            box.style.top = `${topPct}%`;
            box.style.width = `${widthPct}%`;
            box.style.height = `${heightPct}%`;
            box.className = 'border-2 border-rose-500 bg-rose-500/10 shadow-[0_0_10px_rgba(244,63,94,0.3)] transition-all duration-300 pointer-events-auto';
            box.innerHTML = `<div class="absolute -top-6 left-0 bg-rose-600 text-white text-[9px] font-black px-2 py-0.5 rounded uppercase whitespace-nowrap shadow-md">${res.pathology} (${res.confidence}%)</div>`;
            boxOverlay.appendChild(box);
        });
    }

    // C. BUG-FIXED PDF GENERATOR
    const downloadBtn = document.createElement('button');
    downloadBtn.innerHTML = '📄 Download Official Diagnostic Report';
    downloadBtn.className = 'w-full mt-8 bg-slate-900 hover:bg-blue-600 text-white font-bold py-4 px-4 rounded-2xl shadow-lg transition-all transform active:scale-95 text-xs uppercase tracking-widest';
    
    downloadBtn.onclick = () => {
        downloadBtn.innerHTML = '⚙️ PREPARING REPORT...';
        window.scrollTo(0, 0); 
        const element = document.getElementById('result-area');
        const opt = {
            margin: [0.5, 0.5],
            filename: `AI_Report_${new Date().getTime()}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true, letterRendering: true, backgroundColor: '#FFFFFF' },
            jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
        };
        setTimeout(() => {
            html2pdf().set(opt).from(element).save().then(() => {
                downloadBtn.innerHTML = '✓ REPORT DOWNLOADED';
                setTimeout(() => { downloadBtn.innerHTML = '📄 Download Official Diagnostic Report'; }, 3000);
            });
        }, 500);
    };
    predictionsDiv.appendChild(downloadBtn);
}