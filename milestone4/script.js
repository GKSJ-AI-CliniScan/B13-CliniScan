document.addEventListener('DOMContentLoaded', () => {

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const displayFilename = document.getElementById('displayFilename');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');

    const emptyResults = document.getElementById('emptyResults');
    const loadingResults = document.getElementById('loadingResults');
    const findingsContent = document.getElementById('findingsContent');
    const progressFill = document.getElementById('progressFill');
    
    const scannerOverlay = document.getElementById('scannerOverlay');
    const mockXray = document.getElementById('mockXray');
    const findingList = document.getElementById('findingList');
    const statusBadge = document.getElementById('statusBadge');
    const exportReportBtn = document.getElementById('exportReportBtn');

    // Disease configurations
    const detectedObjects = [
        { class: 'Cardiomegaly', conf: '96.2%', color: '#ef4444', x: 25, y: 55, w: 50, h: 40 },
        { class: 'Aortic enlargement', conf: '84.5%', color: '#3b82f6', x: 45, y: 30, w: 25, h: 25 },
        { class: 'Pleural effusion', conf: '71.0%', color: '#f59e0b', x: 65, y: 70, w: 15, h: 20 }
    ];

    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', async () => {
            if (findingList.children.length === 0) {
                alert("No analysis data available to export.");
                return;
            }

            // Retrieve the logged-in user
            const currentUser = JSON.parse(localStorage.getItem('cliniscanCurrentUser')) || { name: 'Unknown User' };

            // Initialize jsPDF
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();

            // Document Styling settings
            const marginX = 20;
            let currentY = 20;

            // Report Header
            doc.setFontSize(18);
            doc.setTextColor(11, 94, 221); // Primary blue
            doc.setFont("helvetica", "bold");
            doc.text("CLINISCAN AI DIAGNOSTIC REPORT", marginX, currentY);
            
            currentY += 15;
            doc.setDrawColor(200, 200, 200);
            doc.line(marginX, currentY, 190, currentY);
            currentY += 10;

            // Metadata
            doc.setFontSize(11);
            doc.setTextColor(50, 50, 50);
            doc.setFont("helvetica", "normal");
            
            doc.text(`Radiologist / User: ${currentUser.name}`, marginX, currentY);
            currentY += 8;
            doc.text(`Patient/File: ${displayFilename.innerText}`, marginX, currentY);
            currentY += 8;
            doc.text(`Date of Analysis: ${new Date().toLocaleString()}`, marginX, currentY);
            
            currentY += 10;
            doc.line(marginX, currentY, 190, currentY);
            currentY += 15;

            // Findings Section Header
            doc.setFontSize(14);
            doc.setTextColor(0, 0, 0);
            doc.setFont("helvetica", "bold");
            doc.text("AI DIAGNOSTIC FINDINGS:", marginX, currentY);
            currentY += 10;

            // List Findings
            doc.setFontSize(11);
            doc.setFont("helvetica", "normal");
            const items = findingList.querySelectorAll('.finding-item');
            
            items.forEach(item => {
                const nameNode = item.querySelector('.finding-name');
                const confNode = item.querySelector('.finding-conf');
                
                if (nameNode && nameNode.innerText.includes('No Abnormalities')) {
                    doc.setTextColor(16, 185, 129); // Success Green
                    doc.text("• No Abnormalities Detected (Normal)", marginX + 5, currentY);
                } else if (nameNode && confNode) {
                    // Extract confidence to determine color
                    let confStr = confNode.innerText;
                    let confVal = parseFloat(confStr);
                    
                    if (confVal >= 80) doc.setTextColor(239, 68, 68); // Red
                    else if (confVal >= 30) doc.setTextColor(245, 158, 11); // Orange
                    else doc.setTextColor(59, 130, 246); // Blue

                    doc.text(`• ${nameNode.innerText}`, marginX + 5, currentY);
                    
                    doc.setTextColor(100, 100, 100);
                    doc.text(`(Confidence: ${confStr})`, marginX + 80, currentY);
                }
                currentY += 8;
            });

            currentY += 15;
            doc.setDrawColor(200, 200, 200);
            doc.line(marginX, currentY, 190, currentY);
            
            // Footer
            currentY += 10;
            doc.setFontSize(9);
            doc.setTextColor(150, 150, 150);
            doc.setFont("helvetica", "italic");
            doc.text("Generated automatically by the CliniScan AI Neural Network System.", marginX, currentY);
            doc.text("This report is for informational purposes and should not replace professional medical judgment.", marginX, currentY + 5);

            // Capture analyzed image
            try {
                const canvas = await html2canvas(document.getElementById('mockXray'), {
                    backgroundColor: '#000000',
                    useCORS: true,
                    scale: 2
                });
                const imgData = canvas.toDataURL('image/jpeg', 0.9);
                
                doc.addPage();
                doc.setFontSize(14);
                doc.setTextColor(0, 0, 0);
                doc.setFont("helvetica", "bold");
                doc.text("ANALYZED X-RAY WITH DETECTIONS:", marginX, 20);
                
                const imgWidth = 170;
                const imgHeight = (canvas.height * imgWidth) / canvas.width;
                doc.addImage(imgData, 'JPEG', marginX, 30, imgWidth, imgHeight);
            } catch (err) {
                console.error("Could not capture analyzed image:", err);
            }

            // Save PDF
            doc.save(`CliniScan_Report_${displayFilename.innerText}.pdf`);
        });
    }

    // File Drag & Drop Logic
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        handleFile(file);
    }, false);

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    let selectedFile = null;

    function handleFile(file) {
        if (!file) return;
        selectedFile = file;

        // X-Ray File Validation
        const validExtensions = ['.png', '.jpg', '.jpeg', '.dcm'];
        const isValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        const isImage = file.type && file.type.startsWith('image/');

        if (!isValidExtension && !isImage) {
            alert("⚠️ SYSTEM ERROR: Invalid file type detected.\n\nPlease upload a valid chest X-ray image format (PNG, JPG, JPEG) or a standard DICOM (.dcm) scan.");
            return;
        }

        displayFilename.innerText = file.name;
        
        // Read the actual uploaded file and set it as the background image
        if (isImage) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const imgUrl = event.target.result;
                
                // --- X-RAY HEURISTIC VALIDATION (Grayscale Check) ---
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    // Downscale for performance
                    canvas.width = Math.min(img.width, 500);
                    canvas.height = Math.min(img.height, 500);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    
                    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                    let colorDiffSum = 0;
                    let sampled = 0;
                    
                    // Sample pixels to verify it is grayscale (R≈G≈B) which is standard for X-Rays
                    for (let i = 0; i < imgData.length; i += 16) { // step by 4 pixels
                        let r = imgData[i];
                        let g = imgData[i+1];
                        let b = imgData[i+2];
                        let maxDiff = Math.max(Math.abs(r-g), Math.abs(r-b), Math.abs(g-b));
                        colorDiffSum += maxDiff;
                        sampled++;
                    }
                    
                    let avgDiff = colorDiffSum / sampled;
                    
                    // If average color variance is high, it's a regular photograph (man, dog, etc)
                    if (avgDiff > 12) { 
                        alert("❌ AI REJECTED: NON-MEDICAL IMAGE DETECTED\n\nThis appears to be a standard color photograph (like a person or animal). CliniScan strictly requires grayscale Medical X-Ray scans for analysis.");
                        clearBtn.click();
                        return;
                    }
                    
                    // Image passed X-ray validation
                    mockXray.style.backgroundImage = `url(${imgUrl})`;
                };
                img.src = imgUrl;
                // ----------------------------------------------------
                
            };
            reader.readAsDataURL(file);
        }
        
        // UI State transition
        dropZone.classList.add('collapsed');
        previewArea.classList.remove('collapsed');
        
        // Reset Results State
        emptyResults.classList.remove('collapsed');
        loadingResults.classList.add('collapsed');
        findingsContent.classList.add('collapsed');
        statusBadge.innerText = 'Ready for Analysis';
        statusBadge.className = 'badge badge-neutral';

        // Clear previous bounding boxes
        mockXray.innerHTML = '';
        findingList.innerHTML = '';
    }

    // Clear Button
    clearBtn.addEventListener('click', () => {
        dropZone.classList.remove('collapsed');
        previewArea.classList.add('collapsed');
        emptyResults.classList.remove('collapsed');
        findingsContent.classList.add('collapsed');
        statusBadge.innerText = 'Awaiting Upload';
        statusBadge.className = 'badge badge-neutral';
        mockXray.innerHTML = '';
        mockXray.style.backgroundImage = ''; // Remove the current loaded image
        selectedFile = null;
    });

    // Run Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Loading Transition
        emptyResults.classList.add('collapsed');
        loadingResults.classList.remove('collapsed');
        scannerOverlay.classList.add('active');
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing DenseNet121...';
        statusBadge.innerText = 'Analyzing';
        statusBadge.className = 'badge badge-neutral';
        mockXray.innerHTML = '';
        findingList.innerHTML = '';

        let progress = 0;
        progressFill.style.width = '0%';
        const interval = setInterval(() => { 
            progress = Math.min(progress + 15, 90); 
            progressFill.style.width = `${progress}%`; 
        }, 300);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            // Send image directly to PyTorch DenseNet121 backend
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData
            });

            clearInterval(interval);
            progressFill.style.width = '100%';

            const data = await response.json();
            if(data.error || !data.success) {
                alert("DensNet API Error: " + (data.error || "Unknown"));
                finishAnalysis({predictions: [], boxes: []});
            } else {
                finishAnalysis(data);
            }
        } catch (error) {
            clearInterval(interval);
            alert("Backend Connection Error: Make sure your Python Flask server 'app.py' is running on port 5000!");
            finishAnalysis({predictions: [], boxes: []});
        }
    });

    function finishAnalysis(data) {
        const predictions = data.predictions || [];
        const boxes = data.boxes || [];

        // Transition UI State
        loadingResults.classList.add('collapsed');
        findingsContent.classList.remove('collapsed');
        scannerOverlay.classList.remove('active');
        
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Re-scan Image';
        
        statusBadge.innerText = `Completed: ${predictions.length} Findings`;
        statusBadge.className = predictions.length > 0 ? 'badge badge-danger' : 'badge badge-success';

        if (predictions.length === 0) {
            findingList.innerHTML = '<li class="finding-item" style="--color: #10b981"><span class="finding-name text-success"><i class="fa-solid fa-check"></i> No Abnormalities Detected (Normal)</span></li>';
        }

        predictions.forEach((obj, index) => {
            setTimeout(() => {
                let color = '#ef4444'; // Critial red
                let probVal = parseFloat(obj.confidence);
                
                if (probVal < 80.0) color = '#f59e0b'; // Warning orange
                if (probVal < 30.0) color = '#3b82f6'; // Info blue

                const li = document.createElement('li');
                li.className = 'finding-item';
                li.style.setProperty('--color', color);
                li.innerHTML = `
                    <span class="finding-name">${obj.class}</span>
                    <span class="finding-conf" style="color: ${color}">${obj.confidence}</span>
                `;
                findingList.appendChild(li);
            }, index * 200); 
        });

        // Overlay YOLOv8 Bounding Boxes
        boxes.forEach((box, index) => {
            setTimeout(() => {
                let color = '#ef4444'; 
                let confVal = parseFloat(box.conf);
                if (confVal < 80.0) color = '#f59e0b'; 
                if (confVal < 30.0) color = '#3b82f6';

                const bbox = document.createElement('div');
                bbox.className = 'ai-bbox';
                bbox.style.left = `${box.x}%`;
                bbox.style.top = `${box.y}%`;
                bbox.style.width = `${box.w}%`;
                bbox.style.height = `${box.h}%`;
                bbox.style.borderColor = color;
                bbox.style.color = color;

                const label = document.createElement('div');
                label.className = 'bbox-label';
                label.innerText = `${box.class} ${box.conf}`;
                
                bbox.appendChild(label);
                mockXray.appendChild(bbox);
            }, index * 200 + 400); // Delay slightly for cool visual pacing
        });

        // Update Dashboard Analytics in LocalStorage
        try {
            let dashData = JSON.parse(localStorage.getItem('cliniscanDashboard'));
            if (!dashData) {
                dashData = {
                    totalScans: 200,
                    anomalies: 3421,
                    diseaseStats: {
                        'Cardiomegaly': 856, 'Aortic enlargement': 543, 'Pleural effusion': 610,
                        'Lung Opacity': 920, 'Nodule/Mass': 345, 'Infiltration': 430, 'Consolidation': 320
                    },
                    recentCases: [
                        { id: '#P-1049', name: 'John Doe', priority: 'Critical', status: 'Review' },
                        { id: '#P-1052', name: 'Jane Smith', priority: 'Critical', status: 'Review' },
                        { id: '#P-1065', name: 'Robert Johnson', priority: 'Critical', status: 'Review' },
                        { id: '#P-1088', name: 'Michael Brown', priority: 'Critical', status: 'Review' }
                    ]
                };
            }
            if (!dashData.recentCases) {
                dashData.recentCases = [
                    { id: '#P-1049', name: 'John Doe', priority: 'Critical', status: 'Review' },
                    { id: '#P-1052', name: 'Jane Smith', priority: 'Critical', status: 'Review' },
                    { id: '#P-1065', name: 'Robert Johnson', priority: 'Critical', status: 'Review' },
                    { id: '#P-1088', name: 'Michael Brown', priority: 'Critical', status: 'Review' }
                ];
            }
            
            dashData.totalScans += 1;
            if (predictions.length > 0) {
                dashData.anomalies += predictions.length;
                let hasCritical = false;
                predictions.forEach(p => {
                    const cls = p.class;
                    if (dashData.diseaseStats[cls]) dashData.diseaseStats[cls]++;
                    else dashData.diseaseStats[cls] = 1;
                    if (parseFloat(p.confidence) >= 80.0) hasCritical = true;
                });

                const sysUser = JSON.parse(localStorage.getItem('cliniscanCurrentUser'));
                const newCase = {
                    id: '#P-' + Math.floor(Math.random() * 9000 + 1000),
                    name: (sysUser && sysUser.name) ? sysUser.name : 'Unknown Patient',
                    priority: hasCritical ? 'Critical' : 'Warning',
                    status: 'Review'
                };
                dashData.recentCases.unshift(newCase);
                if (dashData.recentCases.length > 4) dashData.recentCases.pop();
            }
            localStorage.setItem('cliniscanDashboard', JSON.stringify(dashData));
        } catch(e) { console.error("Could not save dashboard stats", e); }
    }

});
