document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const actionButtons = document.getElementById('action-buttons');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    
    const appLayout = document.querySelector('.app-layout');
    const resultSection = document.getElementById('result-section');
    const primaryClass = document.getElementById('primary-class');
    const primaryConfidence = document.getElementById('primary-confidence');
    const primaryProgress = document.getElementById('primary-progress');
    const lowConfidenceWarning = document.getElementById('low-confidence-warning');
    const topPredictionsList = document.getElementById('top-predictions-list');
    
    let currentFile = null;

    // --- Formatters ---
    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const capitalize = (str) => {
        return str.charAt(0).toUpperCase() + str.slice(1);
    };

    // --- File Handling ---
    const handleFile = (file) => {
        if (!file) return;
        
        // Basic validation
        if (!file.type.match('image/(jpeg|png|webp)')) {
            alert('Please upload a JPG, PNG, or WEBP image.');
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            alert('File size exceeds 5MB limit.');
            return;
        }

        currentFile = file;
        
        // Update UI
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZoneContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            actionButtons.classList.remove('hidden');
            
            // Reset result
            hideResult();
        };
        reader.readAsDataURL(file);
    };

    const resetUpload = () => {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropZoneContent.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        actionButtons.classList.add('hidden');
        hideResult();
    };

    // --- Event Listeners ---
    browseBtn.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    clearBtn.addEventListener('click', resetUpload);

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.add('dragover'));
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('dragover'));
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    // --- Prediction API ---
    const showResult = (data) => {
        appLayout.classList.add('has-result');
        resultSection.classList.remove('hidden');
        resultSection.classList.add('fade-in');
        
        primaryClass.textContent = capitalize(data.predicted_class);
        primaryConfidence.textContent = data.confidence.toFixed(2);
        
        // Small timeout for CSS transition to trigger
        setTimeout(() => {
            primaryProgress.style.width = data.confidence + '%';
        }, 50);

        if (data.confidence < 50) {
            lowConfidenceWarning.classList.remove('hidden');
        } else {
            lowConfidenceWarning.classList.add('hidden');
        }

        // Top predictions
        topPredictionsList.innerHTML = '';
        data.top_predictions.forEach((pred, index) => {
            const li = document.createElement('li');
            li.className = 'prediction-row';
            li.innerHTML = `
                <div class="pred-info">
                    <div>
                        <span class="pred-rank">0${index + 1}</span>
                        <span class="pred-class">${capitalize(pred.class)}</span>
                    </div>
                    <span class="pred-conf">${pred.confidence.toFixed(2)}%</span>
                </div>
                <div class="pred-bar-bg">
                    <div class="pred-bar" style="width: ${pred.confidence}%"></div>
                </div>
            `;
            topPredictionsList.appendChild(li);
        });
    };

    const hideResult = () => {
        appLayout.classList.remove('has-result');
        resultSection.classList.add('hidden');
        resultSection.classList.remove('fade-in');
        primaryProgress.style.width = '0%';
        
        // Remove error message if exists
        const oldError = document.querySelector('.error-msg');
        if (oldError) oldError.remove();
    };

    const showError = (msg) => {
        const errDiv = document.createElement('div');
        errDiv.className = 'error-msg fade-in';
        errDiv.textContent = msg;
        actionButtons.parentElement.appendChild(errDiv);
    };

    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Loading state
        predictBtn.disabled = true;
        const originalContent = predictBtn.innerHTML;
        predictBtn.innerHTML = '<div class="spinner"></div> Analyzing...';
        hideResult();

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            // Check content type to see if we got JSON
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.indexOf("application/json") !== -1) {
                const data = await response.json();
                if (response.ok && data.success) {
                    showResult(data);
                } else {
                    showError(data.error || 'Prediction failed.');
                }
            } else {
                // We got HTML or something else (e.g. 502 Bad Gateway)
                const text = await response.text();
                console.error('Non-JSON response:', text);
                showError(`Server error (${response.status} ${response.statusText}). Please check server logs.`);
            }
        } catch (error) {
            console.error('API Error:', error);
            showError('Unable to connect to the prediction server. Please try again.');
        } finally {
            // Restore button
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalContent;
        }
    });
});
