const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const processingStatus = document.getElementById('processing-status');
const detectionCounter = document.getElementById('detection-counter');
const hazardsCount = document.getElementById('hazards-count');
const activityBody = document.getElementById('activity-body');

let currentJobId = null;

// Handle click on dropzone
dropzone.addEventListener('click', () => fileInput.click());

// Handle file selection
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
    }
});

// Handle drag and drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#0076FF';
    dropzone.style.backgroundColor = '#F1F5F9';
});

dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = '#E2E8F0';
    dropzone.style.backgroundColor = '#F8FAFC';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#E2E8F0';
    dropzone.style.backgroundColor = '#F8FAFC';
    if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
    }
});

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    processingStatus.classList.remove('hidden');
    dropzone.classList.add('hidden');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        currentJobId = data.job_id;
        pollStatus();
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Upload failed. Please try again.');
        resetUI();
    }
}

async function pollStatus() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`/status/${currentJobId}`);
        const data = await response.json();

        detectionCounter.textContent = data.detections;
        hazardsCount.textContent = 8 + data.detections; // Base from mockup + new

        if (data.status === 'processing') {
            setTimeout(pollStatus, 2000);
        } else if (data.status === 'completed') {
            showResults();
        } else if (data.status === 'error') {
            alert('Processing error happened. Check logs.');
            resetUI();
        }
    } catch (error) {
        console.error('Status poll failed:', error);
        setTimeout(pollStatus, 5000);
    }
}

async function showResults() {
    try {
        const response = await fetch(`/results/${currentJobId}`);
        const data = await response.json();

        // Add a new row to the top of the activity table
        const newRow = document.createElement('tr');
        const now = new Date();
        const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

        newRow.innerHTML = `
            <td>${data.results[0] ? data.results[0].filename : 'Uploaded Video'}</td>
            <td>${dateStr}</td>
            <td><span class="status verified">Verified</span></td>
            <td>${data.results.map(r => r.class_name).filter((v, i, a) => a.indexOf(v) === i).join(', ') || 'Processing...'}</td>
        `;

        activityBody.insertBefore(newRow, activityBody.firstChild);

        // Hide processing status after a delay
        setTimeout(() => {
            processingStatus.classList.add('hidden');
            dropzone.classList.remove('hidden');
        }, 3000);

    } catch (error) {
        console.error('Failed to fetch results:', error);
    }
}

function resetUI() {
    processingStatus.classList.add('hidden');
    dropzone.classList.remove('hidden');
}
