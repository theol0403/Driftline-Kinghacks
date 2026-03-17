const KINGSTON_CENTER = [44.2312, -76.4860];
const categoryColors = {
    pothole: '#d1495b',
    longitudinal_crack: '#2f6fed',
    transverse_crack: '#2f6fed',
    alligator_crack: '#2f6fed',
};

const uploadForm = document.getElementById('upload-form');
const videoInput = document.getElementById('video-input');
const gpsInput = document.getElementById('gps-input');
const uploadButton = document.getElementById('upload-button');
const progressCard = document.getElementById('progress-card');
const progressTitle = document.getElementById('progress-title');
const progressPercent = document.getElementById('progress-percent');
const progressFill = document.getElementById('progress-fill');
const progressMeta = document.getElementById('progress-meta');
const currentJobStatus = document.getElementById('current-job-status');
const mapTitle = document.getElementById('map-title');
const hazardPanelTitle = document.getElementById('hazard-panel-title');
const priorityList = document.getElementById('priority-list');
const recentJobsBody = document.getElementById('recent-jobs-body');
const showDashboardMapButton = document.getElementById('show-dashboard-map');
const livePreviewImage = document.getElementById('live-preview-image');
const livePreviewEmpty = document.getElementById('live-preview-empty');

let currentJobId = null;
let currentEventSource = null;
let cachedDashboardHazards = [];
let visibleHazards = [];
let hazardMarkers = new Map();
let selectedHazardKey = null;
let hazardObserver = null;

const map = L.map('hazard-map', {
    zoomControl: false,
}).setView(KINGSTON_CENTER, 13);

L.control.zoom({ position: 'bottomright' }).addTo(map);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);

const hazardLayer = L.layerGroup().addTo(map);

function colorForCategory(category) {
    return categoryColors[category] || '#6b7280';
}

function formatCurrency(value) {
    return `$${Number(value || 0).toFixed(2)}`;
}

function formatDistance(value) {
    return `${Number(value || 0).toFixed(2)} km`;
}

function formatPercent(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatStatusLabel(value) {
    if (!value) {
        return 'Idle';
    }
    return value
        .replaceAll('_', ' ')
        .replace(/\b\w/g, (match) => match.toUpperCase());
}

function hazardKey(hazard, index) {
    const jobId = hazard.job_id || 'network';
    return `${jobId}:${hazard.category}:${hazard.centroid_lat}:${hazard.centroid_lon}:${index}`;
}

function tooltipHtml(hazard) {
    const image = hazard.thumbnail_url
        ? `<img src="${hazard.thumbnail_url}" alt="${hazard.category}" class="tooltip-image">`
        : '<div class="tooltip-image placeholder">No image</div>';

    return `
        <div class="tooltip-card">
            ${image}
            <div class="tooltip-copy">
                <strong>${hazard.category.replaceAll('_', ' ')}</strong>
                <span>${formatPercent(hazard.avg_confidence)} confidence</span>
                <span>${hazard.detection_count} detections</span>
                <span>Score ${Number(hazard.severity_score).toFixed(2)}</span>
            </div>
        </div>
    `;
}

function updateLivePreview(previewImageUrl) {
    if (!previewImageUrl) {
        livePreviewImage.classList.add('hidden');
        livePreviewEmpty.classList.remove('hidden');
        return;
    }

    livePreviewImage.src = `${previewImageUrl}?t=${Date.now()}`;
    livePreviewImage.classList.remove('hidden');
    livePreviewEmpty.classList.add('hidden');
}

function clearHazardObserver() {
    if (hazardObserver) {
        hazardObserver.disconnect();
        hazardObserver = null;
    }
}

function applyMarkerSelection(markerKey) {
    if (selectedHazardKey === markerKey) {
        return;
    }

    selectedHazardKey = markerKey;
    hazardMarkers.forEach((marker, key) => {
        const isSelected = key === markerKey;
        marker.setStyle({
            radius: isSelected ? 14 : marker.options.baseRadius,
            weight: isSelected ? 4 : 2,
            fillOpacity: isSelected ? 0.95 : 0.72,
        });
        if (isSelected) {
            marker.openTooltip();
        } else {
            marker.closeTooltip();
        }
    });

    priorityList.querySelectorAll('.priority-item').forEach((card) => {
        card.classList.toggle('active', card.dataset.markerKey === markerKey);
    });
}

function focusHazard(markerKey, shouldPan = false) {
    const marker = hazardMarkers.get(markerKey);
    if (!marker) {
        return;
    }
    applyMarkerSelection(markerKey);
    if (shouldPan) {
        map.panTo(marker.getLatLng(), { animate: true });
    }
}

function renderHazardList(hazards, titleText) {
    clearHazardObserver();
    hazardPanelTitle.textContent = titleText;
    priorityList.innerHTML = '';

    if (!hazards.length) {
        priorityList.innerHTML = '<p class="empty-state">No verified hazards yet.</p>';
        return;
    }

    hazards.forEach((hazard) => {
        const card = document.createElement('article');
        card.className = 'priority-item';
        card.dataset.markerKey = hazard.markerKey;
        card.innerHTML = `
            <div class="priority-thumb-shell">
                ${
                    hazard.thumbnail_url
                        ? `<img src="${hazard.thumbnail_url}" class="priority-thumb" alt="${hazard.category}">`
                        : '<div class="priority-thumb placeholder">No image</div>'
                }
            </div>
            <div class="priority-copy">
                <strong>${hazard.category.replaceAll('_', ' ')}</strong>
                <span>${hazard.video_filename || 'Route submission'}</span>
                <span>${hazard.detection_count} detections · ${formatPercent(hazard.avg_confidence)} confidence</span>
            </div>
            <div class="priority-metrics">
                <span>Score ${Number(hazard.severity_score).toFixed(2)}</span>
                <span>${Number(hazard.first_seen).toFixed(1)}s to ${Number(hazard.last_seen).toFixed(1)}s</span>
            </div>
        `;
        priorityList.appendChild(card);
    });

    hazardObserver = new IntersectionObserver(
        (entries) => {
            const visibleEntry = entries
                .filter((entry) => entry.isIntersecting)
                .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
            if (visibleEntry) {
                focusHazard(visibleEntry.target.dataset.markerKey, false);
            }
        },
        {
            root: priorityList,
            threshold: [0.6],
        },
    );

    priorityList.querySelectorAll('.priority-item').forEach((card) => {
        hazardObserver.observe(card);
    });
}

function renderHazards(hazards, titleText, sidebarTitle) {
    hazardLayer.clearLayers();
    hazardMarkers = new Map();
    visibleHazards = hazards.map((hazard, index) => ({
        ...hazard,
        markerKey: hazardKey(hazard, index),
    }));
    mapTitle.textContent = titleText;
    renderHazardList(visibleHazards, sidebarTitle);

    if (!visibleHazards.length) {
        map.setView(KINGSTON_CENTER, 13);
        selectedHazardKey = null;
        return;
    }

    const bounds = [];
    visibleHazards.forEach((hazard) => {
        const latlng = [hazard.centroid_lat, hazard.centroid_lon];
        bounds.push(latlng);
        const baseRadius = Math.max(7, Math.min(18, 4 + Number(hazard.detection_count || 0)));
        const marker = L.circleMarker(latlng, {
            radius: baseRadius,
            baseRadius,
            color: colorForCategory(hazard.category),
            fillColor: colorForCategory(hazard.category),
            fillOpacity: 0.72,
            weight: 2,
        }).bindTooltip(tooltipHtml(hazard), {
            direction: 'top',
            sticky: true,
            opacity: 1,
            className: 'hazard-tooltip',
        });

        marker.on('mouseover', () => focusHazard(hazard.markerKey, false));
        marker.on('click', () => focusHazard(hazard.markerKey, true));
        marker.addTo(hazardLayer);
        hazardMarkers.set(hazard.markerKey, marker);
    });

    map.fitBounds(bounds, { padding: [28, 28] });
    focusHazard(visibleHazards[0].markerKey, false);
}

function renderRecentJobs(jobs) {
    recentJobsBody.innerHTML = '';
    if (!jobs.length) {
        recentJobsBody.innerHTML = '<tr><td colspan="6" class="empty-row">No uploads yet.</td></tr>';
        return;
    }

    jobs.forEach((job) => {
        const row = document.createElement('tr');
        const canView = job.status === 'completed';
        row.innerHTML = `
            <td>${job.video_filename}</td>
            <td><span class="pill ${job.status}">${job.status}</span></td>
            <td>${job.verified_hazard_count}</td>
            <td>${formatDistance(job.distance_km)}</td>
            <td>${formatCurrency(job.credits_earned)}</td>
            <td>${canView ? `<button class="inline-button" data-job-id="${job.job_id}">View</button>` : ''}</td>
        `;
        recentJobsBody.appendChild(row);
    });
}

function updateDashboardStats(summary) {
    document.getElementById('total-credits').textContent = formatCurrency(summary.total_credits_earned);
    document.getElementById('total-verified').textContent = Number(summary.total_verified_hazards || 0).toString();
    document.getElementById('total-distance').textContent = formatDistance(summary.total_distance_km);
    document.getElementById('total-jobs').textContent = Number(summary.total_jobs || 0).toString();
}

function setProgress(snapshot) {
    const totalFrames = Number(snapshot.total_frames || 0);
    const framesProcessed = Number(snapshot.frames_processed || 0);
    const percent = totalFrames > 0 ? Math.min(100, Math.round((framesProcessed / totalFrames) * 100)) : 0;

    progressCard.classList.remove('hidden');
    progressTitle.textContent = snapshot.status === 'completed' ? 'Processing complete' : 'Processing route';
    progressPercent.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;
    progressMeta.textContent = `${framesProcessed} / ${totalFrames || 'unknown'} frames · ${snapshot.raw_detection_count || 0} projected detections`;
    currentJobStatus.textContent = formatStatusLabel(snapshot.status);
    updateLivePreview(snapshot.preview_image_url);

    if (snapshot.status === 'error') {
        progressMeta.textContent = snapshot.error || 'Processing failed';
    }
}

async function refreshDashboard(renderDefaultMap = true) {
    const response = await fetch('/api/dashboard');
    const data = await response.json();
    updateDashboardStats(data.summary);
    renderRecentJobs(data.recent_jobs);
    cachedDashboardHazards = data.priority_repairs;

    if (renderDefaultMap && !currentJobId) {
        renderHazards(
            data.priority_repairs,
            'Verified hazards across completed jobs',
            'Highest-priority hazards',
        );
    }
}

async function loadJobResults(jobId) {
    const response = await fetch(`/api/jobs/${jobId}/results`);
    if (!response.ok) {
        throw new Error('Failed to load job results');
    }

    const data = await response.json();
    currentJobId = jobId;
    currentJobStatus.textContent = formatStatusLabel(data.job.status);
    updateLivePreview(data.job.preview_image_url);
    renderHazards(
        data.hazards,
        `Verified hazards for ${data.job.video_filename}`,
        `Verified hazards for ${data.job.video_filename}`,
    );
}

function closeEventSource() {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }
}

async function startUpload(event) {
    event.preventDefault();

    if (!videoInput.files.length || !gpsInput.files.length) {
        window.alert('Select both a video and a GPS CSV.');
        return;
    }

    uploadButton.disabled = true;
    currentJobStatus.textContent = 'Uploading';
    updateLivePreview(null);

    const formData = new FormData();
    formData.append('video', videoInput.files[0]);
    formData.append('gps_csv', gpsInput.files[0]);

    try {
        const response = await fetch('/api/jobs', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        currentJobId = data.job_id;
        setProgress({
            status: 'queued',
            frames_processed: 0,
            total_frames: 0,
            raw_detection_count: 0,
            preview_image_url: null,
        });

        closeEventSource();
        currentEventSource = new EventSource(`/api/jobs/${currentJobId}/events`);
        currentEventSource.onmessage = async (message) => {
            const snapshot = JSON.parse(message.data);
            setProgress(snapshot);

            if (snapshot.status === 'completed') {
                closeEventSource();
                await refreshDashboard(false);
                await loadJobResults(currentJobId);
                uploadButton.disabled = false;
                videoInput.value = '';
                gpsInput.value = '';
            }

            if (snapshot.status === 'error') {
                closeEventSource();
                uploadButton.disabled = false;
                window.alert(snapshot.error || 'Processing failed');
            }
        };
    } catch (error) {
        uploadButton.disabled = false;
        currentJobStatus.textContent = 'Upload Failed';
        window.alert(error.message);
    }
}

recentJobsBody.addEventListener('click', async (event) => {
    const button = event.target.closest('button[data-job-id]');
    if (!button) {
        return;
    }
    await loadJobResults(button.dataset.jobId);
});

priorityList.addEventListener('mouseenter', (event) => {
    const card = event.target.closest('.priority-item');
    if (!card) {
        return;
    }
    focusHazard(card.dataset.markerKey, true);
}, true);

showDashboardMapButton.addEventListener('click', () => {
    currentJobId = null;
    renderHazards(
        cachedDashboardHazards,
        'Verified hazards across completed jobs',
        'Highest-priority hazards',
    );
    currentJobStatus.textContent = 'City View';
});

uploadForm.addEventListener('submit', startUpload);

refreshDashboard().catch((error) => {
    console.error(error);
    priorityList.innerHTML = '<p class="empty-state">Dashboard failed to load.</p>';
});
