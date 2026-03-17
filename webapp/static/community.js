const KINGSTON_CENTER = [44.2312, -76.4860];
const categoryColors = {
    pothole: '#d1495b',
    longitudinal_crack: '#2f6fed',
    transverse_crack: '#2f6fed',
    alligator_crack: '#2f6fed',
};

function colorForCategory(category) {
    return categoryColors[category] || '#6b7280';
}

const map = L.map('community-map', {
    zoomControl: false,
}).setView(KINGSTON_CENTER, 12);

L.control.zoom({ position: 'bottomright' }).addTo(map);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);

const clusterLayer = L.layerGroup().addTo(map);
const communityList = document.getElementById('community-list');

function formatSeverity(value) {
    return Number(value || 0).toFixed(2);
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
                <span>${hazard.detection_count} detections merged</span>
                <span>Severity ${formatSeverity(hazard.severity_score)}</span>
            </div>
        </div>
    `;
}

async function loadCommunityMap() {
    const response = await fetch('/api/community');
    const data = await response.json();
    const hazards = data.hazards || [];

    document.getElementById('community-jobs').textContent = String(data.summary.completed_jobs || 0);
    document.getElementById('community-hazards').textContent = String(data.summary.verified_hazard_count || 0);
    document.getElementById('community-categories').textContent = String(new Set(hazards.map((hazard) => hazard.category)).size);
    document.getElementById('community-max-severity').textContent = hazards.length ? formatSeverity(hazards[0].severity_score) : '0.00';
    document.getElementById('community-summary-chip').textContent = `${hazards.length} verified hazards`;

    clusterLayer.clearLayers();
    communityList.innerHTML = '';

    if (!hazards.length) {
        communityList.innerHTML = '<p class="empty-state">No completed jobs yet.</p>';
        map.setView(KINGSTON_CENTER, 12);
        return;
    }

    const bounds = [];
    hazards.forEach((hazard) => {
        const latlng = [hazard.centroid_lat, hazard.centroid_lon];
        bounds.push(latlng);
        L.circleMarker(latlng, {
            radius: Math.max(7, Math.min(20, 4 + Number(hazard.detection_count || 0))),
            color: colorForCategory(hazard.category),
            fillColor: colorForCategory(hazard.category),
            fillOpacity: 0.75,
            weight: 2,
        })
            .bindTooltip(tooltipHtml(hazard), {
                direction: 'top',
                sticky: true,
                opacity: 1,
                className: 'hazard-tooltip',
            })
            .addTo(clusterLayer);

        const card = document.createElement('article');
        card.className = 'priority-item';
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
                <span>${hazard.detection_count} detections merged</span>
            </div>
            <div class="priority-metrics">
                <span>Score ${formatSeverity(hazard.severity_score)}</span>
                <span>${hazard.unique_frame_count} frames</span>
            </div>
        `;
        communityList.appendChild(card);
    });

    map.fitBounds(bounds, { padding: [28, 28] });
}

loadCommunityMap().catch((error) => {
    console.error(error);
    communityList.innerHTML = '<p class="empty-state">Community map failed to load.</p>';
});
