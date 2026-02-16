const API_BASE = "";
let currentSessionId = "";
let pendingUpdate = null;

// Theme Management
function initTheme() {
    const theme = localStorage.getItem("theme") || "light";
    const html = document.documentElement;
    if (theme === "dark") {
        html.setAttribute("data-theme", "dark");
        updateThemeIcon("sun");
    } else {
        html.removeAttribute("data-theme");
        updateThemeIcon("moon");
    }
}

function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.getAttribute("data-theme") === "dark";
    if (isDark) {
        html.removeAttribute("data-theme");
        localStorage.setItem("theme", "light");
        updateThemeIcon("moon");
    } else {
        html.setAttribute("data-theme", "dark");
        localStorage.setItem("theme", "dark");
        updateThemeIcon("sun");
    }
}

function updateThemeIcon(icon) {
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.innerHTML = `<i class="fas fa-${icon}"></i>`;
}

document.addEventListener("DOMContentLoaded", initTheme);

function log(message) {
    const el = document.getElementById("console");
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    el.innerHTML += `<div><span style="color:#6B7280;font-size:0.75rem">[${time}]</span> ${message}</div>`;
    el.scrollTop = el.scrollHeight;
}

function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    if (loading) {
        btn.disabled = true;
        btn._origHTML = btn.innerHTML;
        btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${btn.textContent.trim()}...`;
    } else {
        btn.disabled = false;
        if (btn._origHTML) btn.innerHTML = btn._origHTML;
    }
}

document.addEventListener("visibilitychange", () => {
    if (!document.hidden && pendingUpdate) {
        const { preview, logMsg, btnId } = pendingUpdate;
        if (preview) updateUI(preview);
        if (logMsg) log(logMsg);
        if (btnId) setLoading(btnId, false);
        pendingUpdate = null;
    }
});

// 1. Upload
async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files[0]) return alert("Please select a file");

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        log("Uploading dataset...");
        setLoading("btn-scan", true);
        const response = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
        const data = await response.json();

        if (!response.ok || data.detail) { log(`Error: ${data.detail || "Upload failed"}`); return; }
        if (!data.session_id) { log("Error: No session ID returned"); return; }

        currentSessionId = data.session_id;
        document.getElementById("session-badge").classList.remove("hidden");
        document.getElementById("session-id-display").innerText = "SESSION: " + currentSessionId.substring(0, 8);

        document.getElementById("btn-scan").disabled = false;
        document.getElementById("btn-train").disabled = true;

        updateUI(data.preview || []);

        if (data.stats) {
            const s = data.stats;
            document.getElementById("stat-rows").innerText = (s.original_rows || 0).toLocaleString();
            if (s.using_dask) {
                log(`DASK MODE: ${s.original_rows?.toLocaleString()} rows | Out-of-core processing enabled`);
            } else {
                log(`Loaded ${s.original_rows?.toLocaleString() || "?"} rows | Memory: ${s.memory_mb}MB (saved ${s.memory_saved_pct}%)`);
            }
        }
        if (data.info) log(data.info);
        log("Upload successful. Session started.");
    } catch (e) {
        log("Error: " + e.message);
    } finally {
        setLoading("btn-scan", false);
    }
}

// 2. Scan
async function scanData() {
    try {
        log("Scanning for anomalies...");
        setLoading("btn-scan", true);
        const response = await fetch(`${API_BASE}/scan/${currentSessionId}`);
        const report = await response.json();

        displayReport(report);
        document.getElementById("btn-clean").disabled = false;

        const missingCount = Object.keys(report.missing_analysis || {}).length;
        const outlierCount = Object.keys(report.outlier_analysis || {}).length;
        document.getElementById("stat-missing").innerText = missingCount > 0 ? `${missingCount} cols` : "Clean";
        document.getElementById("stat-dupes").innerText = report.recommendations?.drop_duplicates ? "Found" : "None";
        document.getElementById("stat-quality").innerText = missingCount === 0 && outlierCount === 0 ? "High" : "Needs work";

        log("Health scan complete.");
    } catch (e) { log("Scan failed: " + e.message); }
    finally { setLoading("btn-scan", false); }
}

// 3. Auto Clean
async function autoClean() {
    try {
        log("Running optimization pipeline...");
        setLoading("btn-clean", true);
        const response = await fetch(`${API_BASE}/auto-clean/${currentSessionId}`, { method: "POST" });
        const data = await response.json();

        if (!response.ok || data.detail) { log(`Cleaning failed: ${data.detail || "Unknown error"}`); return; }

        const msg = `Data cleaned & transformed. ${data.columns?.length || 0} columns.`;
        if (document.hidden) {
            pendingUpdate = { preview: data.preview, logMsg: msg, btnId: "btn-clean" };
        } else {
            if (data.preview) updateUI(data.preview);
            log(msg);
        }
        document.getElementById("btn-undo").disabled = false;
        document.getElementById("btn-train").disabled = false;
        document.getElementById("btn-export").disabled = false;
    } catch (e) { log("Cleaning failed: " + e.message); }
    finally { if (!pendingUpdate) setLoading("btn-clean", false); }
}

// 4. Undo
async function undoAction() {
    try {
        const response = await fetch(`${API_BASE}/undo/${currentSessionId}`, { method: "POST" });
        const data = await response.json();
        if (data.preview) updateUI(data.preview);
        log(data.message);
    } catch (e) { log("Undo failed."); }
}

// 5. Export
async function exportData() {
    try {
        log("Generating export file...");
        setLoading("btn-export", true);
        const res = await fetch(`${API_BASE}/export/${currentSessionId}`);
        if (!res.ok) { log("Export failed: " + (await res.json()).detail); return; }
        const blob = await res.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "processed_data.csv";
        a.click();
        URL.revokeObjectURL(a.href);
        log("Export successful.");
    } catch (e) { log("Export failed: " + e.message); }
    finally { setLoading("btn-export", false); }
}

// 6. Train
async function trainModel() {
    const target = document.getElementById("target-col").value;
    const modelType = document.getElementById("model-type").value;
    if (!target) return alert("Please specify a target column");

    try {
        log(`Training ${modelType} on target: ${target}...`);
        setLoading("btn-train", true);
        const response = await fetch(`${API_BASE}/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: currentSessionId, target_column: target, model_type: modelType })
        });
        const result = await response.json();

        if (result.status === "success") {
            const score = (result.metrics.accuracy || result.metrics.r2_score).toFixed(4);
            const label = result.metrics.accuracy ? "Accuracy" : "R2";
            log(`Model trained! ${result.metrics.type} ${label}: ${score}`);
        } else {
            log("Error: " + result.error);
        }
    } catch (e) { log("Training failed: " + e.message); }
    finally { setLoading("btn-train", false); }
}

// UI Helpers
function updateUI(previewData) {
    document.getElementById("empty-state").classList.add("hidden");
    document.getElementById("data-content").classList.remove("hidden");

    const table = document.getElementById("preview-table");
    if (!previewData.length) return;

    const keys = Object.keys(previewData[0]);
    let html = `<thead><tr>${keys.map(k => `<th>${k}</th>`).join("")}</tr></thead><tbody>`;
    previewData.forEach(row => {
        html += "<tr>" + keys.map(k => {
            const v = row[k];
            return `<td>${typeof v === "number" ? v.toFixed(2) : v}</td>`;
        }).join("") + "</tr>";
    });
    html += "</tbody>";
    table.innerHTML = html;
}

function displayReport(report) {
    const container = document.getElementById("scan-report");
    let html = '<div class="report-grid">';

    html += '<div class="report-section report-section-danger"><div class="report-section-title">Missing Values</div>';
    const missing = report.missing_analysis || {};
    if (Object.keys(missing).length === 0) html += '<div class="report-item"><span>No missing values detected</span></div>';
    for (const col in missing) {
        html += `<div class="report-item"><span>${col}</span><span class="badge badge-danger">${missing[col]}%</span></div>`;
    }
    html += "</div>";

    html += '<div class="report-section report-section-warning"><div class="report-section-title">Outliers Detected</div>';
    const outliers = report.outlier_analysis || {};
    if (Object.keys(outliers).length === 0) html += '<div class="report-item"><span>No outliers detected</span></div>';
    for (const col in outliers) {
        const o = outliers[col];
        html += `<div class="report-item"><span>${col}</span><span class="badge badge-warning">${o.pct}% (${o.count} rows)</span></div>`;
    }
    html += "</div></div>";

    container.innerHTML = html;
}

document.getElementById("fileInput").addEventListener("change", function () {
    if (this.files[0]) document.getElementById("file-name").innerText = this.files[0].name;
});
