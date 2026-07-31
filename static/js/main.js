// ===== Loading Spinner =====
const spinner = document.createElement('div');
spinner.id = 'pageSpinner';
spinner.innerHTML = `
  <div style="position:fixed;inset:0;background:rgba(255,255,255,0.7);
    display:flex;align-items:center;justify-content:center;z-index:9999;display:none;">
    <div style="text-align:center;">
      <div class="hv-spinner"></div>
      <div style="font-size:13px;color:#64748b;margin-top:8px;">Loading…</div>
    </div>
  </div>`;
document.body.appendChild(spinner);

function showSpinner() {
    document.getElementById('pageSpinner').firstElementChild.style.display = 'flex';
}
function hideSpinner() {
    document.getElementById('pageSpinner').firstElementChild.style.display = 'none';
}

// Show spinner on all non-AJAX form submits (except train form which has its own progress)
document.addEventListener('submit', function (e) {
    const form = e.target;
    if (form.id === 'cnnForm') return; // CNN has its own preview handler
    if (form.action && form.action.includes('/ann/train')) return; // ANN has progress bar
    showSpinner();
});

// Show spinner on sidebar navigation links
document.querySelectorAll('.sidebar-link').forEach(link => {
    link.addEventListener('click', () => showSpinner());
});

// Hide spinner when page is fully loaded
window.addEventListener('pageshow', hideSpinner);

// ===== Sidebar toggle persistence (mobile) =====
const sidebarToggleBtn = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');

sidebarToggleBtn?.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    localStorage.setItem('sidebarOpen', sidebar.classList.contains('open'));
});

// ===== Auto-dismiss flash alerts after 5s =====
document.querySelectorAll('.hv-alert').forEach(alert => {
    setTimeout(() => {
        alert.style.transition = 'opacity 0.5s ease';
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 500);
    }, 5000);
});

// ===== Confirm delete buttons =====
document.querySelectorAll('[data-confirm]').forEach(btn => {
    btn.addEventListener('click', e => {
        if (!confirm(btn.dataset.confirm)) e.preventDefault();
    });
});

// ===== Tooltip init (Bootstrap) =====
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el);
});
