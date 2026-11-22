/**
 * ClaimClarity AI - Raw Form Brutalist Edition
 * Frontend Controller & Image Comparison Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }

  // Application State
  const state = {
    selectedFile: null,
    selectedSampleId: null,
    selectedMode: 'neural_dce', // 'neural_dce' | 'hybrid_forensic' | 'classic_clahe'
    currentOriginalUrl: null,
    currentEnhancedUrl: null,
    currentMetadata: null,
    isProcessing: false,
    viewMode: 'split', // 'split' | 'side'
    splitPercent: 50,
    isDragging: false,
    isLoupeActive: false,
    isHoldingCompare: false,
    previousSplitPercent: 50
  };

  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileMetaPill = document.getElementById('file-meta-pill');
  const fileNameEl = document.getElementById('file-name');
  const fileSizeEl = document.getElementById('file-size');
  const btnClearFile = document.getElementById('btn-clear-file');
  const btnEnhance = document.getElementById('btn-enhance');
  const btnEnhanceText = document.getElementById('btn-enhance-text');

  const progressContainer = document.getElementById('progress-container');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const progressPct = document.getElementById('progress-pct');
  const progressStatusTitle = document.getElementById('progress-status-title');
  const progressStatusSub = document.getElementById('progress-status-sub');

  const errorAlert = document.getElementById('error-alert');
  const errorMessage = document.getElementById('error-message');
  const btnDismissError = document.getElementById('btn-dismiss-error');

  const emptyState = document.getElementById('empty-state');
  const previewSingleView = document.getElementById('preview-single-view');
  const imgSinglePreview = document.getElementById('img-single-preview');
  const splitView = document.getElementById('split-view');
  const sideView = document.getElementById('side-view');

  const imgEnhanced = document.getElementById('img-enhanced');
  const splitOverlay = document.getElementById('split-overlay');
  const imgOriginalSplit = document.getElementById('img-original-split');
  const splitHandle = document.getElementById('split-handle');

  const imgOriginalSide = document.getElementById('img-original-side');
  const imgEnhancedSide = document.getElementById('img-enhanced-side');
  const sideOrigDim = document.getElementById('side-orig-dim');
  const sideEnhDim = document.getElementById('side-enh-dim');

  const statusBadge = document.getElementById('status-badge');
  const viewModeSplit = document.getElementById('view-mode-split');
  const viewModeSide = document.getElementById('view-mode-side');
  const btnHoldCompare = document.getElementById('btn-hold-compare');
  const btnToggleLoupe = document.getElementById('btn-toggle-loupe');
  const btnDownloadEnhanced = document.getElementById('btn-download-enhanced');
  const btnDownloadProof = document.getElementById('btn-download-proof');
  const btnDownloadOrig = document.getElementById('btn-download-orig');

  const metricDim = document.getElementById('metric-dim');
  const metricTime = document.getElementById('metric-time');
  const metricHash = document.getElementById('metric-hash');

  const btnShowAudit = document.getElementById('btn-show-audit');
  const navBtnAudit = document.getElementById('nav-btn-audit');
  const auditModal = document.getElementById('audit-modal');
  const auditContent = document.getElementById('audit-content');
  const btnCloseAudit = document.getElementById('btn-close-audit');
  const btnDismissAudit = document.getElementById('btn-dismiss-audit');

  const loupeLens = document.getElementById('loupe-lens');
  const loupeImg = document.getElementById('loupe-img');
  const productCards = document.querySelectorAll('.product-card');

  // Format File Size
  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  // Error Banner
  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove('hidden');
    progressContainer.classList.add('hidden');
    state.isProcessing = false;
    btnEnhance.disabled = false;
    btnEnhanceText.textContent = 'ENHANCE FOR CLAIM REVIEW';
  }

  function clearError() {
    errorAlert.classList.add('hidden');
  }

  btnDismissError.addEventListener('click', clearError);

  // File Upload Handlers (Drag & Drop)
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith('image/')) {
      showError('Selected file is not an image. Please provide a valid JPEG, PNG, or WEBP photo.');
      return;
    }

    clearError();
    clearSampleSelection();
    state.selectedFile = file;
    state.selectedSampleId = null;

    fileNameEl.textContent = file.name.toUpperCase();
    fileSizeEl.textContent = formatBytes(file.size);
    fileMetaPill.classList.remove('hidden');

    const reader = new FileReader();
    reader.onload = (evt) => {
      state.currentOriginalUrl = evt.target.result;
      showPreEnhancePreview(state.currentOriginalUrl);
      btnEnhance.disabled = false;
      statusBadge.textContent = 'READY TO ENHANCE';
      statusBadge.className = 'px-2.5 py-0.5 bg-[#DB4A2B] text-white text-[10px] font-bold uppercase tracking-widest';
    };
    reader.readAsDataURL(file);
  }

  // Mode Select Buttons
  const modeButtons = document.querySelectorAll('.mode-select-btn');
  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mode');
      state.selectedMode = mode;
      modeButtons.forEach(b => {
        b.className = 'mode-select-btn px-2.5 py-2 border border-[#1E1E1E] bg-transparent text-[#1E1E1E] hover:bg-[#D9D6D0] font-bold uppercase tracking-wider text-[11px] text-center transition-all';
      });
      btn.className = 'mode-select-btn px-2.5 py-2 border border-[#1E1E1E] bg-[#1E1E1E] text-[#E4E2DD] font-bold uppercase tracking-wider text-[11px] text-center transition-all';
    });
  });

  // Scenario / Product Card Selection
  productCards.forEach(card => {
    card.addEventListener('click', () => {
      const sampleId = card.getAttribute('data-sample-id');
      selectSample(sampleId, card, true);
    });
  });

  function selectSample(sampleId, cardEl, shouldScroll = false) {
    clearError();
    clearFileSelection();
    state.selectedSampleId = sampleId;
    state.selectedFile = null;

    productCards.forEach(c => c.classList.remove('active'));
    if (cardEl) cardEl.classList.add('active');

    const sampleImg = cardEl ? cardEl.querySelector('img') : null;
    const sampleSrc = sampleImg ? sampleImg.src : `/static/samples/${sampleId}`;

    state.currentOriginalUrl = sampleSrc;
    showPreEnhancePreview(sampleSrc);

    btnEnhance.disabled = false;
    statusBadge.textContent = 'CASE MOUNTED';
    statusBadge.className = 'px-2.5 py-0.5 bg-[#1E1E1E] text-[#DB4A2B] text-[10px] font-bold uppercase tracking-widest border border-[#DB4A2B]';

    if (shouldScroll) {
      const labEl = document.getElementById('evidence-lab');
      if (labEl) {
        labEl.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }

  function clearSampleSelection() {
    productCards.forEach(c => c.classList.remove('active'));
    state.selectedSampleId = null;
  }

  function clearFileSelection() {
    state.selectedFile = null;
    fileInput.value = '';
    fileMetaPill.classList.add('hidden');
  }

  btnClearFile.addEventListener('click', (e) => {
    e.stopPropagation();
    resetWorkspace();
  });

  function resetWorkspace() {
    clearFileSelection();
    clearSampleSelection();
    clearError();
    state.currentOriginalUrl = null;
    state.currentEnhancedUrl = null;
    state.currentMetadata = null;

    emptyState.classList.remove('hidden');
    previewSingleView.classList.add('hidden');
    splitView.classList.add('hidden');
    sideView.classList.add('hidden');

    btnEnhance.disabled = true;
    btnHoldCompare.disabled = true;
    btnToggleLoupe.disabled = true;
    btnDownloadEnhanced.disabled = true;
    btnDownloadProof.disabled = true;
    btnDownloadOrig.disabled = true;

    metricDim.textContent = '100% PRESERVED';
    metricTime.textContent = '-- MS';
    metricHash.textContent = '--';

    statusBadge.textContent = 'AWAITING INPUT';
    statusBadge.className = 'px-2.5 py-0.5 bg-[#1E1E1E] text-[#E4E2DD] text-[10px] font-bold uppercase tracking-widest';
  }

  function showPreEnhancePreview(url) {
    emptyState.classList.add('hidden');
    splitView.classList.add('hidden');
    sideView.classList.add('hidden');
    previewSingleView.classList.remove('hidden');
    imgSinglePreview.src = url;

    btnHoldCompare.disabled = true;
    btnToggleLoupe.disabled = true;
    btnDownloadEnhanced.disabled = true;
    btnDownloadProof.disabled = true;
    btnDownloadOrig.disabled = false;
  }

  // Single Action: Enhance Button Click
  btnEnhance.addEventListener('click', async () => {
    if (state.isProcessing) return;
    if (!state.selectedFile && !state.selectedSampleId) {
      showError('Please upload a claim photo or select a scenario case first.');
      return;
    }
    startEnhancement();
  });

  // Hotkey: Ctrl+Enter / Cmd+Enter
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      if (!btnEnhance.disabled && !state.isProcessing) {
        startEnhancement();
      }
    }
  });

  async function startEnhancement() {
    clearError();
    state.isProcessing = true;
    btnEnhance.disabled = true;
    btnEnhanceText.textContent = 'CALCULATING CLARITY...';
    progressContainer.classList.remove('hidden');

    const setStep = (stepNum, title, sub, pct) => {
      progressStatusTitle.textContent = title;
      progressStatusSub.textContent = sub;
      progressBarFill.style.width = `${pct}%`;
      progressPct.textContent = `${pct}%`;

      for (let i = 1; i <= 4; i++) {
        const stepEl = document.getElementById(`step-${i}`);
        if (stepEl) {
          if (i <= stepNum) {
            stepEl.className = 'text-[#DB4A2B] font-bold';
          } else {
            stepEl.className = 'text-[#444444]';
          }
        }
      }
    };

    setStep(1, 'SEPARATING SENSOR NOISE...', 'EDGE-GATED GRADIENT MASK PROTECTS PHYSICAL SCRATCHES', 25);

    try {
      let response;
      const formData = new FormData();

      if (state.selectedFile) {
        formData.append('file', state.selectedFile);
        formData.append('mode', state.selectedMode);
        setTimeout(() => setStep(2, 'CLAIMDCE-NET INFERENCE...', 'EVALUATING 8 NON-LINEAR ILLUMINATION PARAMETER MAPS', 50), 120);
        setTimeout(() => setStep(3, 'PRESERVING HIGHLIGHTS...', 'QUADRATIC CURVE MAPPING WITH ZERO OVER-SATURATION', 75), 240);
        response = await fetch('/api/enhance', {
          method: 'POST',
          body: formData
        });
      } else {
        setTimeout(() => setStep(2, 'CLAIMDCE-NET INFERENCE...', 'EVALUATING 8 NON-LINEAR ILLUMINATION PARAMETER MAPS', 50), 120);
        setTimeout(() => setStep(3, 'PRESERVING HIGHLIGHTS...', 'QUADRATIC CURVE MAPPING WITH ZERO OVER-SATURATION', 75), 240);
        response = await fetch('/api/enhance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_id: state.selectedSampleId, mode: state.selectedMode })
        });
      }

      setStep(4, 'AUDIT CHAIN OF CUSTODY...', 'COMPUTING SHA-256 INTEGRITY FINGERPRINTS', 92);

      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || 'Failed to enhance photo.');
      }

      setStep(4, 'ENHANCEMENT COMPLETE', '1:1 RESOLUTION MAINTAINED. READY FOR ADJUSTER AUDIT.', 100);

      setTimeout(() => {
        progressContainer.classList.add('hidden');
        renderEnhancedResults(data);
      }, 350);

    } catch (err) {
      showError(err.message || 'Image enhancement failed. Please check network connection or try another photo.');
    } finally {
      state.isProcessing = false;
      btnEnhance.disabled = false;
      btnEnhanceText.textContent = 'ENHANCE FOR CLAIM REVIEW';
    }
  }

  function renderEnhancedResults(data) {
    state.currentOriginalUrl = data.original_image;
    state.currentEnhancedUrl = data.enhanced_image;
    state.currentMetadata = data.metadata;

    emptyState.classList.add('hidden');
    previewSingleView.classList.add('hidden');

    // Set split images
    imgEnhanced.src = state.currentEnhancedUrl;
    imgOriginalSplit.src = state.currentOriginalUrl;

    // Set side-by-side images
    imgOriginalSide.src = state.currentOriginalUrl;
    imgEnhancedSide.src = state.currentEnhancedUrl;

    const w = data.metadata.dimensions.width;
    const h = data.metadata.dimensions.height;
    sideOrigDim.textContent = `${w} × ${h} PX`;
    sideEnhDim.textContent = `${w} × ${h} PX (100% PRESERVED)`;

    // Enable buttons
    btnHoldCompare.disabled = false;
    btnToggleLoupe.disabled = false;
    btnDownloadEnhanced.disabled = false;
    btnDownloadProof.disabled = false;
    btnDownloadOrig.disabled = false;

    // Update status badge
    statusBadge.textContent = 'CLARITY ACTIVE';
    statusBadge.className = 'px-2.5 py-0.5 bg-[#DB4A2B] text-white text-[10px] font-bold uppercase tracking-widest';

    // Populate metrics
    metricDim.textContent = `${w} × ${h} PX (1:1)`;
    metricTime.textContent = `${data.metadata.processing_time_ms} MS`;
    metricHash.textContent = data.metadata.enhanced_sha256.substring(0, 16) + '...';

    // Show appropriate view
    if (state.viewMode === 'split') {
      splitView.classList.remove('hidden');
      sideView.classList.add('hidden');
      updateSplit(50);
    } else {
      splitView.classList.add('hidden');
      sideView.classList.remove('hidden');
    }

    if (window.lucide) lucide.createIcons();
  }

  // Split Slider Position Update
  function updateSplit(pct) {
    state.splitPercent = Math.max(0, Math.min(100, pct));
    splitHandle.style.left = `${state.splitPercent}%`;
    splitOverlay.style.clipPath = `polygon(0 0, ${state.splitPercent}% 0, ${state.splitPercent}% 100%, 0 100%)`;
  }

  // Dragging interaction for split handle
  function onDragStart(e) {
    state.isDragging = true;
    document.body.style.cursor = 'ew-resize';
  }

  function onDragEnd() {
    state.isDragging = false;
    document.body.style.cursor = 'default';
  }

  function onDragMove(e) {
    if (!state.isDragging) return;
    const rect = splitView.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const offsetX = clientX - rect.left;
    const pct = (offsetX / rect.width) * 100;
    updateSplit(pct);
  }

  splitHandle.addEventListener('mousedown', onDragStart);
  window.addEventListener('mouseup', onDragEnd);
  window.addEventListener('mousemove', onDragMove);

  splitHandle.addEventListener('touchstart', onDragStart, { passive: true });
  window.addEventListener('touchend', onDragEnd);
  window.addEventListener('touchmove', onDragMove, { passive: true });

  // Click on split viewport jumps divider
  splitView.addEventListener('click', (e) => {
    if (e.target.closest('#split-handle')) return;
    const rect = splitView.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    updateSplit(pct);
  });

  // View Mode Switcher
  viewModeSplit.addEventListener('click', () => {
    state.viewMode = 'split';
    viewModeSplit.className = 'px-4 py-1.5 text-xs font-bold uppercase tracking-wider bg-[#1E1E1E] text-[#E4E2DD] transition-all';
    viewModeSide.className = 'px-4 py-1.5 text-xs font-bold uppercase tracking-wider bg-transparent text-[#1E1E1E] hover:bg-[#D9D6D0] transition-all';

    if (state.currentEnhancedUrl) {
      splitView.classList.remove('hidden');
      sideView.classList.add('hidden');
      updateSplit(state.splitPercent);
    }
  });

  viewModeSide.addEventListener('click', () => {
    state.viewMode = 'side';
    viewModeSide.className = 'px-4 py-1.5 text-xs font-bold uppercase tracking-wider bg-[#1E1E1E] text-[#E4E2DD] transition-all';
    viewModeSplit.className = 'px-4 py-1.5 text-xs font-bold uppercase tracking-wider bg-transparent text-[#1E1E1E] hover:bg-[#D9D6D0] transition-all';

    if (state.currentEnhancedUrl) {
      splitView.classList.add('hidden');
      sideView.classList.remove('hidden');
    }
  });

  // Hold to Compare Feature (Button + Spacebar)
  function startHoldCompare() {
    if (!state.currentEnhancedUrl || state.isHoldingCompare) return;
    state.isHoldingCompare = true;
    state.previousSplitPercent = state.splitPercent;

    if (state.viewMode === 'split') {
      updateSplit(100); // 100% original overlay
    } else {
      imgOriginalSide.parentElement.classList.add('ring-4', 'ring-[#DB4A2B]');
    }
    btnHoldCompare.classList.add('bg-[#DB4A2B]', 'text-white');
  }

  function stopHoldCompare() {
    if (!state.isHoldingCompare) return;
    state.isHoldingCompare = false;

    if (state.viewMode === 'split') {
      updateSplit(state.previousSplitPercent);
    } else {
      imgOriginalSide.parentElement.classList.remove('ring-4', 'ring-[#DB4A2B]');
    }
    btnHoldCompare.classList.remove('bg-[#DB4A2B]', 'text-white');
  }

  btnHoldCompare.addEventListener('mousedown', startHoldCompare);
  btnHoldCompare.addEventListener('mouseup', stopHoldCompare);
  btnHoldCompare.addEventListener('mouseleave', stopHoldCompare);

  btnHoldCompare.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startHoldCompare();
  });
  btnHoldCompare.addEventListener('touchend', stopHoldCompare);

  // Spacebar hotkey
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.repeat && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
      e.preventDefault();
      startHoldCompare();
    }
  });

  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
      stopHoldCompare();
    }
  });

  // 2.5x Loupe Magnifier
  btnToggleLoupe.addEventListener('click', () => {
    state.isLoupeActive = !state.isLoupeActive;
    if (state.isLoupeActive) {
      btnToggleLoupe.classList.add('bg-[#DB4A2B]', 'text-white');
      loupeImg.src = state.currentEnhancedUrl;
    } else {
      btnToggleLoupe.classList.remove('bg-[#DB4A2B]', 'text-white');
      loupeLens.classList.add('hidden');
    }
  });

  const viewportContainer = document.getElementById('viewport-container');
  viewportContainer.addEventListener('mousemove', (e) => {
    if (!state.isLoupeActive || !state.currentEnhancedUrl) return;

    const targetImg = state.viewMode === 'split' ? imgEnhanced : imgEnhancedSide;
    const imgRect = targetImg.getBoundingClientRect();

    if (
      e.clientX < imgRect.left ||
      e.clientX > imgRect.right ||
      e.clientY < imgRect.top ||
      e.clientY > imgRect.bottom
    ) {
      loupeLens.classList.add('hidden');
      return;
    }

    loupeLens.classList.remove('hidden');
    const lensSize = 200;
    loupeLens.style.left = `${e.clientX - lensSize / 2}px`;
    loupeLens.style.top = `${e.clientY - lensSize / 2}px`;

    const zoom = 2.5;
    const relX = (e.clientX - imgRect.left) / imgRect.width;
    const relY = (e.clientY - imgRect.top) / imgRect.height;

    loupeImg.style.width = `${imgRect.width * zoom}px`;
    loupeImg.style.height = `${imgRect.height * zoom}px`;
    loupeImg.style.left = `${(lensSize / 2) - (relX * imgRect.width * zoom)}px`;
    loupeImg.style.top = `${(lensSize / 2) - (relY * imgRect.height * zoom)}px`;
  });

  viewportContainer.addEventListener('mouseleave', () => {
    if (state.isLoupeActive) {
      loupeLens.classList.add('hidden');
    }
  });

  // Download Actions
  btnDownloadEnhanced.addEventListener('click', () => {
    if (!state.currentEnhancedUrl) return;
    const link = document.createElement('a');
    link.href = state.currentEnhancedUrl;
    link.download = `CLAIMCLARITY_ENHANCED_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  btnDownloadOrig.addEventListener('click', () => {
    if (!state.currentOriginalUrl) return;
    const link = document.createElement('a');
    link.href = state.currentOriginalUrl;
    link.download = `CLAIMCLARITY_ORIGINAL_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  btnDownloadProof.addEventListener('click', async () => {
    if (!state.currentOriginalUrl) return;
    btnDownloadProof.disabled = true;
    const prevText = btnDownloadProof.innerHTML;
    btnDownloadProof.innerHTML = `<span>EXPORTING...</span>`;

    try {
      let response;
      if (state.selectedFile) {
        const formData = new FormData();
        formData.append('file', state.selectedFile);
        formData.append('mode', state.selectedMode);
        response = await fetch('/api/proof-sheet', { method: 'POST', body: formData });
      } else {
        response = await fetch('/api/proof-sheet', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_id: state.selectedSampleId, mode: state.selectedMode })
        });
      }

      if (!response.ok) throw new Error('Proof sheet export failed.');

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `CLAIMCLARITY_PROOF_SHEET_${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      showError('Failed to generate proof sheet: ' + err.message);
    } finally {
      btnDownloadProof.disabled = false;
      btnDownloadProof.innerHTML = prevText;
    }
  });

  // Forensic Audit Modal
  function renderAuditModal() {
    if (!state.currentMetadata) {
      // Default info if not enhanced yet
      auditContent.innerHTML = `
        <div class="p-4 border border-[#1E1E1E] bg-[#D9D6D0]">
          <p class="font-clash font-bold uppercase text-sm">NO ENHANCED EVIDENCE CURRENTLY MOUNTED</p>
          <p class="text-xs text-[#444444] mt-1">Select a scenario or upload an evidence photo to generate cryptographic SHA-256 hashes and latency benchmarks.</p>
        </div>
      `;
      auditModal.classList.remove('hidden');
      return;
    }

    const meta = state.currentMetadata;
    const modelSpecs = meta.model_specs || {};
    let stagesHtml = meta.pipeline_stages.map(s => `
      <div class="flex items-start space-x-3 p-3 border border-[#1E1E1E] bg-[#D9D6D0]">
        <span class="px-2 py-0.5 bg-[#DB4A2B] text-white font-clash font-bold text-xs">${s.step}</span>
        <div>
          <div class="font-clash font-bold text-sm uppercase text-[#1E1E1E]">${s.title}</div>
          <div class="text-xs text-[#444444] mt-0.5">${s.description}</div>
        </div>
      </div>
    `).join('');

    auditContent.innerHTML = `
      <!-- Model Training & Architecture Header -->
      <div class="p-4 border-2 border-[#DB4A2B] bg-[#E4E2DD] space-y-2">
        <div class="flex items-center justify-between">
          <span class="px-2 py-0.5 bg-[#DB4A2B] text-white font-clash font-bold text-[10px] uppercase tracking-wider">
            DEEP LEARNING MODEL
          </span>
          <span class="text-[11px] font-mono text-[#DB4A2B] font-bold uppercase">
            ACCELERATOR: ${meta.device || 'MPS'}
          </span>
        </div>
        <div class="font-clash font-bold text-lg uppercase text-[#1E1E1E]">
          ${modelSpecs.architecture || 'ClaimDCE-Net (Deep Curve Estimation Network)'}
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-[10px] uppercase font-bold text-[#444444]">
          <div>PARAMS: <span class="text-[#1E1E1E]">${modelSpecs.parameters || '79,422'}</span></div>
          <div>EPOCHS: <span class="text-[#1E1E1E]">${modelSpecs.trained_epochs || '12'}</span></div>
          <div>FINAL LOSS: <span class="text-[#DB4A2B]">${modelSpecs.final_loss || '1.7275'}</span></div>
          <div>DATASET: <span class="text-[#1E1E1E]">CAR DAMAGE (HF)</span></div>
        </div>
      </div>

      <div class="grid grid-cols-2 gap-4 p-4 border border-[#1E1E1E] bg-[#D9D6D0]">
        <div>
          <span class="text-[10px] font-bold uppercase text-[#444444]">DIMENSION SPECIFICATION</span>
          <div class="font-clash font-bold text-xl text-[#1E1E1E]">${meta.dimensions.width} × ${meta.dimensions.height} PX</div>
          <span class="text-[10px] text-[#DB4A2B] font-bold uppercase">100% PRESERVED (0 CROPPING)</span>
        </div>
        <div>
          <span class="text-[10px] font-bold uppercase text-[#444444]">INFERENCE TIME</span>
          <div class="font-clash font-bold text-xl text-[#DB4A2B] font-mono">${meta.processing_time_ms} MS</div>
          <span class="text-[10px] text-[#444444] uppercase">PYTORCH METAL SHADERS</span>
        </div>
      </div>

      <div class="space-y-2">
        <span class="text-[10px] font-bold uppercase tracking-wider text-[#1E1E1E]">CRYPTOGRAPHIC CHAIN OF CUSTODY (SHA-256)</span>
        <div class="p-3 border border-[#1E1E1E] bg-[#D9D6D0] font-mono text-[11px] space-y-1.5 break-all">
          <div><strong class="text-[#1E1E1E]">ORIGINAL HASH:</strong> ${meta.original_sha256}</div>
          <div><strong class="text-[#DB4A2B]">ENHANCED HASH:</strong> ${meta.enhanced_sha256}</div>
        </div>
      </div>

      <div class="space-y-2">
        <span class="text-[10px] font-bold uppercase tracking-wider text-[#1E1E1E]">DETERMINISTIC PIPELINE EXECUTION</span>
        <div class="space-y-2">
          ${stagesHtml}
        </div>
      </div>

      <div class="p-4 border border-[#DB4A2B] bg-[#DB4A2B]/10 text-xs text-[#1E1E1E] leading-relaxed">
        <strong class="font-clash uppercase font-bold text-[#DB4A2B]">REGULATORY CHARTER:</strong> ${meta.disclaimer}
      </div>
    `;
        <strong class="font-clash uppercase font-bold text-[#DB4A2B]">REGULATORY CHARTER:</strong> ${meta.disclaimer}
      </div>
    `;

    auditModal.classList.remove('hidden');
  }

  btnShowAudit.addEventListener('click', renderAuditModal);
  if (navBtnAudit) navBtnAudit.addEventListener('click', renderAuditModal);
  btnCloseAudit.addEventListener('click', () => auditModal.classList.add('hidden'));
  btnDismissAudit.addEventListener('click', () => auditModal.classList.add('hidden'));
  auditModal.addEventListener('click', (e) => {
    if (e.target === auditModal) auditModal.classList.add('hidden');
  });

  // Search & Bag buttons (Brutalist interactions)
  const btnQuickSearch = document.getElementById('btn-quick-search');
  if (btnQuickSearch) {
    btnQuickSearch.addEventListener('click', () => {
      const grid = document.getElementById('product-grid');
      if (grid) grid.scrollIntoView({ behavior: 'smooth' });
    });
  }

  const btnQuickBag = document.getElementById('btn-quick-bag');
  if (btnQuickBag) {
    btnQuickBag.addEventListener('click', () => {
      const lab = document.getElementById('evidence-lab');
      if (lab) lab.scrollIntoView({ behavior: 'smooth' });
    });
  }

  // Pre-mount Case 01 on initial page load for live inspection ready out-of-the-box
  if (productCards.length > 0) {
    selectSample('car_door_scratch.jpg', productCards[0], false);
  }
});
