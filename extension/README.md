# ClaimClarity AI - Chromium Extension ("Circle-to-Enhance" Lasso Lens)

A Chromium browser extension (Chrome, Brave, Edge, Opera, Vivaldi, Arc) bringing **interactive "Circle-to-Enhance"** forensic image enhancement directly to your browser with **Raw Form Swiss Brutalist** styling.

---

## ⚡ Key Features

1. **Freehand Optical Lasso Gesture**:
   - Click the extension icon in the toolbar or press <kbd>Alt+C</kbd> (Mac: <kbd>⌥+C</kbd>) on any webpage.
   - Drag to lasso any car part, dent, scratch, invoice, or image on the screen.
   - Dynamic multi-color glowing trail (`#4285F4`, `#DB4A2B`, `#FBBC05`, `#34A853`) with neon shimmer particles.
2. **Brutalist Frame & Confirm Cutout**:
   - Releasing the cursor snaps to an exact pixel-accurate rectangular bounding box with Swiss Brutalist corner L-brackets and a darkened spotlight cutout.
   - Monospace dimension tag `[ W × H PX ]` and floating `[CONFIRM & ENHANCE]` and `[CANCEL]` action buttons.
3. **1:1 Resolution Locked Neural & CLAHE Enhancement**:
   - Default method: **Classic CLAHE** (deterministic, contrast-limited adaptive histogram equalization).
   - Switchable to **ClaimDCE-Net** or **Hybrid Forensic** directly in the extension popup.
   - Takes a native viewport screenshot cropped to the exact bounding box using `window.devicePixelRatio`.
   - Sends the image directly to the local ClaimClarity AI engine (`http://127.0.0.1:5001/api/enhance`).
   - Strict 1:1 image dimension lock with zero artificial interpolation or compression artifacts.
4. **Dual Presentation Architecture**:
   - **Floating On-Page Modal**: Immediately displays an interactive Before/After Split Curtain slider with 4-Task Forensic Appraisal (Defect Classification, Severity, Fraud Risk, Laplacian Sharpness, SNR).
   - **Extension Toolbar Popup**: Pops up under the extension icon with the complete forensic dossier, downloadable enhanced PNG, format switcher, and real-time ML backend status.

---

## 🚀 How to Install via "Load Unpacked" (Developer Mode)

### Step 1: Ensure Local ClaimClarity Server is Running
In your terminal, start the ClaimClarity server:
```bash
cd /Users/karthikeyaganugula/Downloads/Projects/Claim-Clarity
./run.sh
```
Verify it is active at: `http://127.0.0.1:5001/api/health`

---

### Step 2: Load Unpacked in Chrome / Chromium

1. Open **Chrome** (or Edge / Brave / Arc / Opera).
2. Navigate to the Extensions management page:
   - **Chrome**: `chrome://extensions`
   - **Brave**: `brave://extensions`
   - **Edge**: `edge://extensions`
3. Toggle the **"Developer mode"** switch in the top-right corner to **ON**.
4. Click the **"Load unpacked"** button in the top-left corner.
5. Select the `extension/` folder located at:
   ```
   /Users/karthikeyaganugula/Downloads/Projects/Claim-Clarity/extension
   ```
6. The extension **"ClaimClarity AI - Circle to Search Lens"** is now loaded!
7. **Pin the extension icon** in your browser toolbar for quick one-click access.

---

## 🎯 How to Use

1. Navigate to any webpage with images or vehicle claims (e.g. an insurance portal, online auction, or car photo).
2. Click the ClaimClarity extension icon in your browser toolbar and click **[ACTIVATE CIRCLE-TO-ENHANCE]**, or press <kbd>Alt+C</kbd> (Mac: <kbd>⌥+C</kbd>).
3. **Circle any damage or element**: Click and hold to round off the area with the glowing lasso trail.
4. **Release cursor**: The Swiss Brutalist rectangular frame appears with `[CONFIRM & ENHANCE]` and `[CANCEL]`.
5. Click **[CONFIRM & ENHANCE]**:
   - The ML engine processes the crop.
   - The interactive Before/After Split Curtain slider pops up right on your screen.
   - Click the extension icon in the toolbar at any time to inspect the full forensic dossier under the icon!

---

## 🛠️ Architecture & Permissions

- **Manifest V3 Compliant**: Built strictly for current and future Chromium standards.
- **Permissions**:
  - `activeTab`: Accesses the active tab when triggered.
  - `scripting`: Allows injecting selection overlay into pre-opened tabs.
  - `storage`: Caches latest enhancement results for instant popup preview.
- **Host Permissions**:
  - `http://127.0.0.1:5001/*`: Direct high-speed relay to local PyTorch/MPS inference pipeline.
  - `<all_urls>`: Enables tab viewport capture across any webpage.
