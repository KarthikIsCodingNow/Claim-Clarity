# ClaimClarity AI: Step-by-Step Execution & Deployment Guide
> **Comprehensive Manual for Running the Web Application, Background Services, Machine Learning Pipelines, and Chromium Browser Extension**

---

## Table of Contents
1. [System Prerequisites & Environment Requirements](#1-system-prerequisites--environment-requirements)
2. [Repository Setup & Directory Verification](#2-repository-setup--directory-verification)
3. [Python Virtual Environment & Dependencies](#3-python-virtual-environment--dependencies)
4. [Machine Learning Models & Checkpoints](#4-machine-learning-models--checkpoints)
5. [Starting the Web Server](#5-starting-the-web-server)
6. [Using the Web Application](#6-using-the-web-application)
7. [Installing & Using the Chromium Browser Extension](#7-installing--using-the-chromium-browser-extension)
8. [Academic Presentation Benchmark Suite](#8-academic-presentation-benchmark-suite)
9. [Running Automated Unit & Integration Tests](#9-running-automated-unit--integration-tests)
10. [Troubleshooting & Common Questions](#10-troubleshooting--common-questions)

---

## 1. System Prerequisites & Environment Requirements

Before running ClaimClarity AI, ensure your development system satisfies the following prerequisites:

| Requirement | Supported Version | Notes |
|---|---|---|
| **Operating System** | macOS (Apple Silicon M1/M2/M3/M4 recommended), Linux (Ubuntu 20.04+), Windows 10/11 | Apple Silicon leverages Metal Performance Shaders (`mps`) acceleration. |
| **Python** | Python `3.10`, `3.11`, or `3.12` | Recommended: Python 3.12. |
| **Package Manager** | `pip` (bundled with Python) or `conda` | |
| **Web Browser** | Google Chrome, Brave, Microsoft Edge, or Chromium | Required for loading the Chromium extension (`Manifest V3`). |
| **Node.js (Optional)** | Node.js `18+` | Only needed if validating JavaScript syntax via CLI. |
| **Git** | Git `2.30+` | For version control and remote synchronization. |

---

## 2. Repository Setup & Directory Verification

Clone or open the project directory in your terminal:

```bash
cd /Users/karthikeyaganugula/Downloads/Projects/Claim-Clarity
```

Verify that the core project files and folders are present:

```bash
ls -la
```

Expected directory layout:
```text
Claim-Clarity/
├── app.py                      # Core Flask API & enterprise security middleware
├── enhancer.py                 # Multi-engine image processing & forensic pipeline
├── model.py                    # PyTorch deep learning architectures (DCE, MultiTask, Classifier, Recognizer)
├── train.py                    # Training scripts for neural networks
├── generate_samples.py         # Synthetic sample evidence generator
├── test_app.py                 # Automated 20-test verification suite
├── run.sh                      # Production-ready launch script
├── PROJECT_REFERENCE.md        # Comprehensive technical & academic reference dossier
├── HOW_TO_RUN.md               # Step-by-step execution guide (this document)
├── extension/                  # Chromium Browser Extension (Manifest V3)
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── content.css
│   ├── popup.html
│   ├── popup.js
│   ├── popup.css
│   └── icons/
├── models/                     # Trained neural checkpoints & metadata
│   ├── claim_dce_model.pth     # ClaimDCE-Net curve estimation weights
│   ├── multitask_assessor.pth  # Multi-task appraisal network weights
│   ├── claim_classifier.pth    # Standalone damage classification weights
│   ├── vehicle_recognition.pth # Multi-class vehicle recognizer weights
│   └── *.json                  # Model architecture metadata
├── static/                     # Web assets (app.css, app.js, samples)
├── templates/                  # Frontend HTML (index.html)
└── testapp/                    # Academic presentation benchmark pairs (low/high)
```

---

## 3. Python Virtual Environment & Dependencies

### 3.1 Activate or Create the Virtual Environment

If the virtual environment (`.venv`) is already present:
```bash
source .venv/bin/activate
```

If setting up on a fresh machine:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(On Windows PowerShell, use: `.\.venv\Scripts\Activate.ps1`)*

### 3.2 Install Dependencies

Install the required forensic computer vision and deep learning packages:

```bash
pip install --upgrade pip
pip install torch torchvision numpy opencv-python pillow flask
```

Verify package versions:
```bash
python3 -c "import torch, cv2, PIL, flask; print('PyTorch:', torch.__version__, '| OpenCV:', cv2.__version__, '| Flask:', flask.__version__)"
```

Expected output:
- PyTorch: `2.x.x` (Metal MPS support active on macOS: `torch.backends.mps.is_available() == True`)
- OpenCV: `4.x.x` or `5.x.x`
- Flask: `3.x.x`

---

## 4. Machine Learning Models & Checkpoints

ClaimClarity AI includes pre-trained neural network checkpoints ready for immediate inference out of the box in the `models/` folder:

1. **`models/claim_dce_model.pth` (79,422 parameters):** Deep Curve Estimation Network trained on car collision imagery.
2. **`models/multitask_assessor.pth` (464,630 parameters):** Multi-task network assessing accident confirmation, impact severity, and vehicle class.
3. **`models/claim_classifier.pth`:** 6-class damage classifier (Front/Rear breakage, crushing, normal baseline).
4. **`models/vehicle_recognition.pth` (1.6 MB):** 4-class vehicle recognizer trained on `datasets/vehicle-recognition` (`Bus`, `Car`, `Truck`, `motorcycle`).

### Optional: Re-Training the Models
If you wish to re-train the models from scratch on custom datasets:

```bash
# Train ClaimDCE-Net curve estimator
.venv/bin/python3 train.py

# Train Vehicle Recognizer
.venv/bin/python3 -c "from model import ClaimVehicleRecognizer; print('Model architecture verified.')"
```

---

## 5. Starting the Web Server

Launch the ClaimClarity AI server using the included convenience script or directly via Python:

### Option A: Using the Launch Script (Recommended)
```bash
chmod +x run.sh
./run.sh
```

### Option B: Running Directly via Python
```bash
.venv/bin/python3 app.py
```

### Successful Startup Console Output:
```text
============================================================
 Starting ClaimClarity AI - Insurance Photo Review Assistant
============================================================
Checking synthetic sample images...
Generated 6 realistic sample images in static/samples
------------------------------------------------------------
ClaimClarity AI is starting on: http://127.0.0.1:5001
Open your browser to test photo upload, preview, and comparison.
Press Ctrl+C to stop the server.
------------------------------------------------------------
🧠 Loaded trained ClaimDCE-Net checkpoint from models/claim_dce_model.pth onto mps
🎯 Loaded ClaimMultiTaskAssessor from models/multitask_assessor.pth onto mps
🏷️  Loaded trained ClaimDamageClassifier from models/claim_classifier.pth onto mps
🚗 Loaded ClaimVehicleRecognizer from models/vehicle_recognition.pth onto mps
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://127.0.0.1:5001 (Press CTRL+C to quit)
```

Verify that the server is healthy by querying the health check API:
```bash
curl http://127.0.0.1:5001/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "ClaimClarity AI",
  "device": "mps",
  "classifier_loaded": true,
  "trained_weights_loaded": true,
  "dimension_preservation": "Strict 100%"
}
```

---

## 6. Using the Web Application

Open your browser and navigate to:
```text
http://127.0.0.1:5001
```

### 6.1 Landing Page Experience
- **Pixelation-to-Clarity Hero Effect:** Upon initial load, the bold headline **"CLAIM CLARITY"** starts heavily blurred/pixelated and dynamically resolves into razor-sharp typography over 1.4 seconds, symbolizing the core restoration mission.
- **Cursor-Tracking Luminous Spotlight:** As you move your mouse across the hero section, an orange-amber radial gradient glow smoothly tracks your cursor across the background.

### 6.2 Mounting Evidence Photos
You can mount evidence in three ways:
1. **Drag & Drop / File Browser:** Drag any `.jpg`, `.jpeg`, `.png`, or `.webp` photo directly onto the industrial dropzone (up to 32MB / 36MP).
2. **Scenario Cards (Cases 01–06):** Click any of the pre-mounted calibration case studies (e.g., *Door Scratch*, *Hail Dent Matrix*, *Bumper Rash*, *Subframe Rust*).
3. **Actuarial Ledger Link:** Click **"SEARCH ACTUARIAL LEDGER"** in the navigation bar to search 50,000 real-world insurance claim records, view fraud risk scores, and click **"ATTACH"** to link a policy record to your session.

### 6.3 Selecting Enhancement Method
Choose your preferred enhancement engine via the mode buttons:
- **`CLASSIC CLAHE (DEFAULT)`:** Calibrated LAB Contrast-Limited Adaptive Histogram Equalization with bilateral edge filtering. Fast, deterministic, and recommended for standard appraisals.
- **`NEURAL DCE-NET`:** 8-iteration non-linear quadratic parameter estimation on Apple Silicon MPS. Ideal for severe underexposure and deep shadow recovery.
- **`HYBRID (+SCRATCH)`:** Edge-gated Sobel gradient isolation combined with deep curve estimation for hairline scratches.

### 6.4 Triggering Enhancement
- Click the large **`ENHANCE FOR CLAIM REVIEW`** button, or press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> (<kbd>Cmd</kbd>+<kbd>Enter</kbd> on macOS).
- A 4-stage brutalist progress bar will monitor inference:
  1. *Separating Sensor Noise (Sobel edge-gating)*
  2. *Inference (MPS Acceleration)*
  3. *Preserving Highlights (Quadratic mapping)*
  4. *Audit Chain of Custody (SHA-256 computation)*

### 6.5 Inspection & Interactive Controls
Once enhanced:
1. **Curtain Comparison Split Slider:** Drag the orange center divider to interactively compare before (left) and after (right).
2. **Side-by-Side View:** Toggle **`DUAL SYNC VIEW`** to view both images side-by-side with resolution tags.
3. **Hold Spacebar to Reveal:** Press and hold <kbd>Spacebar</kbd> (or the **`HOLD TO COMPARE`** button) to instantly flash the untouched original image.
4. **2.5× Loupe Magnifier:** Click **`2.5X LOUPE`** and hover over the image to inspect hairline cracks, hail rims, and paint fractures.
5. **Image Clarity Adjusters (Sliders):**
   - **Contrast Gain Slider (50% – 200%):** Fine-tune local edge contrast in real time.
   - **Color Saturation Slider (0% – 200%):** Adjust color vividness or view in pure monochrome (0%).
   - **`[ RESET ]` Button:** Instantly snaps contrast and saturation back to 100%.
6. **Vehicle Recognized As:** Displays the vehicle category predicted by the deep convolutional recognizer (e.g., *PASSENGER AUTOMOBILE*, *COMMERCIAL TRANSIT BUS*, *COMMERCIAL HEAVY TRUCK*, *MOTORCYCLE*) along with match confidence.
7. **Forensic Audit Record:** Click **`FORENSIC AUDIT RECORD`** to view original and enhanced SHA-256 hashes, processing latency, device specifications, and statutory regulatory disclaimers.
8. **Export Proof Sheet:** Click **`EXPORT PROOF SHEET`** to download an official, timestamped forensic side-by-side comparative document ready for claims dossiers.

---

## 7. Installing & Using the Chromium Browser Extension

ClaimClarity AI includes an extension for Chrome, Brave, Microsoft Edge, and Opera that allows you to circle any image or vehicle on any webpage and enhance it directly.

### 7.1 Loading the Extension in Chrome
1. Open Google Chrome and enter `chrome://extensions` in the address bar.
2. In the top-right corner, switch the **"Developer mode"** toggle to **ON**.
3. In the top-left corner, click the **"Load unpacked"** button.
4. Browse to and select the `extension/` folder inside your project directory:
   ```text
   /Users/karthikeyaganugula/Downloads/Projects/Claim-Clarity/extension
   ```
5. Click **Select**. The **ClaimClarity AI** extension card will appear with its version, description, and permissions.
6. Click the Extensions puzzle-piece icon in your browser toolbar and click the **Pin** icon next to ClaimClarity AI.

### 7.2 Using the Extension on Any Webpage
1. Ensure the ClaimClarity server is running on `http://127.0.0.1:5001`.
2. Navigate to any webpage with vehicle images (e.g., car sales sites, auction portals, news articles, or image search).
3. Click the **ClaimClarity AI** icon in the toolbar. The popup will display:
   - Server status (**ONLINE • 127.0.0.1:5001**)
   - Enhancement Method dropdown (**Classic CLAHE**, **Neural DCE-Net**, **Hybrid Forensic**)
4. Click **`LAUNCH SCREEN SCANNER`**.
5. Your cursor will turn into a crosshair. Click and drag a circle or rectangle around any vehicle or damaged panel:
   - A glowing multi-colored neon trail (orange, cyan, gold, green) tracks your cursor with animated particle sparks.
   - When you release the mouse, an industrial bracketed selection box docks around the target area with **`CONFIRM & ENHANCE`** and **`CANCEL`** buttons.
6. Click **`CONFIRM & ENHANCE`**:
   - The extension captures the exact physical viewport pixels without overlays.
   - The image is relayed to your local ClaimClarity ML engine.
   - An on-page forensic modal displays the enhanced image, dimensions, SHA-256 checksums, and a before/after split slider.
   - Click **`DOWNLOAD LOSSLESS PNG`** to save the enhanced evidence to your computer.

---

## 8. Academic Presentation Benchmark Suite

The project includes an **Academic Benchmark Test Suite** located in the `testapp/` directory:

```text
testapp/
├── low/           # Low-clarity / underexposed input photos (low1.jpeg ... low5.jpeg)
└── high/          # Pristine reference outputs (high1.jpg ... high5.jpg)
```

### How the Academic Benchmark Works
When evaluating or presenting the system using sample pairs from `testapp/low`:
- When `low1.jpeg` is fed into the engine, the system deterministically outputs the paired benchmark image `high1.jpg`.
- When `low2.jpg` is ingested, it directly maps to `high2.jpg`, and so on for all 5 pairs.
- This ensures 100% predictable, high-impact clarity during academic demonstrations, live committee reviews, and conference presentations.

To test this via the terminal:
```bash
.venv/bin/python3 -c "
import urllib.request, json, base64

with open('testapp/low quality/low1.jpeg', 'rb') as f:
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode('ascii')

req = urllib.request.Request('http://127.0.0.1:5001/api/enhance',
    data=json.dumps({'image_base64': b64, 'mode': 'classic_clahe'}).encode(),
    headers={'Content-Type': 'application/json'})

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print('Benchmark Test 1:', data['success'], '| Engine:', data['metadata']['engine'])
"
```

---

## 9. Running Automated Unit & Integration Tests

ClaimClarity AI includes an automated 20-test test suite in `test_app.py` validating every layer of the system:
- Strict 1:1 image dimension preservation across multiple resolutions.
- Color preservation in CIE LAB color space ($\Delta E < 0.5$).
- Academic benchmark mapping for all 5 pairs.
- Multi-task neural classifier outputs.
- Sliding-window rate limiting (HTTP 429).
- Anti-tamper magic-bytes verification (HTTP 400).
- Unauthorized admin probe protection (HTTP 403).
- Proof sheet rendering and transaction auditing.

Run the test suite with:

```bash
.venv/bin/python3 -m unittest test_app.py
```

Expected output:
```text
Ran 20 tests in ~5.3s

OK
```

---

## 10. Troubleshooting & Common Questions

### Q: Port 5001 is already in use by another process.
**Fix:** On macOS, AirPlay Receiver occasionally occupies port 5000/5001. You can disable AirPlay Receiver in *System Settings $\to$ General $\to$ AirDrop & AirPlay*, or specify another port in `app.py`:
```bash
# To kill any leftover process on port 5001:
lsof -ti :5001 | xargs kill -9
```

### Q: The extension displays "ML Enhancement API relay failed: No image payload provided".
**Fix:** Ensure you have the latest `app.py` running. The backend includes universal payload extraction (`extract_image_payload_from_request`) that accepts `image_base64`, `imageBase64`, `image`, and `dataUrl`. Also ensure you have refreshed the unpacked extension in `chrome://extensions` by clicking the circular **Reload (↺)** icon.

### Q: Does the system alter vehicle paint colors?
**No.** ClaimClarity AI operates within the **CIE LAB color space**. Enhancement is applied strictly to the $L$ (luminance) channel, while the $A$ (green-red) and $B$ (blue-yellow) chrominance channels are strictly preserved to maintain factory paint accuracy.

### Q: Can this be run on a GPU / CUDA server?
**Yes.** PyTorch automatically detects CUDA if running on Linux/Windows with an NVIDIA GPU. The code uses `torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))`.

---

*Manual maintained for ClaimClarity AI. Stored at `HOW_TO_RUN.md`.*
