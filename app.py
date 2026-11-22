"""
ClaimClarity AI - Insurance Claim Photo Enhancement Web Application
Flask Backend Server - Hardened Deep Neural Edition
"""

import os
import base64
import io
import json
import time
import uuid
import logging
import threading
from collections import defaultdict
from typing import Tuple, Optional

from flask import Flask, render_template, request, jsonify, send_file
from enhancer import ClaimPhotoEnhancer

# Configure logging without disclosing sensitive paths
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ClaimClarity")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

enhancer = ClaimPhotoEnhancer()

# Concurrency throttling to protect GPU/MPS memory from flood attacks
INFERENCE_SEMAPHORE = threading.Semaphore(4)

# Tabular claims actuarial ledger cache
CLAIMS_CSV_PATH = os.path.join(os.path.dirname(__file__), "datasets", "claim-images", "archive", "insurance_claims_dataset.csv")
_claims_cache = None
_claims_cache_lock = threading.Lock()

def get_claims_cache():
    global _claims_cache
    with _claims_cache_lock:
        if _claims_cache is None:
            _claims_cache = []
            if os.path.exists(CLAIMS_CSV_PATH):
                import csv
                try:
                    with open(CLAIMS_CSV_PATH, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            _claims_cache.append(row)
                except Exception as e:
                    logger.error(f"Failed to load claims CSV: {e}")
        return _claims_cache

# In-memory sliding window rate limiter (35 requests / minute per client IP)
REQUEST_WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 35
ip_request_history = defaultdict(list)
rate_limit_lock = threading.Lock()


def is_rate_limited(client_ip: str) -> bool:
    now = time.time()
    with rate_limit_lock:
        history = ip_request_history[client_ip]
        # Prune timestamps older than window
        valid_history = [t for t in history if now - t < REQUEST_WINDOW_SECONDS]
        ip_request_history[client_ip] = valid_history
        if len(valid_history) >= MAX_REQUESTS_PER_WINDOW:
            return True
        ip_request_history[client_ip].append(now)
        return False


def validate_image_payload(image_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Strict server-side validation against 'not-a-photo' uploads, polyglots, and decompression bombs.
    Checks:
    - Minimum & maximum payload size
    - Magic bytes file signature (JPEG, PNG, WEBP)
    - File extension whitelist
    - Valid image dimensions (minimum 16x16, maximum 8192x8192 or 36MP)
    """
    if not image_bytes or len(image_bytes) < 16:
        return False, "Empty or truncated payload. Please upload a valid image."

    if len(image_bytes) > 32 * 1024 * 1024:
        return False, "File exceeds maximum permitted upload limit (32MB)."

    # Verify magic bytes file signature
    is_jpeg = image_bytes[:3] == b"\xff\xd8\xff"
    is_png = image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    is_webp = len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        return False, "Security validation failed: File signature does not match authentic JPEG, PNG, or WEBP photo."

    # Whitelist allowed file extensions
    ext = os.path.splitext(filename.lower())[1]
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        return False, f"Prohibited file extension '{ext}'. Only authentic .jpg, .jpeg, .png, and .webp photos are permitted."

    # Decompression bomb and corrupt stream check via Pillow
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            w, h = pil_img.size
            if w < 16 or h < 16:
                return False, "Image dimensions too small (minimum 16x16 pixels required)."
            if w > 8192 or h > 8192 or (w * h) > 36_000_000:
                return False, "Image exceeds maximum allowed dimensions (8192x8192 or 36MP max)."
    except Exception:
        return False, "Corrupted or unreadable image stream."

    return True, ""


# Immutable Claim Scenario Evidence Catalogue
SAMPLES = [
    {
        "id": "car_door_scratch.jpg",
        "title": "CASE 01 / DOOR PANEL SCRATCH",
        "category": "COLLISION / VANDALISM",
        "price": "$1,450 REPAIR ESTIMATE",
        "description": "Dark metallic door panel in dimly lit garage with fine scratch and high ISO sensor noise.",
        "path": "/static/samples/car_door_scratch.jpg",
        "badge": "LOW LIGHT & GRAIN",
        "ref": "CLAIM-REF-0891"
    },
    {
        "id": "windshield_crack.jpg",
        "title": "CASE 02 / WINDSHIELD SPIDER CRACK",
        "category": "COMPREHENSIVE GLASS",
        "price": "$620 REPLACEMENT",
        "description": "Stone impact point with radiating hairline fissures obscured by ambient haze and windshield glare.",
        "path": "/static/samples/windshield_crack.jpg",
        "badge": "HAZE & GLARE",
        "ref": "CLAIM-REF-0412"
    },
    {
        "id": "undercarriage_damage.jpg",
        "title": "CASE 03 / UNDERCARRIAGE CORROSION",
        "category": "STRUCTURAL / CHASSIS",
        "price": "$3,800 FRAME REBUILD",
        "description": "Wheel arch frame rail with severe shadow falloff, rust corrosion pitting, and sheared bolt head.",
        "path": "/static/samples/undercarriage_damage.jpg",
        "badge": "DEEP SHADOW FALLOFF",
        "ref": "CLAIM-REF-1109"
    },
    {
        "id": "hail_dent.jpg",
        "title": "CASE 04 / HOOD HAIL INDENTATIONS",
        "category": "WEATHER / CATASTROPHE",
        "price": "$2,150 PDR REPAIR",
        "description": "Metallic hood surface with subtle concave depressions under flat, overcast diffuse lighting.",
        "path": "/static/samples/hail_dent.jpg",
        "badge": "SUBTLE CURVATURE",
        "ref": "CLAIM-REF-2384"
    },
    {
        "id": "bumper_crack.jpg",
        "title": "CASE 05 / BUMPER FRACTURE & SCUFF",
        "category": "LOW-SPEED IMPACT",
        "price": "$980 BUMPER RESHELL",
        "description": "Plastic fascia impact split with abrasive road rash scuffs on lower valence skirt.",
        "path": "/static/samples/bumper_crack.jpg",
        "badge": "PLASTIC SPLIT & RASH",
        "ref": "CLAIM-REF-3051"
    },
    {
        "id": "vin_stamping.jpg",
        "title": "CASE 06 / OBSCURED VIN STAMPING",
        "category": "FRAUD & AUDIT CHECK",
        "price": "FORENSIC ID VERIFIED",
        "description": "Engine bay serial stamping obscured by oil residue, soot, and flash falloff shadow.",
        "path": "/static/samples/vin_stamping.jpg",
        "badge": "METAL STAMP AUDIT",
        "ref": "CLAIM-REF-4890"
    }
]

VALID_SAMPLE_IDS = {s["id"] for s in SAMPLES}
VALID_SAMPLE_IDS.update({
    "low1.jpeg", "low2.jpg", "low3.jpg", "low4.jpg", "low5.jpeg",
    "testapp/low1.jpeg", "testapp/low2.jpg", "testapp/low3.jpg", "testapp/low4.jpg", "testapp/low5.jpeg"
})


def resolve_sample_path(sample_id: str) -> Optional[str]:
    """Safely resolve sample path across static/samples, static/samples/testapp, and testapp."""
    clean_id = os.path.basename(sample_id)
    candidate_paths = [
        os.path.join(app.static_folder, "samples", clean_id),
        os.path.join(app.static_folder, "samples", "testapp", clean_id),
        os.path.join("testapp", "low quality", clean_id),
        os.path.join("testapp", "low", clean_id),
        os.path.join("testapp", "high quality", clean_id),
        os.path.join("testapp", "high", clean_id),
        os.path.join(app.static_folder, "samples", sample_id)
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            return p
    return None


@app.before_request
def handle_cors_preflight():
    """Handle CORS preflight requests for browser extension and client integrations."""
    if request.method == "OPTIONS" and request.path.startswith("/api/"):
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        return response


@app.after_request
def apply_security_headers(response):
    """Enforce defense-in-depth HTTP security headers."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://api.fontshare.com; "
        "font-src 'self' https://cdn.fontshare.com https://api.fontshare.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self';"
    )
    if request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLES, model_meta=enhancer.train_metadata, current_year=2026)


@app.route("/admin", methods=["GET", "POST"])
@app.route("/administrator", methods=["GET", "POST"])
@app.route("/wp-admin", methods=["GET", "POST"])
def admin_forbidden():
    """Defend against unauthorized administrative route access and URL probing."""
    client_ip = request.remote_addr
    logger.warning(f"Unauthorized administrative probe detected from IP {client_ip} on path {request.path}")
    return jsonify({
        "error": "Forbidden: Administrative access is restricted to verified claims compliance officers.",
        "code": "ACCESS_DENIED",
        "status": 403
    }), 403


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({"samples": SAMPLES})


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    return jsonify({
        "status": "ready",
        "device": str(enhancer.device),
        "model_architecture": "ClaimDCE-Net",
        "weights_path": "models/claim_dce_model.pth",
        "metadata": enhancer.train_metadata
    })


@app.route("/api/classifier-info", methods=["GET"])
def get_classifier_info():
    meta = dict(enhancer.classifier_metadata or {})
    if "classes" not in meta and "tasks" in meta and "damage_mode" in meta["tasks"]:
        meta["classes"] = meta["tasks"]["damage_mode"]["classes"]
    return jsonify({
        "status": "ready" if enhancer.classifier_model is not None else "initializing",
        "device": str(enhancer.device),
        "model_architecture": meta.get("model_architecture", "ClaimMultiTaskAssessor"),
        "weights_path": "models/multitask_assessor.pth",
        "metadata": meta
    })


@app.route("/api/claims/search", methods=["GET"])
def search_claims():
    client_ip = request.remote_addr or "127.0.0.1"
    if is_rate_limited(client_ip):
        return jsonify({"error": "Rate limit exceeded. Please wait a moment."}), 429

    query = request.args.get("q", "").strip().lower()
    claim_type = request.args.get("type", "").strip().lower()
    limit = min(int(request.args.get("limit", 15)), 50)

    claims = get_claims_cache()
    if not claims:
        return jsonify({"claims": [], "total_matches": 0, "total_dataset_records": 0})

    results = []
    for c in claims:
        if query:
            match = (
                query in c.get("Claim_ID", "").lower()
                or query in c.get("Policy_Number", "").lower()
                or query in c.get("Customer_ID", "").lower()
                or query in c.get("Claim_Type", "").lower()
                or query in c.get("Claim_Status", "").lower()
            )
            if not match:
                continue
        if claim_type and c.get("Claim_Type", "").lower() != claim_type:
            continue

        results.append(c)
        if len(results) >= limit:
            break

    return jsonify({
        "total_matches": len(results),
        "total_dataset_records": len(claims),
        "claims": results
    })


@app.route("/api/claims/<claim_id>", methods=["GET"])
def get_claim_detail(claim_id):
    claim_id_clean = os.path.basename(claim_id.strip()).upper()
    claims = get_claims_cache()
    for c in claims:
        if c.get("Claim_ID", "").upper() == claim_id_clean:
            return jsonify({"success": True, "claim": c})
    return jsonify({"error": f"Claim record '{claim_id}' not found in actuarial ledger."}), 404


def extract_image_payload_from_request(req) -> Tuple[Optional[bytes], str, str, Optional[str]]:
    """
    Universally extracts image bytes, source filename, mode, and any error message
    from multipart/form-data, JSON payloads, URL-encoded forms, or raw base64.
    Returns: (image_bytes, source_name, mode, error_message)
    """
    mode = "classic_clahe"
    source_name = "uploaded_photo.png"

    # 1. Check file uploads in req.files
    if req.files:
        uploaded_file = None
        for key in ["file", "image", "photo", "evidence", "upload"]:
            if key in req.files and req.files[key].filename:
                uploaded_file = req.files[key]
                break
        if not uploaded_file:
            first_file = next(iter(req.files.values()), None)
            if first_file and first_file.filename:
                uploaded_file = first_file

        if uploaded_file:
            raw_bytes = uploaded_file.read()
            is_valid, err_msg = validate_image_payload(raw_bytes, uploaded_file.filename or "upload.png")
            if not is_valid:
                return None, "", "", err_msg
            target_mode = req.form.get("mode") or req.args.get("mode") or mode
            return raw_bytes, os.path.basename(uploaded_file.filename), target_mode, None

    # 2. Extract from JSON, form, or query parameters
    json_data = req.get_json(silent=True) or {}
    form_data = req.form or {}
    combined = {}
    if isinstance(json_data, dict):
        combined.update(json_data)
    if form_data:
        combined.update(form_data.to_dict())

    target_mode = combined.get("mode") or req.args.get("mode") or mode

    # Check for sample_id scenario
    if "sample_id" in combined:
        sample_id = combined["sample_id"]
        if sample_id not in VALID_SAMPLE_IDS:
            return None, "", "", "Unauthorized or unknown claim scenario identifier."
        sample_path = resolve_sample_path(sample_id)
        if not sample_path or not os.path.exists(sample_path):
            return None, "", "", "Sample evidence file not found on disk."
        try:
            with open(sample_path, "rb") as f:
                return f.read(), sample_id, target_mode, None
        except Exception as e:
            return None, "", "", f"Failed to read sample file: {str(e)}"

    # Check for base64 image data under any key (snake_case or camelCase)
    b64_data = None
    for key in ["image_base64", "imageBase64", "image", "dataUrl", "data_url", "screenshot", "photo", "payload", "file_base64"]:
        if key in combined and combined[key]:
            b64_data = combined[key]
            break

    # Raw body fallback
    if not b64_data and req.data:
        try:
            raw_str = req.data.decode("utf-8", errors="ignore").strip()
            if raw_str.startswith("data:image/") or not raw_str.startswith("{"):
                b64_data = raw_str
        except Exception:
            pass

    if b64_data and isinstance(b64_data, str):
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        b64_data = b64_data.strip()
        try:
            raw_bytes = base64.b64decode(b64_data)
        except Exception:
            return None, "", "", "Malformed base64 image data."

        is_valid, err_msg = validate_image_payload(raw_bytes, "upload.png")
        if not is_valid:
            return None, "", "", err_msg
        return raw_bytes, source_name, target_mode, None

    return None, "", "", "No image payload provided. Please upload a file or select a scenario."


@app.route("/api/enhance", methods=["POST"])
def enhance_photo():
    client_ip = request.remote_addr or "127.0.0.1"

    # Rate limiting check: max 35 requests / minute
    if is_rate_limited(client_ip):
        logger.warning(f"Rate limit exceeded for client IP: {client_ip}")
        return jsonify({
            "error": "Rate limit exceeded. Maximum 35 enhancement requests per minute allowed.",
            "code": "RATE_LIMIT_EXCEEDED"
        }), 429

    try:
        image_bytes, source_name, mode, err_msg = extract_image_payload_from_request(request)
        if err_msg or not image_bytes:
            return jsonify({"error": err_msg or "No image payload provided. Please upload a file or select a scenario."}), 400

        # Concurrency throttle: protect GPU/MPS memory from simultaneous saturation
        acquired = INFERENCE_SEMAPHORE.acquire(timeout=10.0)
        if not acquired:
            return jsonify({"error": "Server is currently experiencing high processing volume. Please retry in a few seconds."}), 503

        try:
            enhanced_bytes, metadata, orig_bgr, enh_bgr = enhancer.enhance(image_bytes, mode=mode, source_name=source_name)
        finally:
            INFERENCE_SEMAPHORE.release()

        orig_mime = "image/jpeg" if source_name.lower().endswith((".jpg", ".jpeg")) else "image/png"
        orig_b64 = f"data:{orig_mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        enh_b64 = f"data:image/png;base64,{base64.b64encode(enhanced_bytes).decode('ascii')}"

        metadata["source_filename"] = source_name
        metadata["original_size_bytes"] = len(image_bytes)
        metadata["enhanced_size_bytes"] = len(enhanced_bytes)

        return jsonify({
            "success": True,
            "original_image": orig_b64,
            "enhanced_image": enh_b64,
            "metadata": metadata
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        event_ref = uuid.uuid4().hex[:8]
        logger.exception(f"Internal enhancement fault [ref: {event_ref}]")
        return jsonify({
            "error": f"An error occurred during evidence enhancement (Event Ref: {event_ref}). The incident has been recorded to the audit log."
        }), 500


@app.route("/api/proof-sheet", methods=["POST"])
def get_proof_sheet():
    client_ip = request.remote_addr or "127.0.0.1"
    if is_rate_limited(client_ip):
        return jsonify({"error": "Rate limit exceeded. Please wait a moment before exporting."}), 429

    try:
        image_bytes, source_name, mode, err_msg = extract_image_payload_from_request(request)
        if err_msg or not image_bytes:
            return jsonify({"error": err_msg or "No image payload provided."}), 400

        acquired = INFERENCE_SEMAPHORE.acquire(timeout=10.0)
        if not acquired:
            return jsonify({"error": "Server busy. Please try again in a few moments."}), 503

        try:
            enhanced_bytes, metadata, orig_bgr, enh_bgr = enhancer.enhance(image_bytes, mode=mode, source_name=source_name)
            proof_sheet_bytes = enhancer.generate_proof_sheet(orig_bgr, enh_bgr, metadata)
        finally:
            INFERENCE_SEMAPHORE.release()

        return send_file(
            io.BytesIO(proof_sheet_bytes),
            mimetype="image/png",
            as_attachment=True,
            download_name="ClaimClarity_Neural_ProofSheet.png"
        )
    except Exception:
        event_ref = uuid.uuid4().hex[:8]
        logger.exception(f"Internal proof sheet fault [ref: {event_ref}]")
        return jsonify({"error": f"Failed to generate proof sheet (Event Ref: {event_ref})."}), 500


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    import cv2
    import numpy as np
    import torch
    return jsonify({
        "status": "healthy",
        "service": "ClaimClarity AI",
        "model": "ClaimDCE-Net",
        "trained_weights_loaded": enhancer.neural_model is not None,
        "classifier_loaded": enhancer.classifier_model is not None,
        "device": str(enhancer.device),
        "pytorch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "dimension_preservation": "Strict 100%",
        "security": {
            "rate_limiting": "Active (35 req/min)",
            "concurrency_throttling": "Active (4 max)",
            "magic_bytes_validation": "Active (JPEG/PNG/WEBP)",
            "decompression_bomb_protection": "Active (36MP max)",
            "strict_sample_whitelisting": "Active",
            "security_headers": "Active (CSP, HSTS-ready, No-Sniff, Deny-Frame)",
            "information_disclosure_guard": "Active"
        }
    })


if __name__ == "__main__":
    from generate_samples import create_sample_images
    create_sample_images(os.path.join(app.static_folder, "samples"))
    
    port = int(os.environ.get("PORT", 5001))
    print(f"🛡️ ClaimClarity AI (Security Hardened) running at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
