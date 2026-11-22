"""
ClaimClarity AI - Insurance Claim Photo Enhancement Web Application
Flask Backend Server - Deep Neural Edition
"""

import os
import base64
import io
import json
from flask import Flask, render_template, request, jsonify, send_file
from enhancer import ClaimPhotoEnhancer

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

enhancer = ClaimPhotoEnhancer()

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


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLES, model_meta=enhancer.train_metadata)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({"samples": SAMPLES})


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    return jsonify({
        "status": "ready",
        "device": str(enhancer.device),
        "model_architecture": "ClaimDCE-Net",
        "weights_path": enhancer.model_path,
        "metadata": enhancer.train_metadata
    })


@app.route("/api/enhance", methods=["POST"])
def enhance_photo():
    try:
        image_bytes = None
        source_name = "uploaded_photo.png"
        mode = "neural_dce"

        # Check for direct file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file was selected for upload."}), 400
            image_bytes = file.read()
            source_name = file.filename
            mode = request.form.get("mode", "neural_dce")
        # Check for sample selection
        elif request.is_json and "sample_id" in request.json:
            sample_id = request.json["sample_id"]
            sample_path = os.path.join(app.static_folder, "samples", os.path.basename(sample_id))
            if not os.path.exists(sample_path):
                return jsonify({"error": f"Sample image '{sample_id}' not found."}), 404
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
            source_name = sample_id
            mode = request.json.get("mode", "neural_dce")
        # Check for raw base64 upload
        elif request.is_json and "image_base64" in request.json:
            b64_data = request.json["image_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            mode = request.json.get("mode", "neural_dce")
        else:
            return jsonify({"error": "No image payload provided. Please upload a file or select a sample."}), 400

        if not image_bytes or len(image_bytes) == 0:
            return jsonify({"error": "Received empty image data."}), 400

        # Run refined neural enhancement
        enhanced_bytes, metadata, orig_bgr, enh_bgr = enhancer.enhance(image_bytes, mode=mode)

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
    except Exception as e:
        return jsonify({"error": f"Internal enhancement failure: {str(e)}"}), 500


@app.route("/api/proof-sheet", methods=["POST"])
def get_proof_sheet():
    try:
        image_bytes = None
        mode = "neural_dce"

        if "file" in request.files:
            file = request.files["file"]
            image_bytes = file.read()
            mode = request.form.get("mode", "neural_dce")
        elif request.is_json and "sample_id" in request.json:
            sample_id = request.json["sample_id"]
            sample_path = os.path.join(app.static_folder, "samples", os.path.basename(sample_id))
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
            mode = request.json.get("mode", "neural_dce")
        elif request.is_json and "image_base64" in request.json:
            b64_data = request.json["image_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            mode = request.json.get("mode", "neural_dce")
        else:
            return jsonify({"error": "No image payload provided."}), 400

        enhanced_bytes, metadata, orig_bgr, enh_bgr = enhancer.enhance(image_bytes, mode=mode)
        proof_sheet_bytes = enhancer.generate_proof_sheet(orig_bgr, enh_bgr, metadata)

        return send_file(
            io.BytesIO(proof_sheet_bytes),
            mimetype="image/png",
            as_attachment=True,
            download_name="ClaimClarity_Neural_ProofSheet.png"
        )
    except Exception as e:
        return jsonify({"error": f"Proof sheet generation failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    import cv2
    import numpy as np
    import torch
    return jsonify({
        "status": "healthy",
        "service": "ClaimClarity AI",
        "model": "ClaimDCE-Net",
        "trained_weights_loaded": enhancer.neural_model is not None,
        "device": str(enhancer.device),
        "pytorch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "dimension_preservation": "Strict 100%"
    })


if __name__ == "__main__":
    from generate_samples import create_sample_images
    create_sample_images(os.path.join(app.static_folder, "samples"))
    
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 ClaimClarity AI (Neural Edition) running at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
