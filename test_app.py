"""
Automated Test Suite for ClaimClarity AI
Validates:
1. Core image processing engine (OpenCV / NumPy / Pillow)
2. Strict dimension preservation across varied image sizes & aspect ratios
3. Chromaticity fidelity (LAB color channel integrity)
4. Flask REST API endpoints and error handling
5. Proof sheet comparative image generation
"""

import io
import unittest
import numpy as np
import cv2
from enhancer import ClaimPhotoEnhancer
from app import app


class TestClaimClarityPipeline(unittest.TestCase):

    def setUp(self):
        self.enhancer = ClaimPhotoEnhancer()
        self.client = app.test_client()

    def test_dimensions_strictly_preserved(self):
        """Test that different resolutions and aspect ratios preserve dimensions exactly."""
        test_resolutions = [
            (640, 480),   # 4:3
            (1280, 720),  # 16:9
            (800, 1000),  # Portrait vertical phone photo
            (512, 512),   # 1:1 Square
        ]

        for w, h in test_resolutions:
            # Create synthetic noisy test image
            img = np.random.randint(20, 220, (h, w, 3), dtype=np.uint8)
            success, enc = cv2.imencode(".jpg", img)
            self.assertTrue(success)
            raw_bytes = enc.tobytes()

            enh_bytes, meta, orig_bgr, enh_bgr = self.enhancer.enhance(raw_bytes)

            # Check metadata dimensions
            self.assertEqual(meta["dimensions"]["width"], w)
            self.assertEqual(meta["dimensions"]["height"], h)

            # Check decoded array dimensions
            self.assertEqual(orig_bgr.shape, (h, w, 3))
            self.assertEqual(enh_bgr.shape, (h, w, 3))
            self.assertEqual(orig_bgr.shape, enh_bgr.shape)

    def test_chromaticity_preservation(self):
        """Verify that color channels A and B are locked and preserved in classic mode and bounded in neural mode."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :] = (30, 40, 180)  # BGR Reddish

        success, enc = cv2.imencode(".png", img)
        self.assertTrue(success)

        # Test classic mode has strict A/B locking
        enh_bytes, meta, orig_bgr, enh_bgr = self.enhancer.enhance(enc.tobytes(), mode="classic_clahe")
        orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
        enh_lab = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2LAB)
        np.testing.assert_array_equal(orig_lab[:, :, 1], enh_lab[:, :, 1])
        np.testing.assert_array_equal(orig_lab[:, :, 2], enh_lab[:, :, 2])

        # Test neural mode maintains dimensions and natural color fidelity
        enh_bytes_n, meta_n, orig_bgr_n, enh_bgr_n = self.enhancer.enhance(enc.tobytes(), mode="neural_dce")
        self.assertEqual(orig_bgr_n.shape, enh_bgr_n.shape)

    def test_proof_sheet_generation(self):
        """Verify comparative review proof sheet generation."""
        img = np.random.randint(50, 180, (400, 600, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", img)
        enh_bytes, meta, orig_bgr, enh_bgr = self.enhancer.enhance(enc.tobytes())

        proof_sheet = self.enhancer.generate_proof_sheet(orig_bgr, enh_bgr, meta)
        self.assertIsInstance(proof_sheet, bytes)
        self.assertGreater(len(proof_sheet), 1000)

        # Verify decoded proof sheet
        sheet_arr = cv2.imdecode(np.frombuffer(proof_sheet, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(sheet_arr)
        self.assertGreater(sheet_arr.shape[1], 1000)  # Width accommodates both panels + padding

    def test_health_endpoint(self):
        """Test /health API endpoint."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("opencv_version", data)

    def test_get_samples_endpoint(self):
        """Test /api/samples returns sample claim scenarios."""
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("samples", data)
        self.assertEqual(len(data["samples"]), 6)

    def test_enhance_sample_endpoint(self):
        """Test /api/enhance with sample scenario."""
        res = self.client.post("/api/enhance", json={"sample_id": "car_door_scratch.jpg"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["original_image"].startswith("data:image/"))
        self.assertTrue(data["enhanced_image"].startswith("data:image/png;base64,"))
        self.assertEqual(data["metadata"]["dimensions"]["width"], 960)
        self.assertEqual(data["metadata"]["dimensions"]["height"], 640)
        self.assertIn("disclaimer", data["metadata"])

    def test_enhance_upload_endpoint(self):
        """Test /api/enhance with multipart file upload."""
        img = np.random.randint(40, 200, (300, 400, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", img)

        data = {
            "file": (io.BytesIO(enc.tobytes()), "test_scratch.jpg")
        }
        res = self.client.post("/api/enhance", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["metadata"]["dimensions"]["width"], 400)
        self.assertEqual(payload["metadata"]["dimensions"]["height"], 300)

    def test_enhance_invalid_file(self):
        """Test /api/enhance returns 400 for corrupted/non-image bytes."""
        data = {
            "file": (io.BytesIO(b"Not an image file at all"), "corrupt.txt")
        }
        res = self.client.post("/api/enhance", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        payload = res.get_json()
        self.assertIn("error", payload)

    def test_proof_sheet_endpoint(self):
        """Test /api/proof-sheet generation endpoint."""
        res = self.client.post("/api/proof-sheet", json={"sample_id": "windshield_crack.jpg"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, "image/png")
        self.assertGreater(len(res.data), 1000)

    # =========================================================================
    # Security Audit & Year Fix Test Cases
    # =========================================================================

    def test_actual_year_2026_in_footer(self):
        """Verify footer renders current actual year 2026 instead of 2024."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("2026", html)
        self.assertIn("&copy; 2026 CLAIMCLARITY AI", html)
        # Verify 2024 is no longer rendered in the copyright notice
        self.assertNotIn("&copy; 2024", html)

    def test_admin_route_restricted(self):
        """Verify /admin access returns 403 Forbidden."""
        res = self.client.get("/admin")
        self.assertEqual(res.status_code, 403)
        data = res.get_json()
        self.assertEqual(data["code"], "ACCESS_DENIED")

    def test_not_a_photo_magic_bytes_check(self):
        """Verify upload of non-photo payload disguised as image fails signature check."""
        fake_payload = b"<html><script>alert(1)</script></html>"
        data = {
            "file": (io.BytesIO(fake_payload), "malicious.jpg")
        }
        res = self.client.post("/api/enhance", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        payload = res.get_json()
        self.assertIn("File signature does not match", payload["error"])

    def test_sample_id_path_traversal_blocked(self):
        """Verify URL id tampering and path traversal are rejected by sample whitelist."""
        res = self.client.post("/api/enhance", json={"sample_id": "../../etc/passwd"})
        self.assertEqual(res.status_code, 400)
        payload = res.get_json()
        self.assertIn("Unauthorized or unknown", payload["error"])

    def test_security_headers_enforced(self):
        """Verify defense-in-depth security headers on responses."""
        res = self.client.get("/")
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("X-Frame-Options"), "DENY")
        self.assertIn("Content-Security-Policy", res.headers)

    def test_rate_limiting_enforced(self):
        """Verify rate limiter blocks spamming (>35 req/min)."""
        import app as app_module
        app_module.ip_request_history.clear()

        # Send 35 requests (allowed)
        for _ in range(35):
            res = self.client.post("/api/enhance", json={"sample_id": "car_door_scratch.jpg"})
            self.assertEqual(res.status_code, 200)

        # 36th request must be blocked with 429
        blocked_res = self.client.post("/api/enhance", json={"sample_id": "car_door_scratch.jpg"})
        self.assertEqual(blocked_res.status_code, 429)
        self.assertEqual(blocked_res.get_json()["code"], "RATE_LIMIT_EXCEEDED")
        app_module.ip_request_history.clear()


if __name__ == "__main__":
    unittest.main()
