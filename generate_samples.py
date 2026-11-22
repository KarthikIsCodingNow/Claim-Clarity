import os
import cv2
import numpy as np


def create_sample_images(output_dir="static/samples"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Door panel scratch in low garage lighting
    w, h = 960, 640
    img1 = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            base_val = 30 + 25 * np.sin(x / 300.0) + 15 * np.cos(y / 200.0)
            img1[y, x] = [base_val * 1.2, base_val * 0.7, base_val * 0.4]

    for x in range(w):
        crease_y = int(240 + 20 * np.sin(x / 150.0))
        if 0 <= crease_y < h:
            cv2.line(img1, (x, max(0, crease_y - 2)), (x, min(h - 1, crease_y + 2)), (80, 50, 30), 1)

    scratch_points = []
    curr_x, curr_y = 120, 360
    for i in range(70):
        curr_x += np.random.randint(8, 14)
        curr_y += np.random.randint(-3, 4)
        scratch_points.append((curr_x, curr_y))

    for i in range(len(scratch_points) - 1):
        pt1 = scratch_points[i]
        pt2 = scratch_points[i + 1]
        cv2.line(img1, pt1, pt2, (180, 190, 200), thickness=2, lineType=cv2.LINE_AA)
        if i % 5 == 0:
            branch = (pt2[0] + np.random.randint(-5, 6), pt2[1] + np.random.randint(6, 15))
            cv2.line(img1, pt2, branch, (130, 140, 150), 1, lineType=cv2.LINE_AA)

    shadow_mask = np.linspace(1.0, 0.4, h)[:, None]
    img1 = img1 * shadow_mask[:, :, None]
    noise1 = np.random.normal(0, 14, img1.shape)
    img1_noisy = np.clip(img1 + noise1, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "car_door_scratch.jpg"), img1_noisy, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 2. Windshield glass impact & spider crack with low-contrast haze
    img2 = np.ones((h, w, 3), dtype=np.float32) * 90
    cv2.rectangle(img2, (0, 320), (w, h), (55, 65, 60), -1)
    cv2.ellipse(img2, (480, 750), (600, 500), 0, 200, 340, (110, 115, 115), 18)

    center = (450, 280)
    cv2.circle(img2, center, 12, (230, 235, 240), -1)
    cv2.circle(img2, center, 24, (190, 200, 210), 2)
    cv2.circle(img2, center, 38, (160, 170, 180), 1)

    np.random.seed(42)
    for angle in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        curr = list(center)
        length = np.random.randint(60, 240)
        for step in range(length // 8):
            rad = angle + np.random.uniform(-0.15, 0.15)
            next_pt = (int(curr[0] + 8 * np.cos(rad)), int(curr[1] + 8 * np.sin(rad)))
            cv2.line(img2, tuple(curr), next_pt, (235, 240, 245), 1, cv2.LINE_AA)
            curr = next_pt

    glare = np.zeros((h, w, 3), dtype=np.float32)
    cv2.circle(glare, (380, 220), 250, (60, 60, 70), -1)
    glare = cv2.GaussianBlur(glare, (101, 101), 0)
    img2_noisy = np.clip(img2 * 0.75 + glare + np.random.normal(0, 8, img2.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "windshield_crack.jpg"), img2_noisy, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 3. Vehicle undercarriage / suspension rust & impact damage
    img3 = np.zeros((h, w, 3), dtype=np.float32)
    img3[:, :] = [25, 28, 26]
    cv2.line(img3, (100, 180), (860, 420), (45, 55, 60), 38)
    cv2.line(img3, (400, 260), (550, 580), (40, 50, 55), 30)

    cv2.ellipse(img3, (460, 280), (140, 70), 25, 0, 360, (20, 45, 110), -1)
    cv2.ellipse(img3, (460, 280), (110, 50), 25, 0, 360, (15, 35, 90), -1)
    rust_noise = np.random.normal(0, 22, (h, w, 3))
    rust_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(rust_mask, (460, 280), (140, 70), 25, 0, 360, 1.0, -1)
    img3 += (rust_noise * rust_mask[:, :, None])

    cv2.circle(img3, (620, 350), 22, (80, 95, 105), -1)
    cv2.circle(img3, (620, 350), 12, (30, 35, 40), -1)
    cv2.line(img3, (615, 335), (635, 365), (140, 150, 160), 3)

    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - 500)**2 + (y_coords - 320)**2)
    vignette = np.clip(1.0 - (dist_from_center / 550.0)**1.5, 0.2, 1.0)
    img3 = img3 * vignette[:, :, None]
    img3_noisy = np.clip(img3 + np.random.normal(0, 12, img3.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "undercarriage_damage.jpg"), img3_noisy, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 4. Hail damage dent on metallic car hood
    img4 = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        grad = 130 + 40 * np.cos(y / 120.0)
        img4[y, :] = [grad, grad * 0.98, grad * 0.95]

    cv2.ellipse(img4, (480, 200), (380, 120), -10, 0, 360, (175, 170, 165), -1)
    img4 = cv2.GaussianBlur(img4, (71, 71), 0)

    dent_centers = [(350, 260, 35), (550, 310, 45), (460, 420, 30)]
    for cx, cy, rad in dent_centers:
        cv2.circle(img4, (cx, cy), rad, (90, 88, 85), -1)
        cv2.circle(img4, (cx, cy + int(rad * 0.3)), int(rad * 0.8), (170, 165, 160), -1)

    img4 = cv2.GaussianBlur(img4, (31, 31), 0)
    flake_noise = np.random.normal(0, 6, img4.shape)
    img4_noisy = np.clip(img4 + flake_noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "hail_dent.jpg"), img4_noisy, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 5. Plastic bumper crack & paint scuff
    img5 = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        val = 45 + 15 * np.cos(y / 180.0)
        img5[y, :] = [val * 0.9, val * 0.9, val * 1.0]

    cv2.ellipse(img5, (480, 300), (420, 200), -5, 0, 180, (75, 75, 80), -1)
    img5 = cv2.GaussianBlur(img5, (51, 51), 0)

    pts = [(200, 220), (320, 290), (440, 340), (560, 380), (680, 410), (800, 430)]
    for i in range(len(pts) - 1):
        cv2.line(img5, pts[i], pts[i+1], (15, 15, 18), 3, cv2.LINE_AA)
        cv2.line(img5, (pts[i][0]+1, pts[i][1]+1), (pts[i+1][0]+1, pts[i+1][1]+1), (180, 175, 170), 1, cv2.LINE_AA)

    for _ in range(120):
        sx = np.random.randint(300, 680)
        sy = np.random.randint(360, 520)
        cv2.line(img5, (sx, sy), (sx + np.random.randint(10, 40), sy + np.random.randint(-4, 4)), (150, 150, 155), 1)

    noise5 = np.random.normal(0, 10, img5.shape)
    img5_out = np.clip(img5 + noise5, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "bumper_crack.jpg"), img5_out, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 6. VIN Tag / Stamped serial plate in engine bay
    img6 = np.zeros((h, w, 3), dtype=np.uint8)
    img6[:, :] = [24, 26, 24] # Dark engine compartment
    cv2.rectangle(img6, (220, 180), (740, 460), (70, 75, 80), -1)
    cv2.rectangle(img6, (220, 180), (740, 460), (130, 135, 140), 2)
    # Rivet mounts
    cv2.circle(img6, (250, 210), 10, (160, 165, 170), -1)
    cv2.circle(img6, (710, 210), 10, (160, 165, 170), -1)
    cv2.circle(img6, (250, 430), 10, (160, 165, 170), -1)
    cv2.circle(img6, (710, 430), 10, (160, 165, 170), -1)
    # Stamped text
    cv2.putText(img6, "CHASSIS / VIN: 1HGCR2F83HA029", (260, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (45, 50, 55), 2, cv2.LINE_AA)
    cv2.putText(img6, "SPEC: FED-MOTOR-STD-208 / COMP", (260, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 55, 60), 2, cv2.LINE_AA)
    cv2.putText(img6, "PAINT CODE: NH-731P (NIGHTHAWK)", (260, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 55, 60), 2, cv2.LINE_AA)

    # Add dark oil smudge & shadow
    f6 = img6.astype(np.float32)
    smudge = np.zeros((h, w, 3), dtype=np.float32)
    cv2.ellipse(smudge, (460, 320), (180, 90), 20, 0, 360, (30, 25, 20), -1)
    smudge = cv2.GaussianBlur(smudge, (51, 51), 0)
    f6 = np.clip(f6 - smudge + np.random.normal(0, 9, f6.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "vin_stamping.jpg"), f6, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print("Generated 6 realistic sample images in", output_dir)


if __name__ == "__main__":
    create_sample_images()
