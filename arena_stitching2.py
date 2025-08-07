import cv2
import numpy as np
import json
from arena_utils import warp_arena_frame, load_arena_settings

# ---------- Feature utilities ----------

def detect_and_match(imgA, imgB, max_feats=4000, ratio=0.75):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_feats, scaleFactor=1.2, nlevels=8)
    kpa, da = orb.detectAndCompute(grayA, None)
    kpb, db = orb.detectAndCompute(grayB, None)
    if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(da, db, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, None, None
    return kpa, kpb, good

def estimate_homography(kpa, kpb, matches, ransac_thresh=3.0):
    src = np.float32([kpa[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kpb[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, ransac_thresh)  # map imgB -> imgA
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers

# ---------- Blending utilities ----------

def make_canvas_and_warp(base, add, H):
    hA, wA = base.shape[:2]
    hB, wB = add.shape[:2]

    # corners of add in its own coords
    cornersB = np.float32([[0,0],[wB,0],[wB,hB],[0,hB]]).reshape(-1,1,2)
    # map to base coords
    mappedB = cv2.perspectiveTransform(cornersB, H)

    # corners of current base
    cornersA = np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2)

    all_pts = np.concatenate([cornersA, mappedB], axis=0)
    [xmin, ymin] = np.floor(all_pts.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(all_pts.max(axis=0).ravel()).astype(int)

    tx, ty = -xmin, -ymin
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)

    size = (xmax - xmin, ymax - ymin)
    base_warped = cv2.warpPerspective(base, T, size)
    add_warped  = cv2.warpPerspective(add,  T @ H, size)

    return base_warped, add_warped

def feather_blend(imgA, imgB):
    # Build masks
    maskA = (imgA.sum(axis=2) > 0).astype(np.uint8) * 255
    maskB = (imgB.sum(axis=2) > 0).astype(np.uint8) * 255

    overlap = cv2.bitwise_and(maskA, maskB)
    onlyA   = cv2.bitwise_and(maskA, cv2.bitwise_not(maskB))
    onlyB   = cv2.bitwise_and(maskB, cv2.bitwise_not(maskA))

    # Distance transform for smooth alpha in overlap
    # (avoid seaborn etc.; stick to cv2 + numpy)
    if np.any(overlap):
        distA = cv2.distanceTransform((overlap>0).astype(np.uint8), cv2.DIST_L2, 3)
        distB = cv2.distanceTransform((overlap>0).astype(np.uint8), cv2.DIST_L2, 3)
        wA = distA / (distA + distB + 1e-6)
        wB = 1.0 - wA
        wA = wA[..., None]
        wB = wB[..., None]
    else:
        # no overlap: stick to simple overlay
        wA = np.zeros((*imgA.shape[:2],1), dtype=np.float32)
        wB = np.ones((*imgB.shape[:2],1), dtype=np.float32)

    out = np.zeros_like(imgA, dtype=np.float32)
    out += (imgA.astype(np.float32) * (onlyA[...,None]/255.0))
    out += (imgB.astype(np.float32) * (onlyB[...,None]/255.0))
    out += (imgA.astype(np.float32) * wA * (overlap[...,None]/255.0))
    out += (imgB.astype(np.float32) * wB * (overlap[...,None]/255.0))
    return np.clip(out, 0, 255).astype(np.uint8)

def stitch_pair(base, add, min_inliers=30):
    kpa, kpb, good = detect_and_match(base, add)
    if good is None:
        return None, False
    H, inliers = estimate_homography(kpa, kpb, good)
    if H is None or inliers < min_inliers:
        return None, False
    A, B = make_canvas_and_warp(base, add, H)
    blended = feather_blend(A, B)
    return blended, True

def stitch_sequence(images):
    """Panorama-stitch a list of images left->right."""
    # Filter out None
    imgs = [im for im in images if im is not None]
    if not imgs:
        return None
    pano = imgs[0]
    for idx in range(1, len(imgs)):
        stitched, ok = stitch_pair(pano, imgs[idx])
        if not ok:
            # fallback: place side-by-side (resized to same height)
            h = pano.shape[0]
            right = cv2.resize(imgs[idx], (int(imgs[idx].shape[1]*h/imgs[idx].shape[0]), h))
            pano = np.hstack([pano, right])
        else:
            pano = stitched
    return pano

# ---------- Your existing camera code, with panorama stitching ----------

def open_all_cameras(settings):
    caps = {}
    for cell_key, cell in settings["cells"].items():
        cam_id = int(cell["camera"])
        if cam_id not in caps:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                caps[cam_id] = cap
            else:
                print(f"Failed to open camera {cam_id}")
    return caps

def get_frame_from_camera(caps, cam_id):
    cap = caps.get(cam_id)
    if cap:
        ret, frame = cap.read()
        if ret:
            return frame
    return None

def get_cell_frame(settings, caps, key):
    cell = settings["cells"].get(key)
    if not cell:
        return None
    cam_id = int(cell.get("camera", -1))
    raw = get_frame_from_camera(caps, cam_id)
    if raw is None:
        return None
    try:
        return warp_arena_frame(raw, cell_key=key)  # keep your per-cell warp if needed
    except Exception:
        return raw  # fallback: raw frame if warp fails

def stitch_arena(settings, caps):
    rows = settings["rows"]
    cols = settings["cols"]
    row_panos = []

    # 1) stitch each row left->right using panorama
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            key = f"{r},{c}"
            img = get_cell_frame(settings, caps, key)
            if img is None:
                # create a blank placeholder of typical size
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            row_imgs.append(img)
        row_pano = stitch_sequence(row_imgs)
        if row_pano is not None:
            row_panos.append(row_pano)

    # 2) stitch the row panoramas top->bottom (treat as another panorama, but vertically)
    if not row_panos:
        return None
    # To stitch vertically, rotate 90°, stitch left->right, rotate back:
    rot_rows = [cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE) for im in row_panos]
    rot_pano = stitch_sequence(rot_rows)
    if rot_pano is None:
        return None
    full_pano = cv2.rotate(rot_pano, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return full_pano

# --- Main Loop ---
if __name__ == "__main__":
    settings = load_arena_settings()
    caps = open_all_cameras(settings)

    if not caps:
        print("No cameras available")
        exit()

    try:
        while True:
            stitched = stitch_arena(settings, caps)
            if stitched is not None:
                cv2.imshow("Live Stitched Arena View", stitched)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
