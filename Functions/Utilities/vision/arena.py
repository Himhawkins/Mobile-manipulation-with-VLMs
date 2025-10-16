# vision_dashboard/vision/arena.py

import cv2
import numpy as np

class ArenaProcessor:
    """Handles perspective warping and stitching of multi-camera feeds."""
    def __init__(self, settings: dict):
        self.settings = settings

    def stitch_arena(self, get_frame):
        """Stitches frames from all cameras into a single arena view."""
        rows, cols = self.settings.get("rows", 1), self.settings.get("cols", 1)
        grid, overlaps = [[] for _ in range(rows)], {}
        raw_frames = {}

        for r in range(rows):
            for c in range(cols):
                key = f"{r},{c}"
                cell = self.settings.get("cells", {}).get(key, {})
                cam_id = int(cell.get("camera", -1))
                
                # Use a single consistent frame per camera for this stitch operation
                if cam_id not in raw_frames:
                    
                    raw_frames[cam_id] = get_frame(cam_id) if get_frame(cam_id) is not None else None

                frame = raw_frames[cam_id]
                if frame is None:
                    w, h = int(cell.get("width", 640)), int(cell.get("height", 480))
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                grid[r].append(self._warp_frame(frame, key))
                overlaps[key] = int(cell.get("overlap", 0))
        
        try:
            stitched_rows = [np.hstack(row_images) for row_images in grid if row_images]
            stitched_full = np.vstack(stitched_rows) if stitched_rows else None
        except ValueError:
            stitched_full = None
            
        return stitched_full, grid, overlaps

    def _warp_frame(self, frame, cell_key):
        """Performs perspective warping based on settings for a given cell."""
        cell = self.settings.get("cells", {}).get(cell_key)
        if not cell: return frame

        REF_W, REF_H = 800, 600
        rotation = cell.get("rotation", 0)
        out_w, out_h = int(cell.get("width", REF_W)), int(cell.get("height", REF_H))
        
        src_pts = np.array([cell.get(k, [0,0]) for k in ["topLeft","topRight","bottomRight","bottomLeft"]], dtype=np.float32)
        dst_pts = np.array([[0,0], [out_w,0], [out_w,out_h], [0,out_h]], dtype=np.float32)

        if rotation != 0:
            rot_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            frame = cv2.rotate(frame, rot_code[rotation])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(frame, matrix, (out_w, out_h))