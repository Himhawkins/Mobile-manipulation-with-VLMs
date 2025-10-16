import cv2
import numpy as np
import os
import threading

class ArenaProcessor:
    """Handles perspective warping and stitching of multi-camera feeds."""
    def __init__(self, settings: dict):
        self.settings = settings
        self.calibration_cache = {}  # Cache to store loaded calibration data (mtx, dist)
        self.cache_lock = threading.Lock() # Use a lock for thread-safe access to the cache
        self.homography_matrix_inv={}
        self.ft=True

    def _load_calibration(self, cam_id: int):
        """Loads and caches camera matrix and distortion coefficients."""
        with self.cache_lock:
            if cam_id in self.calibration_cache:
                return self.calibration_cache[cam_id]
        
        # Determine the file path based on your specified naming convention
        calib_file = f"vision/camera/camera_calibration_data_{cam_id}.npz"
        
        if not os.path.exists(calib_file):
            print(f"Warning: Calibration file '{calib_file}' not found for camera {cam_id}. Using uncalibrated frames.")
            return None, None

        try:
            data = np.load(calib_file)
            # Assuming your file keys are 'camera_matrix' and 'dist_coeffs'
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
            
            with self.cache_lock:
                self.calibration_cache[cam_id] = (camera_matrix, dist_coeffs)
            return camera_matrix, dist_coeffs
            
        except KeyError:
            print(f"Error: Keys 'camera_matrix' or 'dist_coeffs' missing in {calib_file}.")
            return None, None
        except Exception as e:
            print(f"Error loading calibration data from {calib_file}: {e}")
            return None, None

    def _undistort_frame(self, frame, cam_id):
        """Applies undistortion to a raw frame using cached calibration data."""
        mtx, dist = self._load_calibration(cam_id)
        
        if mtx is None or dist is None:
            return frame  # Return original frame if calibration is not available
        
        # Use cv2.undistort for simplicity and reliability, as previously discussed.
        # Passing None for the newCameraMatrix uses the original matrix.
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, None)
        return undistorted_frame
    
   

#    def init_steps(self,get_frame):
#         rows, cols = self.settings.get("rows", 1), self.settings.get("cols", 1)
#         grid, overlaps = [[] for _ in range(rows)], {}
#         raw_frames = {}
#         processed_frames = {}  # Cache to store undistorted frames
#         ff=[]
#         for r in range(rows):
#             for c in range(cols):
#                 key = f"{r},{c}"
#                 cell = self.settings.get("cells", {}).get(key, {})
#                 cam_id = int(cell.get("camera", -1))
                
#                 # 1. Get RAW frame from the source only once per camera ID
#                 if cam_id not in raw_frames:
#                     raw_frames[cam_id] = get_frame(cam_id)
                
#                 frame_raw = raw_frames[cam_id]

#                 # 2. Undistort the frame only once per camera ID
#                 if cam_id not in processed_frames:
#                     if frame_raw is not None:
#                         # Undistort the frame before any warping/stitching
#                         processed_frames[cam_id] = self._undistort_frame(frame_raw, cam_id)
#                         self.homography_matrix_inv[cam_id]=  np.linalg.inv(self.load_homography(cam_id))

#                     else:
#                         processed_frames[cam_id] = None

    #                 frame = processed_frames[cam_id]


    def create_merged_frame(self,images, homographies, scale_factor=0.4):
        """
        Creates a merged panoramic frame using Laplacian pyramid blending.

        This method provides a more seamless blend than simple alpha blending
        without requiring the opencv-contrib-python package. It helps reduce
        visible seams. This version includes fixes for shape mismatches.
        """
        if not images or len(images) != len(homographies):
            raise ValueError("Input lists must be non-empty and of the same length.")

        # --- Steps 1 & 2: Calculate canvas size and global transform (same as before) ---
        all_real_coords = []
        for img, H in zip(images, homographies):
            h, w = img.shape[:2]
            corners = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
            real_coords_homogenous = H @ corners
            real_coords_cartesian = real_coords_homogenous[:2] / real_coords_homogenous[2]
            all_real_coords.append(real_coords_cartesian)

        all_real_coords = np.hstack(all_real_coords)
        x_min, y_min = np.min(all_real_coords, axis=1)
        x_max, y_max = np.max(all_real_coords, axis=1)
        
        canvas_w = int(np.ceil(scale_factor * (x_max - x_min)))
        canvas_h = int(np.ceil(scale_factor * (y_max - y_min)))
        canvas_size = (canvas_w, canvas_h)
        
        T = np.array([
            [scale_factor, 0, -scale_factor * x_min],
            [0, scale_factor, -scale_factor * y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        # --- Step 3: Warp all images and create their masks (same as before) ---
        warped_images = []
        warped_masks = []
        for img, H in zip(images, homographies):
            H_final = T @ H
            warped_img = cv2.warpPerspective(img, H_final, canvas_size)
            warped_images.append(warped_img)
            
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, H_final, canvas_size)
            warped_masks.append(warped_mask)

        # --- Step 4: Laplacian Pyramid Blending (Iterative) ---
        merged_frame = warped_images[0]
        
        for i in range(1, len(warped_images)):
            img1 = merged_frame
            img2 = warped_images[i]

            mask1 = np.where(np.all(img1 > 0, axis=-1), 255, 0).astype(np.uint8)
            mask2 = warped_masks[i]
            overlap_mask = cv2.bitwise_and(mask1, mask2)
            
            if np.sum(overlap_mask) == 0:
                merged_frame = cv2.add(img1, img2)
                continue
            
            # This blending mask can be improved, but we'll focus on fixing the crash
            overlap_contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not overlap_contours:
                merged_frame = cv2.add(img1, img2)
                continue
            
            bounding_rect = cv2.boundingRect(np.vstack(overlap_contours))
            x, y, w, h = bounding_rect
            blend_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            
            # Create a linear gradient mask
            gradient = np.repeat(np.linspace(0.0, 1.0, w), h).reshape(w, h).T
            blend_mask[y:y+h, x:x+w] = gradient
            blend_mask = cv2.cvtColor(blend_mask, cv2.COLOR_GRAY2BGR)

            # --- Pyramid Blending Core Logic ---
            levels = 6
            gp1 = [img1.astype(np.float32)]
            gp2 = [img2.astype(np.float32)]
            gp_mask = [blend_mask]
            for j in range(levels - 1):
                gp1.append(cv2.pyrDown(gp1[j]))
                gp2.append(cv2.pyrDown(gp2[j]))
                gp_mask.append(cv2.pyrDown(gp_mask[j]))

            lp1 = [gp1[levels - 1]]
            lp2 = [gp2[levels - 1]]
            for j in range(levels - 1, 0, -1):
                size = (gp1[j - 1].shape[1], gp1[j - 1].shape[0])
                
                # FIX 1: Ensure upsampled pyramid levels match before subtraction
                up1 = cv2.pyrUp(gp1[j], dstsize=size)
                up2 = cv2.pyrUp(gp2[j], dstsize=size)
                
                lp1.append(cv2.subtract(gp1[j-1], up1))
                lp2.append(cv2.subtract(gp2[j-1], up2))
            
            lp1.reverse()
            lp2.reverse()

            LS = []
            for l1, l2, g_mask in zip(lp1, lp2, gp_mask):
                ls = l1 * (1.0 - g_mask) + l2 * g_mask
                LS.append(ls)

            blended_region = LS[0]
            for j in range(1, levels):
                size = (LS[j].shape[1], LS[j].shape[0])
                
                # FIX 2: Ensure upsampled blended region matches the next level before adding
                up_blended = cv2.pyrUp(blended_region, dstsize=size)
                
                blended_region = cv2.add(up_blended, LS[j])

            # --- Combine blended region with non-overlapping parts ---
            blended_region = np.clip(blended_region, 0, 255).astype(np.uint8)
            
            non_overlap1 = cv2.subtract(mask1, overlap_mask)
            non_overlap2 = cv2.subtract(mask2, overlap_mask)

            part1 = cv2.bitwise_and(img1, img1, mask=non_overlap1)
            part2 = cv2.bitwise_and(img2, img2, mask=non_overlap2)
            blended_part = cv2.bitwise_and(blended_region, blended_region, mask=overlap_mask)

            merged_frame = part1 + part2 + blended_part

        return merged_frame



    def create_merged_frame_ADV_BLEND(self, images, homographies, scale_factor=0.4):
        """
        Creates a merged panoramic frame using provided homography matrices 
        and incorporates OpenCV's advanced Multi-Band Blending (Pyramid Blending) 
        and Seam Finding to reduce ghosting and color spots.

        Args:
            images (list of np.ndarray): List of input camera images (H, W, 3).
            homographies (list of np.ndarray): 3x3 homography matrices.
            scale_factor (float): The scaling factor for the final canvas resolution.

        Returns:
            np.ndarray: The blended, merged panoramic image (uint8).
        """
        if not images or len(images) != len(homographies):
            raise ValueError("Input lists must be non-empty and of the same length.")

        # --- 1. Calculate Canvas Bounding Box & Global Transformation (YOUR LOGIC) ---
        all_real_coords = []
        for img, H in zip(images, homographies):
            h, w = img.shape[:2]
            corners = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
            real_coords_homogenous = H @ corners
            real_coords_cartesian = real_coords_homogenous[:2] / real_coords_homogenous[2]
            all_real_coords.append(real_coords_cartesian)
        
        all_real_coords = np.hstack(all_real_coords)
        x_min, y_min = np.min(all_real_coords, axis=1)
        x_max, y_max = np.max(all_real_coords, axis=1)

        canvas_w = int(np.ceil(scale_factor * (x_max - x_min)))
        canvas_h = int(np.ceil(scale_factor * (y_max - y_min)))
        canvas_size = (canvas_w, canvas_h)
        
        T = np.array([
            [scale_factor, 0, -scale_factor * x_min],
            [0, scale_factor, -scale_factor * y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        # --- 2. Warp Images and Masks (Initial Warping) ---
        final_H_list = []
        warped_images_f = []
        warped_masks_f = []
        
        for img, H in zip(images, homographies):
            H_final = T @ H
            final_H_list.append(H_final)

            # Convert to float and warp the image (0 to 1 range)
            img_f = img.astype(np.float32) / 255.0  
            warped_img = cv2.warpPerspective(img_f, H_final, canvas_size)
            warped_images_f.append(warped_img)
            
            # Create and warp a mask (0 to 1 range)
            mask = np.ones_like(img[:,:,0], dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, H_final, canvas_size)
            warped_masks_f.append(warped_mask.astype(np.float32) / 255.0)
        
        # --- 3. FIX: Seam Finding (Reduces Ghosting) ---
        # We must use the high-level 'create' access or an alternate finder, 
        # as direct instantiation of the full name fails in your environment.
        
        # Try the recommended GraphCut/Dijkstra first via the officially supported 'create' syntax
        # If the module is missing, fall back to a safer, simpler Seam Finder like Voronoi.
        try:
            # Most ideal Seam Finder for ghosting/parallax
            seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
        except Exception:
            # Fallback to a simpler seam finder that is more likely to be exposed.
            # This will still be better than your original hard-cut logic.
            print("Warning: GraphCutSeamFinder unavailable. Falling back to simpler seam finding.")
            seam_finder = cv2.detail.VoronoiSeamFinder()
        
        # Seam finder requires 8-bit masks
        warped_masks_u8 = [np.clip(m * 255, 0, 255).astype(np.uint8) for m in warped_masks_f]
        
        # Run the seam finder (masks are modified in place)
        corners = []
        for H_final in final_H_list:
            # Project the image origin (0, 0, 1) using the composite homography
            corner_homogenous = H_final @ np.array([0, 0, 1]).T
            
            # Convert from homogeneous to Cartesian (x, y) coordinates
            x = int(corner_homogenous[0] / corner_homogenous[2])
            y = int(corner_homogenous[1] / corner_homogenous[2])
            
            # OpenCV's SeamFinder expects a list of cv2.Point, or a tuple/list of (x, y) integers
            corners.append((x, y)) # Store as a tuple of (x, y) integers
        
        warped_images_f = [img.astype(np.float32) for img in warped_images_f]

        # 2. FIX: Ensure masks are a list of uint8, 1-channel arrays
        # The mask arrays must be (H, W), not (H, W, 1).
  
        seam_finder.find(warped_images_f, corners, warped_masks_u8) 
        print("Merging")    
        # --- 4. FIX: Blending (Reduces Color Spots/Patchy Colors) ---
        
        # MultiBand Blender is the best for color correction and smoothness.
        blender = cv2.detail.MultiBandBlender(num_bands=5)

        # Setup blender parameters (required by the API)
        corners = [np.array([0, 0], dtype=np.int32) for _ in range(len(warped_images_f))]
        sizes = [img.shape[:2][::-1] for img in images] 

        x_min = min(c[0] for c in corners)
        y_min = min(c[1] for c in corners)
        x_max = max(c[0] + s[0] for c, s in zip(corners, sizes))
        y_max = max(c[1] + s[1] for c, s in zip(corners, sizes))

        # Create the bounding box (Rect)
        dst_roi = (x_min, y_min, x_max - x_min, y_max - y_min)


        
        blender.prepare(dst_roi)

        # Feed the warped images (float) and the seamed masks (8-bit) into the blender
        for img_w, mask_w_seamed,corner in zip(warped_images_f, warped_masks_u8,corners):
            if img_w.dtype != np.uint8:
                img_w = cv2.convertScaleAbs(img_w)
            blender.feed(img_w, mask_w_seamed,tl=corner)
        print("Merging")
        # Finalize the blend
        result_pano_f, result_mask_u8 = blender.blend(None, None)

        # Scale back to 8-bit and clip to prevent weird/leaky colors
        final_pano_uint8 = np.clip(result_pano_f * 255.0, 0, 255).astype(np.uint8)
        
        return final_pano_uint8


    def create_merged_frameold2_ALPHABLEND(self,images, homographies, scale_factor=0.4):
        """
        Creates a merged panoramic frame using homography matrices and alpha blending.

        Args:
            images (list of np.ndarray): List of input camera images (H, W, 3).
            homographies (list of np.ndarray): 3x3 homography matrices (image -> real-life).
            scale_factor (float): The scaling factor for the final canvas resolution.

        Returns:
            np.ndarray: The blended, merged panoramic image.

        """
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA) # or cv2.Stitcher_SCANS
        stitcher.setPanoConfidenceThresh(0.1)
        
        status, pano = stitcher.stitch(images)
        return pano

        if not images or len(images) != len(homographies):
            raise ValueError("Input lists must be non-empty and of the same length.")

        # --- 1. Determine the Total Canvas Bounding Box in the Real-Life Frame ---
        all_real_coords = []
        for img, H in zip(images, homographies):
            h, w = img.shape[:2]
            corners = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
            real_coords_homogenous = H @ corners
            real_coords_cartesian = real_coords_homogenous[:2] / real_coords_homogenous[2]
            all_real_coords.append(real_coords_cartesian)

        all_real_coords = np.hstack(all_real_coords)
        x_min, y_min = np.min(all_real_coords, axis=1)
        x_max, y_max = np.max(all_real_coords, axis=1)

        # --- 2. Define the Global Transformation and Canvas Size ---
        
        # Apply the specified scale factor
        canvas_w = int(np.ceil(scale_factor * (x_max - x_min)))
        canvas_h = int(np.ceil(scale_factor * (y_max - y_min)))
        canvas_size = (canvas_w, canvas_h)
        
        # Global Transformation (T)
        T = np.array([
            [scale_factor, 0, -scale_factor * x_min],
            [0, scale_factor, -scale_factor * y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        # --- 3. Warp Images and Masks ---

        warped_images = []
        warped_masks = []

        for img, H in zip(images, homographies):
            # Calculate the final composite homography: H_final = T * H
            H_final = T @ H
            
            # Warp the image
            warped_img = cv2.warpPerspective(img, H_final, canvas_size)
            warped_images.append(warped_img)
            
            # Warp the mask (a white image of the original size)
            mask = np.ones_like(img[:,:,0], dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, H_final, canvas_size)
            warped_masks.append(warped_mask)


        # --- 4. Blending with cv2.addWeighted (Alpha Blending) ---

        # Start with the first warped image as the base
        merged_frame = warped_images[0]
        merged_mask = warped_masks[0]
        
        # Iterate through the remaining images and blend them onto the base
        for i in range(1, len(warped_images)):
            
            current_img = warped_images[i]
            current_mask = warped_masks[i]

            # 1. Find the **overlap region** mask: where both the merged image and current image are valid.
            overlap_mask = cv2.bitwise_and(current_mask, merged_mask)
            overlap_mask_3c = cv2.cvtColor(overlap_mask, cv2.COLOR_GRAY2BGR) # 3 channels for element-wise ops
            overlap_mask_3c = overlap_mask_3c.astype(np.float32)
            # 2. **Calculate a feathering mask** for the current image in the overlap zone.
            # This creates a gradient from 1.0 (center of current image) to 0.0 (edge of merged image)
            # We can use a simple linear gradient across the overlap width, or just a constant alpha.
            # A constant alpha=0.5 in overlap regions is a simple yet effective blending technique:
            
            # Convert images to float for precise blending
            merged_f = merged_frame.astype(np.float32)
            current_f = current_img.astype(np.float32)

            # 3. **Blend the overlap region**
            # We use a simple 50/50 blend in the overlap zone (alpha=0.5)
            blended_overlap = cv2.addWeighted(merged_f, 0.6, current_f, 0.3, 0)
            
            # 4. **Combine the parts to update the merged frame:**
            
            # a) Keep the non-overlap region of the existing merged frame
            non_overlap_merged_mask = cv2.bitwise_and(merged_mask, cv2.bitwise_not(current_mask))
            non_overlap_merged_mask_3c = cv2.cvtColor(non_overlap_merged_mask, cv2.COLOR_GRAY2BGR)
            non_overlap_merged_mask_3c = non_overlap_merged_mask_3c.astype(np.float32)
            
            # b) Keep the non-overlap region o`f the current image
            non_overlap_current_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(merged_mask))
            non_overlap_current_mask_3c = cv2.cvtColor(non_overlap_current_mask, cv2.COLOR_GRAY2BGR)
            non_overlap_current_mask_3c = non_overlap_current_mask_3c.astype(np.float32)
            # print("Merging",merged_f,"MASK",non_overlap_merged_mask_3c)
            # Update the new merged frame with: (Old Non-Overlap) + (Current Non-Overlap) + (Blended Overlap)
            merged_frame = cv2.bitwise_and(merged_f, non_overlap_merged_mask_3c)
            # print("Merging")
            merged_frame += cv2.bitwise_and(current_f, non_overlap_current_mask_3c)
            merged_frame += cv2.bitwise_and(blended_overlap, overlap_mask_3c)
            
            # 5. **Update the overall merged mask** (it's the union of the two masks)
            merged_mask = cv2.bitwise_or(merged_mask, current_mask)

        return merged_frame.astype(np.uint8)

    def create_merged_frameold(self,images, homographies):
        """
        Creates a merged panoramic frame from multiple images using homography matrices.

        Args:
            images (list of np.ndarray): List of input camera images.
            homographies (list of np.ndarray): List of 3x3 homography matrices, H_i,
                                                mapping image i coordinates to the common
                                                real-life reference frame.

        Returns:
            np.ndarray: The blended, merged panoramic image.
        """
        # print("MERGING ")
        if  images is None or  homographies is None or len(images) != len(homographies):
            raise ValueError("Input lists must be non-empty and of the same length.",len(images),len(homographies))
        
        # --- 1. Determine the Total Canvas Bounding Box in the Real-Life Frame ---
        
        # List to store the real-life coordinates of all image corners
        all_real_coords = []
        
        for i, (img, H) in enumerate(zip(images, homographies)):
            h, w = img.shape[:2]
            # Image corners in homogeneous coordinates
            corners = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
            
            

            # if self.ft:
            #     print("acorners:",corners)
            # Project corners to the real-life common reference frame
            # H is 3x3, corners is 3x4 -> result is 3x4
            real_coords_homogenous = H @ corners
            if self.ft:
                print("acorners:",corners)
                print("rc homog:",real_coords_homogenous)
            # Convert from homogeneous to Cartesian coordinates (divide by z/w)
            real_coords_cartesian = real_coords_homogenous[:2] / real_coords_homogenous[2]
            

            all_real_coords.append(real_coords_cartesian)

        if self.ft:
            print("acorners:",corners)
            print("all real coorrds:",all_real_coords)
            print("HOmographies:",homographies)

        # Convert to a single numpy array for easy min/max calculation
        all_real_coords = np.hstack(all_real_coords)
        
        # Find min and max bounds for the bounding box
        x_min, y_min = np.min(all_real_coords, axis=1)
        x_max, y_max = np.max(all_real_coords, axis=1)
        
        # --- 2. Define the Global Transformation to the Canvas ---
        
        # Determine the required canvas size (in pixels)
        # We use a scale factor S=1 for simplicity, assuming the real-life
        # frame coordinates are already in a reasonable unit (e.g., millimeters or pixels).
        # You can adjust this scale factor (S) if you need a higher or lower resolution.
        scale_factor = 0.4
        
        # Calculate dimensions
        canvas_w = int(np.ceil(scale_factor * (x_max - x_min)))
        canvas_h = int(np.ceil(scale_factor * (y_max - y_min)))
        canvas_size = (canvas_w, canvas_h)
        
        # Define the global transformation matrix (T) that scales and translates 
        # the real-life coordinates to the top-left (0,0) of the canvas.
        T = np.array([
            [scale_factor, 0, -scale_factor * x_min],
            [0, scale_factor, -scale_factor * y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        # --- 3. Warp and Combine the Images ---
        # print("MERGING ",images[0].shape)
        # Initialize the merged frame and a corresponding count map for blending
        merged_frame = np.zeros((*canvas_size[::-1], images[0].shape[2]), dtype=np.float32)
        count_map = np.zeros(canvas_size[::-1], dtype=np.float32)
        
        for img, H in zip(images, homographies):
            # Calculate the final composite homography: H_final = T * H
            H_final = T @ H
            
            # Warp the image onto the canvas
            # Note: cv2.warpPerspective uses the inverse homography internally, but 
            # the function takes the forward homography (H_final) as input.
            warped_img = cv2.warpPerspective(
                img.astype(np.float32), 
                H_final, 
                canvas_size
            )
            
            # Create a mask for the valid pixels in the warped image
            mask = (warped_img.sum(axis=2) > 0).astype(np.float32)
            # 
            # Add the warped image to the merged frame
            merged_frame += warped_img
            
            # Increment the count map where an image contributed
            count_map += mask
        
        # --- 4. Blending (Simple Average) ---

        # Avoid division by zero by setting zero counts to 1 (they won't affect the final image)
        count_map[count_map == 0] = 1 
        # print("MERGING hello",count_map == 0,"hello")
        # Normalize by the count map to get the average color in overlapping regions    
        # Expand count_map to 3 channels for element-wise division
        count_map_3d = np.dstack([count_map] * merged_frame.shape[2]) 
        
        merged_frame_blended = (merged_frame / count_map_3d).astype(np.uint8)
        # print("MERGING ")
        self.ft=False
        return merged_frame_blended
                    

    def stitch_arena(self, get_frame):
        """Stitches frames from all cameras into a single arena view."""
        rows, cols = self.settings.get("rows", 1), self.settings.get("cols", 1)
        grid, overlaps = [[] for _ in range(rows)], {}
        raw_frames = {}
        processed_frames = {}  # Cache to store undistorted frames
        ff=[]
        for r in range(rows):
            for c in range(cols):
                key = f"{r},{c}"
                cell = self.settings.get("cells", {}).get(key, {})
                cam_id = int(cell.get("camera", -1))
                
                # 1. Get RAW frame from the source only once per camera ID
                if cam_id not in raw_frames:
                    raw_frames[cam_id] = get_frame(cam_id)
                    
                frame_raw = raw_frames[cam_id]

                # 2. Undistort the frame only once per camera ID
                if cam_id not in processed_frames:
                    if frame_raw is not None:
                        # Undistort the frame before any warping/stitching
                        processed_frames[cam_id] = self._undistort_frame(frame_raw, cam_id)
                        if cam_id not in self.homography_matrix_inv.keys():
                            self.homography_matrix_inv[cam_id]=  self.load_homography(cam_id) #np.linalg.inv(self.load_homography(cam_id))
                    else:
                        processed_frames[cam_id] = None

                frame = processed_frames[cam_id]
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame = cv2.merge([frame, frame, frame])
                # Handle missing frames (e.g., camera failed or returned None)
                if frame is None:
                    print("lol none frame")
                    w, h = int(cell.get("width", 640)), int(cell.get("height", 480))
                    # Create a black frame as a placeholder
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                # 3. Proceed to perspective warping and stitching
                # grid[r].append(frame)
                k=self._warp_frame(frame, key,cam_id)
                # grid[r].append(k)
                ff.append(frame)
                # ff.append(self.get_mat(k,self.homography_matrix_inv[cam_id]))
                overlaps[key] = int(cell.get("overlap", 0))
        o=self.create_merged_frameold2_ALPHABLEND(ff,list(self.homography_matrix_inv.values()))
        # o = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
        return o,0,0

        try:
            stitched_rows = [np.hstack(row_images) for row_images in grid if row_images]
            stitched_full = np.vstack(stitched_rows) if stitched_rows else None
        except ValueError:
            stitched_full = None
            
        return stitched_full, grid, overlaps

    def load_homography(self,camera_id):
        filepath = f"vision/camera/homography_matrix_{camera_id}.npz"
        if not os.path.exists(filepath):
            print(f"Homography file '{filepath}' not found.")
            return None
        
        print(f"Loading homography matrix from {filepath}...")
        try:
            data = np.load(filepath)
            homography_matrix = data['homography']
            print("Homography matrix loaded successfully.")
            return homography_matrix
        except Exception as e:
            print(f"Error loading homography matrix: {e}")
            return None
    


    def _warp_frame(self, frame, cell_key,cam_id):
        """Performs perspective warping based on settings for a given cell."""
        if cam_id not in self.homography_matrix_inv.keys():
            self.homography_matrix_inv[cam_id]=  self.load_homography(cam_id) #np.linalg.inv(self.load_homography(cam_id))
            


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
        
        # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)


        # return cv2.warpPerspective(frame, self.homography_matrix_inv[cam_id], (out_w, out_h))
        return frame