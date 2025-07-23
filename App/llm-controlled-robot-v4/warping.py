import numpy as np
import cv2
from detection import detect_and_get_centroids, detect_objects

def warp_arena(frame, arena_corners, output_size):
    '''
    Original arena frame to perfect shaped arena
    '''
    if len(arena_corners) != 4:
        raise ValueError("arena_corners must contain exactly 4 points")

    w, h = output_size

    src = np.array(arena_corners, dtype=np.float32)

    def order_points(pts):
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],       # top-left
            pts[np.argmin(diff)],    # top-right
            pts[np.argmax(s)],       # bottom-right
            pts[np.argmax(diff)]     # bottom-left
        ], dtype=np.float32)

    src_ordered = order_points(src)

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_ordered, dst)
    warped = cv2.warpPerspective(frame, M, (w, h))
    return warped


def unwarp_points(warped_points, arena_corners, output_size):
    '''
    Warped points (perfect shaped arena) to Actual Points (original arena)
    '''
    if len(arena_corners) != 4:
        raise ValueError("arena_corners must contain exactly 4 points")

    h, w = output_size

    def order_points(pts):
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],       # top-left
            pts[np.argmin(diff)],    # top-right
            pts[np.argmax(s)],       # bottom-right
            pts[np.argmax(diff)]     # bottom-left
        ], dtype=np.float32)

    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    src_pts = order_points(arena_corners)
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)

    warped_points_np = np.array(warped_points, dtype=np.float32).reshape(-1, 1, 2)
    original_points = cv2.perspectiveTransform(warped_points_np, M).reshape(-1, 2)

    return [tuple(map(int, pt)) for pt in original_points]

def warp_points(unwarped_points, arena_corners, output_size):
    '''
    Actual Points (original arena) to Warped points (perfect shaped arena)
    '''
    if len(arena_corners) != 4:
        raise ValueError("arena_corners must contain exactly 4 points")

    h, w = output_size

    def order_points(pts):
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],       # top-left
            pts[np.argmin(diff)],    # top-right
            pts[np.argmax(s)],       # bottom-right
            pts[np.argmax(diff)]     # bottom-left
        ], dtype=np.float32)

    src_pts = order_points(arena_corners)
    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # Compute forward transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply transform to the input points
    unwarped_np = np.array(unwarped_points, dtype=np.float32).reshape(-1, 1, 2)
    warped_np = cv2.perspectiveTransform(unwarped_np, M).reshape(-1, 2)

    return [tuple(pt) for pt in warped_np]

# if __name__ == "__main__":
#     frame = cv2.imread("arena_img_test1.png")

#     arena_corners = detect_and_get_centroids(frame=frame, prompt="Blue Circles")
#     warped_size = (800, 800)

#     # Get warped image
#     warped_img = warp_arena(frame, arena_corners, warped_size)

#     list = ["A", "B", "Home"]
#     warped_pts = detect_objects(frame=warped_img, prompt_list=list)

#     # Unwarp back to original frame
#     orig_pts = unwarp_points(warped_pts, arena_corners, warped_size)
#     print("Unwarped points (original frame):", orig_pts)

#     # Draw points on warped image
#     for (x, y) in warped_pts:
#         cv2.circle(warped_img, (int(x), int(y)), 5, (0, 0, 255), -1)

#     # Draw points on original image
#     frame_with_pts = frame.copy()
#     for (x, y) in orig_pts:
#         cv2.circle(frame_with_pts, (int(x), int(y)), 5, (0, 255, 0), -1)

#     # Show both images
#     cv2.imshow("Warped Image with Points", warped_img)
#     cv2.imshow("Original Image with Unwarped Points", frame_with_pts)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    frame = cv2.imread("arena_img_test1.png")

    arena_corners = detect_and_get_centroids(frame=frame, prompt="Blue Circles")
    warped_size = (800, 800)

    # Get warped image
    warped_img = warp_arena(frame, arena_corners, warped_size)

    list = ["A", "B", "Home"]
    orig_pts = detect_objects(frame=frame, prompt_list=list)

    # Unwarp back to original frame
    warped_pts = warp_points(orig_pts, arena_corners, warped_size)
    print("Unwarped points (original frame):", orig_pts)

    # Draw points on warped image
    for (x, y) in warped_pts:
        cv2.circle(warped_img, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Draw points on original image
    frame_with_pts = frame.copy()
    for (x, y) in orig_pts:
        cv2.circle(frame_with_pts, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Show both images
    cv2.imshow("Warped Image with Points", warped_img)
    cv2.imshow("Original Image with Unwarped Points", frame_with_pts)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
