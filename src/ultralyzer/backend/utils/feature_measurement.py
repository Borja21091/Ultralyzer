import cv2
import numpy as np
import bottleneck as bn
from skimage import morphology
from scipy.spatial import KDTree
from skimage.draw import polygon
from backend.utils.graph import PathOrderer


def boxcount(Z, k):
    S = np.add.reduceat(np.add.reduceat(Z,
                        np.arange(0, Z.shape[0], k), axis=0),
                        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])

def fractal_dimension_boxcount(Z):

    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def sandbox(Z, pts, r):
    current_counts = []
    for y, x in pts:
        y_min = max(0, y - r)
        y_max = min(Z.shape[0], y + r + 1)
        x_min = max(0, x - r)
        x_max = min(Z.shape[1], x + r + 1)

        current_counts.append(np.sum(Z[y_min:y_max, x_min:x_max]))

    return np.mean(current_counts)

def fractal_dimension_sandbox(Z):
    pts = np.argwhere(Z)
    if len(pts) == 0:
        return 0

    if len(pts) > 2000:
        indices = np.random.choice(len(pts), 2000, replace=False)
        pts = pts[indices]

    p = min(Z.shape) // 2
    n = int(np.floor(np.log(p)/np.log(2)))
    sizes = 2**np.arange(1, n + 1)

    counts = []
    for r in sizes:
        counts.append(sandbox(Z, pts, r))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return coeffs[0]

############### REORDER COORDINATES ###############

def _remove_branching_points(skel: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Remove branching points from a skeletonized image.
    
    Parameters:
    ----------
    
        skel (numpy.ndarray): Binary skeletonized image where the skeleton is represented by 1s.
        kernel_size (int, optional): Size of the kernel used to count neighbours. Default is 3.
    
    Returns:
    -------
        numpy.ndarray: Skeletonized image with branching points removed.
    """
    
    # Define kernel
    kernel = np.ones((kernel_size, kernel_size))
    
    # Ensure binary image (0s and 1s only) as float
    skel = np.clip(skel, 0, 1, dtype=np.float32)
    
    # Count neighbours for each pixel
    im_filt = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    im_filt = im_filt * skel # Remove non-skeleton pixels
    
    # Identify branching points (3+ neighbours)
    branch_points = im_filt > 3
    
    # Remove branching points
    skel_nbp = skel * ~branch_points
    
    return skel_nbp

def _remove_small_branches(skel: np.ndarray, min_length: int = 10) -> np.ndarray:
    """
    Remove small branches from skeleton (sparse tensor).
    
    Parameters:
    ----------
    
        skel (np.ndarray): Labeled skeleton tensor without branching points of shape (H,W).
        min_length (int, optional): Minimum length of branch to keep. Default is 10.
    
    Returns:
    -------
        torch.Tensor: Skeleton tensor with small branches removed.
    """
    # Flatten to compute counts of each label
    flattened = skel.flatten()
    counts = np.bincount(flattened)

    # Ignore background label (often 0)
    counts[0] = 0

    # Boolean mask for labels to keep
    keep_mask = (counts >= min_length)

    # Create a mapping array that re-labels only the valid labels
    mapping = np.zeros_like(counts)
    labels_to_keep = np.nonzero(keep_mask)[0]
    mapping[labels_to_keep] = np.arange(1, labels_to_keep.size + 1)

    # Remap the original skeleton in one pass
    skel_nsb = mapping[skel]

    return skel_nsb

def generate_vessel_skeleton(vessels, od_mask, od_centre, min_length=10) -> list[np.ndarray]:
    
    # Remove optic disc
    vessels[od_mask > 0] = 0
    
    # Close vessels
    filt = morphology.disk(3)
    v_small = cv2.morphologyEx(vessels.astype(np.uint8), cv2.MORPH_CLOSE, filt)
    
    # Skeletonize using OpenCV
    v_skel_all = morphology.skeletonize(v_small)
    
    # Remove branching points
    v_skel = _remove_branching_points(v_skel_all)
    v_skel_labels = morphology.label(v_skel)
    
    # Remove small branches (less than 10 pixels)
    v_skel = _remove_small_branches(v_skel_labels, min_length)
    
    # Reorder coordinates w.r.t. OD center
    orderer = PathOrderer(v_skel, od_centre)
    coords_sorted = orderer.reorder_coords()
    
    return coords_sorted

############### TORTUOSITY ###############

def curve_length(x, y):
    return np.sum(((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2) ** 0.5)

def chord_length(x, y):
    return distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])

def distance_2p(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def detect_inflection_points(x, y):
    cf = np.convolve(y, [1, -1])
    inflection_points = []
    for i in range(2, len(x)):
        if np.sign(cf[i]) != np.sign(cf[i - 1]):
            inflection_points.append(i - 1)
    return inflection_points

def tortuosity_density(x, y, v_length):
    inflection_points = detect_inflection_points(x, y)
    n = len(inflection_points)
    if not n:
        return 0
    starting_position = 0
    sum_segments = 0
    
    # we process the curve dividing it on its inflection points
    for in_point in inflection_points:
        segment_x = x[starting_position:in_point]
        segment_y = y[starting_position:in_point]
        seg_curve = curve_length(segment_x, segment_y)
        seg_chord = chord_length(segment_x, segment_y)
        if seg_chord:
            sum_segments += seg_curve / seg_chord - 1
        starting_position = in_point

    # return ((n - 1)/curve_length)*sum_segments  # This is the proper formula
    return (n - 1)/n + (1/v_length)*sum_segments # This is not

############### VESSEL WIDTH ###############

def calculate_vessel_widths(mask, coords) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    
    # Refine coordinates
    coords_refined = refine_coords(coords) # dtype = np.int16
    
    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Get diameter of refined vessel skeleton
    vessel_map = np.zeros_like(dist, dtype=bool)
    r_c = np.concatenate(coords_refined, axis=0)
    vessel_map[r_c[:,0], r_c[:,1]] = True
    vessel_map = dist * vessel_map + 2.0
    
    # Calculate edges of the vessels
    edges1, edges2 = compute_vessel_edges(coords_refined, vessel_map)
    
    # Binary image of pixels within the edges
    mask_edges = np.zeros_like(mask, dtype=bool)
    for edge1, edge2 in zip(edges1, edges2):
        combined = np.vstack((edge1, edge2[::-1]))
        rr, cc = polygon(combined[:, 0], combined[:, 1], shape=mask.shape)
        mask_edges[rr, cc] = True
        
    # AND with segmentation mask
    mask_edges = mask_edges & (mask > 0)
    
    # Identify the edges of the vessels (Canny edge detection)
    mask_edges = cv2.Canny(mask_edges.astype(np.uint8), 0, 1)
    
    # Locate edges in the original image for each vessel
    on_pixels = np.argwhere(mask_edges).astype(np.float32)
    tree = KDTree(on_pixels)
    edges1 = [on_pixels[tree.query(e)[1]] for e in edges1]
    edges2 = [on_pixels[tree.query(e)[1]] for e in edges2]
    
    # Calculate vessel width at each point + average width
    widths = [np.linalg.norm(e1 - e2, axis=1) for e1, e2 in zip(edges1, edges2)]
    avg_width = [np.mean(w, dtype=float) for w in widths]
    
    return widths, r_c, avg_width

def refine_coords(coords: list[np.ndarray], dtype: type = np.int16):
    return [refine_path(c).astype(dtype) for c in coords]

def refine_path(data: np.ndarray, window: int = 4):
    # Simple moving average
    return bn.move_mean(data, window=window, axis=0, min_count=1)

def compute_vessel_edges(coords: list[np.ndarray], dist_map: np.ndarray):
    edges1 = []
    edges2 = []
    for path in coords:
        r, c = path[:,0], path[:,1]
        delta = np.gradient(path, axis=0)
        angles = np.arctan2(delta[:,1], delta[:,0])
        d = dist_map[r, c]
        offset_r = d * np.cos(angles + np.pi/2)
        offset_c = d * np.sin(angles + np.pi/2)
        r_edge1 = r + offset_r
        c_edge1 = c + offset_c
        r_edge2 = r - offset_r
        c_edge2 = c - offset_c
        edges1.append(np.stack([r_edge1, c_edge1], axis=1))
        edges2.append(np.stack([r_edge2, c_edge2], axis=1))
        
    return edges1, edges2


