import cv2
import torch
import numpy as np
from PIL import Image
import skimage.morphology as morphology
from scipy.ndimage import minimum_filter
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF


class FixShape(T.Transform):
    def __init__(self, factor=32):
        """Forces input to have dimensions divisble by 32"""
        super().__init__()
        self.factor = factor

    def __call__(self, img):
        M, N = img.shape[-2:]
        pad_M = (self.factor - M%self.factor) % self.factor
        pad_N = (self.factor - N%self.factor) % self.factor
        return TF.pad(img, padding=(0, 0, pad_N, pad_M)), (M, N)

    def __repr__(self):
        return self.__class__.__name__


def get_bounding_box(image: np.ndarray) -> tuple:
    """
    Gets the bounding box of the image.
    
    Parameters:
    ----------
    
        image (np.ndarray): The image to get the bounding box of.
        
    Returns:
    -------
    
        tuple: The top, bottom, left, and right coordinates of the bounding box.
    """
    # Get the coordinates of the non-zero pixels
    bin_im = np.where(image > 0)
    
    # If there are no non-zero pixels, return an empty bounding box
    if len(bin_im[0]) == 0:
        return [-1]*4
    
    # Get the bounding box coordinates
    top = np.min(bin_im[0])
    bottom = np.max(bin_im[0])
    left = np.min(bin_im[1])
    right = np.max(bin_im[1])
    
    return top, bottom, left, right

def get_mask(image: np.ndarray):
    """
    Generate a binary mask from the input image.
    This function creates a binary mask by summing the pixel values across the color channels
    and applying a threshold. It then erodes the mask to remove edge artifacts using binary
    closing and a minimum filter.
    
    Parameters:
    ----------
    
        image (np.ndarray): Input image as a NumPy array with shape (height, width, channels).
    
    Returns:
    -------
    
        np.ndarray: Binary mask with the same height and width as the input image.
    """
    mask = np.sum(image, axis = 2) > 0.
    t, b, l, r = get_bounding_box(mask)
    
    # Erode the mask to remove edge artifacts
    mask[t:b, l:r] = morphology.binary_closing(mask[t:b, l:r], morphology.disk(3))
    mask[t:b, l:r] = minimum_filter(mask[t:b, l:r]*1., size = 300)
    
    return mask

def preprocess_image(image: np.ndarray):
    """
    Preprocesses an image for segmentation.
    
    Parameters:
    ----------
    
        image (np.ndarray): The image to preprocess.
        
    Returns:
    -------
    
        np.ndarray: The preprocessed image.
        np.ndarray: The mask of the image.
    """
    # Mask
    mask = get_mask(image)
    
    # Convert image to HSV and equalize the V channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Open the green channel
    image[:,:,1] = cv2.morphologyEx(image[:,:,1], cv2.MORPH_OPEN, morphology.disk(3))

    return image, mask

def rough_crop(img: Image.Image, crop_size: tuple, centre: tuple = None) -> tuple:
    """
    Crop the image to the specified size.
    """
    width, height = img.size
    if centre is None:
        centre = (height // 2, width // 2) # (row, col)
        
    # Crop margins
    left = centre[1] - crop_size[1] // 2
    top = centre[0] - crop_size[0] // 2
    right = centre[1] + crop_size[1] // 2
    bottom = centre[0] + crop_size[0] // 2
    
    return img.crop((left, top, right, bottom)), (top, left)

def localise_centre_mass(map) -> tuple:
    """
    Find the centre of mass of the largest connected component in a binary map.
    
    Args:
        map (np.ndarray): Binary image.
        
    Returns:
        tuple: (y, x)(row, col) coordinates of the centre of mass.
    """
    # Find contours
    contours, _ = cv2.findContours(map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    
    if M['m00'] == 0:
        return None
    
    # Calculate centre of mass
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
    return (cY, cX)

# ------------------------- UWF ------------------------- #

def get_uwf_transform(size=(256, 256)):
    """Tensor, dimension and normalisation default augs for UWF Vessel Segmentation"""
    return T.Compose([
        T.PILToTensor(),
        T.Resize(size=size, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        FixShape(factor=32),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])

################### DISC ###################

def preprocess_uwf_disc_loc_seg(img: Image.Image, crop_size=(1024, 1024), centre=None) -> tuple:
    """
    Preprocess the UWF disc image for localisation.
    """
    # Split channels
    _, g, _ = img.split()

    # CLAHE for green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = Image.fromarray(clahe.apply(np.array(g))).convert('L')
    
    # Rough crop
    g_clahe_crop, tl = rough_crop(g_clahe, crop_size, centre)
    
    return g_clahe_crop, tl
    
def process_uwf_disc_map(map):
    """
    Post-process the UWF disc map.
    """
    # Keep largest connected component
    num_labels, labels_im = cv2.connectedComponents(map.astype(np.uint8), connectivity=8)
    largest_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
    map = np.zeros_like(map)
    map[labels_im == largest_label] = 1
    
    # Fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel)
    
    # Edge smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    map = cv2.morphologyEx(map, cv2.MORPH_OPEN, kernel)
    map = cv2.GaussianBlur(map, (5, 5), 0)
    _, map = cv2.threshold(map, 0.5, 1, cv2.THRESH_BINARY)
    map = map.astype(np.uint8) * 255
    
    return map

################### VESSEL ###################

def find_vessels(ypred: np.ndarray) -> tuple:
    """
    Find vessels in the predicted segmentation map.
    
    Parameters:
    ----------
        ypred (np.ndarray): The predicted segmentation map with shape (C, H, W).
        
    Returns:
    -------
        tuple: A tuple containing the vessel mask and the color map.
    """
    # Binarize based on vessel probability
    vessel_prob = ypred[0] + ypred[1] + ypred[2]
    vessel_mask = vessel_prob >= 0.3
    
    # Find the maximum probability
    ypred_array = np.amax(ypred[0:3], axis=0)
    
    # Create color map: set to 0 where channel doesn't have max probability
    cmap = np.where(ypred[0:3] == ypred_array, 255, 0).astype(np.uint8)
    
    # Apply vessel mask to all channels
    cmap *= vessel_mask
        
    return vessel_mask, cmap.transpose(1, 2, 0)

def process_uwf_vessel_map(vessel_map):
    """
    Process the UWF vessel map to remove small objects and fill holes.
    """
    # Remove small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_map = cv2.morphologyEx(vessel_map.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    vessel_map = cv2.morphologyEx(vessel_map, cv2.MORPH_CLOSE, kernel)
    
    return vessel_map

