import cv2
import numpy as np
import skimage.morphology as morphology
from scipy.ndimage import minimum_filter


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

