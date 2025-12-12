import os
import numpy as np
from PIL import Image
from typing import Tuple, Any
from torchvision import tv_tensors
from abc import ABC, abstractmethod
from pathlib import PosixPath, PurePath
from backend.models.unet.model import UNetModel
from definitions import MODELS_DIR, MODEL_BASE_URL_UWF
from backend.utils.preprocessing import preprocess_image, get_bounding_box, find_vessels, get_mask, get_uwf_transform
from backend.utils.preprocessing import preprocess_uwf_disc_fov_loc_seg, process_uwf_disc_map, localise_centre_mass, process_uwf_fov_map

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch

from torchvision.transforms import v2 as T
from backend.models.models import SegmentationModel

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

class Segmentor(ABC):
    """Abstract base class for segmentation models"""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> Any:
        """
        Segment an image.
        
        Args:
            image: Input image (H, W, 3) as numpy array
        """
        pass
    
    def _extract_patches(self, image: torch.Tensor, patchsize: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patches from the image, including edge patches that are adjusted to fit boundaries.

        Args:
            image: Input image (1, C, H, W) as torch tensor
            patchsize: Size of the patches to extract

        Returns:
            Tuple of (patches, positions) where:
            - patches: (N, C, P, P) as torch tensor
            - positions: (N, 2) containing (row_start, col_start) for each patch
        """
        c = image.shape[1]
        h = image.shape[2]
        w = image.shape[3]
        stride = patchsize // 2
        
        # Calculate grid dimensions
        n_h = (h - patchsize) // stride + 1
        n_w = (w - patchsize) // stride + 1
        
        # Extract standard grid patches using unfold
        patches_unfolded = torch.nn.functional.unfold(image, kernel_size=patchsize, stride=stride)
        patches_grid = patches_unfolded.view(1, c, patchsize, patchsize, -1).permute(4, 1, 2, 3, 0).squeeze(-1)  # (n_h*n_w, C, P, P)
        
        # Generate positions for grid patches
        i_indices = torch.arange(n_h, device=image.device, dtype=torch.long)[:, None]
        j_indices = torch.arange(n_w, device=image.device, dtype=torch.long)[None, :]
        grid_i = (i_indices * stride).expand(n_h, n_w).reshape(-1)
        grid_j = (j_indices * stride).expand(n_h, n_w).reshape(-1)
        grid_positions = torch.stack([grid_i, grid_j], dim=1)  # (n_h*n_w, 2)
        
        # Extract edge patches
        # Bottom edge
        bottom_i = (h - patchsize) * torch.ones(n_w, device=image.device, dtype=torch.long)
        bottom_j = j_indices.squeeze(0) * stride
        bottom_positions = torch.stack([bottom_i, bottom_j], dim=1)  # (n_w, 2)
        bottom_patches = image[:, :, h-patchsize:h, :].unfold(3, patchsize, stride).permute(0, 1, 3, 4, 2).reshape(n_w, c, patchsize, patchsize)
        
        # Right edge
        right_i = i_indices.squeeze(1) * stride
        right_j = (w - patchsize) * torch.ones(n_h, device=image.device, dtype=torch.long)
        right_positions = torch.stack([right_i, right_j], dim=1)  # (n_h, 2)
        right_patches = image[:, :, :, w-patchsize:w].unfold(2, patchsize, stride).permute(0, 1, 2, 4, 3).reshape(n_h, c, patchsize, patchsize)
        
        # Bottom-right corner
        corner_patch = image[:, :, h-patchsize:h, w-patchsize:w]  # (1, C, P, P)
        corner_position = torch.tensor([[h - patchsize, w - patchsize]], device=image.device, dtype=torch.long)
        
        # Concatenate all patches and positions
        patches = torch.cat([patches_grid, bottom_patches, right_patches, corner_patch], dim=0)  # (N, C, P, P)
        positions = torch.cat([grid_positions, bottom_positions, right_positions, corner_position], dim=0)  # (N, 2)
        
        return patches, positions
    
    def _combine_outputs(self, outputs: torch.Tensor, positions: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Combine the model outputs into a single mask using patch positions (vectorized).

        Args:
            outputs: Model outputs (N, C, P, P) as torch tensor
            positions: Patch positions (N, 2) containing (row_start, col_start)
            image_size: Size of the original image (H, W)

        Returns:
            Segmentation mask (1, C, H, W) as torch tensor
        """
        h, w = image_size
        c = outputs.shape[1]
        p = outputs.shape[2]
        
        # Initialize mask and count
        mask = torch.zeros(1, c, h, w, device=outputs.device, dtype=outputs.dtype)
        count = torch.zeros(1, 1, h, w, device=outputs.device, dtype=outputs.dtype)
        
        # Vectorized placement: compute all end positions
        row_starts = positions[:, 0]
        col_starts = positions[:, 1]
        row_ends = torch.clamp(row_starts + p, max=h)
        col_ends = torch.clamp(col_starts + p, max=w)
        
        # Compute valid patch sizes (in case patches extend beyond image)
        patch_hs = row_ends - row_starts
        patch_ws = col_ends - col_starts
        
        # Use advanced indexing to place all patches at once
        for idx in range(outputs.shape[0]):
            r_start = row_starts[idx].item()
            r_end = row_ends[idx].item()
            c_start = col_starts[idx].item()
            c_end = col_ends[idx].item()
            p_h = patch_hs[idx].item()
            p_w = patch_ws[idx].item()
            
            mask[:, :, r_start:r_end, c_start:c_end] += outputs[idx:idx+1, :, :p_h, :p_w]
            count[:, :, r_start:r_end, c_start:c_end] += 1.0
        
        # Average overlapping regions
        mask = mask / (count + 1e-8)
        
        return mask


class UWFAVSegmentor(Segmentor):
    
    DEFAULT_MODEL_NAME = 'av_segmentation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)

    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF artery/vein segmentation model
        """
        super().__init__("uwf_av_segmentor", "1.0")
        self._patchsize = 256
        self._batch = 32
        
        self._threshold = threshold
        self.device = DEVICE
        self.model = UNetModel(n_channels=3, 
                               n_classes=4, 
                               kernel_size=3, 
                               mc=1024, bilinear=False).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF Artery/Vein segmentation has been loaded with GPU acceleration!")
        self.model.eval()

    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def patchsize(self):
        return self._patchsize
    
    @patchsize.setter
    def patchsize(self, value: int):
        self._patchsize = value
    
    @property
    def batch(self):
        return self._batch
    
    @batch.setter
    def batch(self, value: int):
        self._batch = value
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    def segment(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """Create segmentation mask for arteries and veins"""
        if isinstance(image, (str, PurePath, PosixPath)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        image, mask = preprocess_image(image)
        h, w = image.shape[:2]
        mask = mask.reshape(h, w, 1)
        
        t, b, l, r = get_bounding_box(mask)
        imout = np.zeros((4, h, w), dtype=np.float32)
        imout[:, t:b, l:r] = self._segment(image[t:b, l:r, :], 
                              patchsize=self.patchsize, 
                              use_softmax=True)

        vessel_mask, cmap = find_vessels(imout)

        return cmap, vessel_mask
    
    ############ PRIVATE METHODS ############
    
    def _segment(self, 
                 image: np.ndarray, 
                 patchsize: int = 512, 
                 use_softmax: bool = True,
                 batch: int = 32) -> np.ndarray:
        """
        Split the image into patches and apply the model to each patch.

        Args:
            image: Input image (H, W, C) as numpy array
            patchsize: Size of the patches to extract
            channels: Number of output channels

        Returns:
            Segmentation mask (C, H, W) as numpy array
        """
        h, w = image.shape[:2]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE) # (1, C, H, W)
        
        # Run segmentation in patches
        if patchsize > max(h, w):
            mask = self.model(image_tensor)
            
        else:
            # Extract patches from the image
            patches, positions = self._extract_patches(image_tensor, patchsize) # (N, C, P, P)
            
            # Apply the model to each patch based on batch size
            outputs = []
            for i in range(0, patches.shape[0], batch):
                batch_patches = patches[i:min(i+batch, patches.shape[0])]
                with torch.no_grad():
                    scores = self.model(batch_patches)
                outputs.append(scores)
            outputs = torch.cat(outputs, dim=0)
            # Combine the outputs into a single mask
            mask = self._combine_outputs(outputs, positions, image.shape[:2])
            
        if use_softmax:
            mask = torch.softmax(mask, dim=1)

        return mask.cpu().numpy().squeeze()


class UWFVesselSegmentor(Segmentor):
    
    DEFAULT_MODEL_NAME = 'vessel_segmentation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF binary vessel segmentation model
        """
        super().__init__("uwf_vessel_segmentor", "1.0")
        self._patchsize = 256
        self._batch = 32
        
        self._threshold = threshold
        self.device = DEVICE
        self.model = SegmentationModel('segformer', 'resnet34', in_channels=3).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("Binary vessel segmentation has been loaded with GPU acceleration!")
        self.model.eval()
        
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def patchsize(self):
        return self._patchsize
    
    @patchsize.setter
    def patchsize(self, value: int):
        self._patchsize = value
    
    @property
    def batch(self):
        return self._batch
    
    @batch.setter
    def batch(self, value: int):
        self._batch = value
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    def segment(self, image) -> np.ndarray:
        """Segment vessels in the image."""
        if isinstance(image, (str, PurePath, PosixPath)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        image = image.astype(np.float32) / 255.0

        mask = get_mask(image)
        h, w = image.shape[:2]
        mask = mask.reshape(h, w, 1)
        
        t, b, l, r = get_bounding_box(mask)
        imout = np.zeros((1, h, w), dtype=np.uint8)
        imout[:, t:b, l:r] = self._segment(image[t:b, l:r, :])
        
        return (None, imout.squeeze())

    ############ PRIVATE METHODS ############
    
    @torch.inference_mode()
    def _segment(self, image: np.ndarray, soft_pred=False):
        """
        Split the image into patches and apply the model to each patch.

        Args:
            image: Input image (H, W, C) as numpy array
            soft_pred: If True, return soft predictions. If False, return binary mask.

        Returns:
            Segmentation mask (C, H, W) as numpy array
        """
        h, w = image.shape[:2]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE) # (1, C, H, W)
        self.transform = get_uwf_transform(size=self.patchsize)
        
        # Run segmentation in patches
        if self.patchsize > max(h, w):
            mask = self.model(image_tensor)
            
        else:
            # Extract patches from the image
            patches, positions = self._extract_patches(image_tensor, self.patchsize) # (N, C, P, P)
            # Apply the model to each patch based on batch size
            outputs = []
            for i in range(0, patches.shape[0], self.batch):
                batch_patches = patches[i : min(i + self.batch, patches.shape[0])]
                with torch.no_grad():
                    scores = self.model(self.transform(batch_patches))
                outputs.append(scores)
            outputs = torch.cat(outputs, dim=0)
            # Combine the outputs into a single mask
            mask = self._combine_outputs(outputs, positions, image.shape[:2])
            
        if not soft_pred:
            mask = mask.sigmoid() >= self.threshold
            
        return mask.cpu().numpy().squeeze()


class UWFDiscLocaliser(Segmentor):
    
    DEFAULT_MODEL_NAME = 'od_localisation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF rough disc localisation.
        """
        super().__init__("uwf_disc_localiser", "1.0")
        self._patchsize = 512
        self._batch = 32
        self.transform = get_uwf_transform(size=(512, 512))
        self._threshold = threshold
        self.device = DEVICE
        self.model = SegmentationModel('segformer', 'resnet34', in_channels=1).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF disc localisation has been loaded with GPU acceleration!")
        self.model.eval()
        
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    def segment(self, img, soft_pred=False) -> Tuple:
        """Segment disc in the image.
        
        Returns:
        --------
            pred (np.ndarray): Segmentation mask
            loc (tuple): (y, x) coordinates of the disc centre in the original image"""
        if isinstance(img, (str, PurePath, PosixPath)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
            
        # Preprocess image
        img, tl = preprocess_uwf_disc_fov_loc_seg(img)
        img_shape = (img.height, img.width)
        
        # If downsamples to (1024, 1024), prepare for upsampling
        RESIZE = T.Resize(img_shape, antialias=True)
        
        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()[:, :M, :N]
            
            # Resize back to native resolution
            pred = RESIZE(tv_tensors.Image(pred))[0]
                
            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return (pred.cpu().numpy(), None)
            else:
                pred = (pred > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
                pred = process_uwf_disc_map(pred)
                loc = localise_centre_mass(pred) # Location in cropped image
                loc = (loc[0] + tl[0], loc[1] + tl[1]) # (row, col) -> (y, x) # Location in original image
                return (pred, loc)


class UWFDiscDetailedSegmenter(Segmentor):
    
    DEFAULT_MODEL_NAME = 'od_segmentation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF detailed disc segmentation.
        """
        super().__init__("uwf_disc_seg", "1.0")
        self.transform = get_uwf_transform(size=(256, 256))
        self._threshold = threshold
        self.device = DEVICE
        self.model = SegmentationModel('segformer', 'resnet34', in_channels=1).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF disc segmentation has been loaded with GPU acceleration!")
        self.model.eval()
        
    def __call__(self, x: Tuple):
        """Direct call for inference on single image"""
        return self.segment(*x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    @torch.inference_mode()
    def segment(self, img, od_centre: tuple[float], soft_pred=False):
        """
        Inference on a single image
        """
        if isinstance(img, (str, PurePath, PosixPath)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
            
        # Preprocess image
        img, _ = preprocess_uwf_disc_fov_loc_seg(img, centre=od_centre, crop_size=(256, 256))
        img_shape = (img.height, img.width)
        
        # If downsamples to (256, 256), prepare for upsampling
        if img_shape != (256, 256):
            RESIZE = T.Resize(img_shape, antialias=True)
        else:
            RESIZE = None
        
        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()[:, :M, :N]
            
            # Resize back to native resolution
            if RESIZE is not None:
                pred = RESIZE(tv_tensors.Image(pred))[0]
                
            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return pred.cpu().numpy()
            else:
                pred = (pred > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
                pred = process_uwf_disc_map(pred)
                return pred
    

class UWFDiscSegmentor(Segmentor):
    """Wrapper class for UWF disc segmentation combining localisation and detailed segmentation"""
    
    def __init__(self, 
                 localiser: UWFDiscLocaliser = None, 
                 segmenter: UWFDiscDetailedSegmenter = None):
        """
        Core inference class for UWF disc segmentation.
        """
        super().__init__("uwf_disc_full_seg", "1.0")
        self.localiser = localiser if localiser is not None else UWFDiscLocaliser()
        self.segmenter = segmenter if segmenter is not None else UWFDiscDetailedSegmenter()
    
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    ############ PUBLIC METHODS ############
    
    def segment(self, image) -> np.ndarray:
        """Segment disc in the image."""
        if isinstance(image, (str, PurePath, PosixPath)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        # First localise disc
        _, loc = self.localiser.segment(image)
        
        # Then segment disc in detail
        disc_mask = self.segmenter.segment(image, od_centre=loc)
        h, w = disc_mask.shape[:2]
        
        # Prepare output
        output = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        
        # Place disc mask in output
        output[int(loc[0]) - h//2 : int(loc[0]) + h//2, 
               int(loc[1]) - w//2 : int(loc[1]) + w//2] = disc_mask
        
        return output


class UWFFoveaLocaliser(Segmentor):
    
    DEFAULT_MODEL_NAME = 'fovea_localisation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF rough disc localisation.
        """
        super().__init__("uwf_fovea_localiser", "1.0")
        self.transform = get_uwf_transform(size=(512, 512))
        self._threshold = threshold
        self.device = DEVICE
        self.model = SegmentationModel('segformer', 'resnet34', in_channels=1).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF fovea localisation has been loaded with GPU acceleration!")
        self.model.eval()
        
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    @torch.inference_mode()
    def segment(self, img, soft_pred=False) -> Tuple:
        """
        Inference on a single image
        """
        if isinstance(img, (str, PurePath, PosixPath)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
        
        # Preprocess image
        img, tl = preprocess_uwf_disc_fov_loc_seg(img)
        img_shape = (img.height, img.width)
        
        # If downsamples to (1024, 1024), prepare for upsampling
        RESIZE = T.Resize(img_shape, antialias=True)
        
        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()[:, :M, :N]
            
            # Resize back to native resolution
            pred = RESIZE(tv_tensors.Image(pred))[0]
                
            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return (pred.cpu().numpy(), None, None)
            else:
                pred = (pred > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
                pred = process_uwf_fov_map(pred)
                loc = localise_centre_mass(pred) # Location in cropped image
                loc = (loc[0] + tl[0], loc[1] + tl[1]) # (row, col) -> (y, x) # Location in original image
                return (pred, loc, tl)


class UWFFoveaSegmentor(Segmentor):
    """Wrapper class for UWF fovea segmentation"""
    
    def __init__(self, 
                 localiser: UWFFoveaLocaliser = None):
        """
        Core inference class for UWF fovea segmentation.
        """
        super().__init__("uwf_fovea_full_seg", "1.0")
        self.localiser = localiser if localiser is not None else UWFFoveaLocaliser()
    
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    ############ PUBLIC METHODS ############
    
    def segment(self, image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Segment fovea in the image."""
        if isinstance(image, (str, PurePath, PosixPath)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        # First localise fovea
        pred, loc, tl = self.localiser.segment(image)
        h, w = pred.shape[:2]
        
        # Prepare output
        output = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        
        # Place fovea location in output
        output[tl[0]:tl[0] + h, tl[1]:tl[1] + w] = pred
        
        return (output, loc)
