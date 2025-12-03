import cv2
import numpy as np
from PIL import Image
from typing import Any
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from backend.models.database import DatabaseManager
from sklearn.preprocessing import PolynomialFeatures
from skimage.morphology import area_opening, skeletonize
from scipy.ndimage import rotate, distance_transform_edt
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.linear_model import RANSACRegressor, LinearRegression


class ArcadeDetector(object):
    
    def __init__(self, 
                 name: str, 
                 mask: np.ndarray, 
                 db_manager: DatabaseManager = None):
        self.name: str = name
        self.mask: np.ndarray = mask
        self.mask_original: np.ndarray = mask.copy()
        self.db_manager: DatabaseManager = db_manager or DatabaseManager()
        
        # Convert to 2D if image has more than 2 dimensions
        if len(self.mask.shape) > 2:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
    
    def __call__(self):
        
        # Crop mask to disc region
        self.crop_around_disc()
        
        # Rotate mask to vertical orientation
        try:
            self.rotate(-self.angle)
        except Exception as e:
            print(f"Could not rotate image {self.name} - {e}")
        
        # Remove small vessels by distance transform
        self.dist_transform()
        
        # Further remove small vessels by morphological area opening
        self.area_opening()
        
        # Detect parabola via circle Hough transform
        self.detect_parabola()
        
        # Remove small vessels by morphological area opening
        self.area_opening()
        
        # Image reconstruction via morphological closing using a rectangular kernel rotated between -70 & 70
        self.rectangular_closing()
        
        # Skeletonize the mask
        self.skeleton()
    
    @property
    def metrics(self):
        return self.db_manager.get_metrics_by_filename(self.name)
    
    @property
    def angle(self) -> float | Any:
        return self.metrics.disc_fovea_angle_deg if self.metrics else None
    
    @property
    def eye(self) -> str | Any:
        return self.metrics.laterality if self.metrics else None
    
    @property
    def disc_x(self) -> float | Any:
        return self.metrics.disc_center_x if self.metrics else None
    
    @property
    def disc_y(self) -> float | Any:
        return self.metrics.disc_center_y if self.metrics else None
    
    @property
    def disc_diameter(self) -> float | Any:
        return self.metrics.disc_diameter_px if self.metrics else None
    
    def view(self):
        plt.imshow(self.mask, cmap="gray")
        plt.axis("off")
        plt.show()
        
    def size(self):
        return np.shape(self.mask)
    
    ############ GETTER/SETTER ############
    
    def get_coordinates(self, mask: np.ndarray = None):
        """Get vessel pixel coordinates from the mask.
        
        Returns:
            coords: Nx2 array with (x, y) coordinates of vessel pixels"""
        if mask is None:
            mask = self.mask
        bool_array = np.where(mask > 0)
        coords = np.flip(np.column_stack(bool_array), axis=1)
        return coords
    
<<<<<<< Updated upstream
    ############ PREPROCESSING ############          
=======
    ############ PREPROCESSING ############
>>>>>>> Stashed changes

    def dist_transform(self, min_trigger_area=1000, min_trigger_num=2, threshold_quantile=0.99):
        areas = self._compute_areas()
        if sum(areas > min_trigger_area) >= min_trigger_num:
            dt_mask = np.asarray(distance_transform_edt(self.mask))
            thres = np.quantile(dt_mask, threshold_quantile)
            self.mask = ((dt_mask > thres)*255).astype(np.uint8)

    def rectangular_opening(self, min_trigger_area=1000, min_trigger_num=2, min_angle=-40, max_angle=40, num_angles=20):
        areas = self._compute_areas()
        if sum(areas > min_trigger_area) >= min_trigger_num:
            angles = np.linspace(min_angle, max_angle, num_angles)
            for angle in angles:
                kernel = np.ones((2,5),np.uint8) 
                kernel = rotate(kernel, angle)
                self.mask = cv2.erode(self.mask, kernel, iterations = 1)
                self.mask = cv2.dilate(self.mask, kernel, iterations = 1)
    
    def area_opening(self, min_trigger_area=1000, min_trigger_num=2, threshold_quantile=0.8, threshold_cap=300, connectivity=1):
        areas = self._compute_areas()
        if sum(areas > min_trigger_area) >= min_trigger_num:
            area_threshold_interim = np.quantile(areas, threshold_quantile)
            area_threshold = min(area_threshold_interim, threshold_cap)
            self.mask = area_opening(self.mask, area_threshold, connectivity=connectivity)
            
    def detect_parabola(self, min_trial_radius=5, max_trial_radius=15, binary_quantile=0.985):
        hough_radii = np.arange(min_trial_radius, max_trial_radius + 1, 1)
        hough_results = hough_circle(self.mask, hough_radii)
        # Select the most prominent circle
        _, _, _, radius = hough_circle_peaks(hough_results, hough_radii, total_num_peaks=1)
        
        self.mask = hough_circle(self.mask, radius[0])[0]
        binary_threshold = np.quantile(self.mask, binary_quantile)
        binary_mask = (self.mask > binary_threshold)*255
        self.mask = binary_mask.astype(np.uint8)
        
    def rectangular_closing(self, min_angle=-70, max_angle=70, num_angles=15):
        angles = np.linspace(min_angle, max_angle, num_angles)
        for angle in angles:
            kernel = np.ones((20,2),np.uint8) 
            kernel = rotate(kernel, angle)
            self.mask = cv2.dilate(self.mask, kernel, iterations = 1)
            self.mask = cv2.erode(self.mask, kernel, iterations = 1)
            
    def skeleton(self):
        self.mask = (skeletonize(self.mask)*255).astype(np.uint8)    
        
<<<<<<< Updated upstream
    def crop_around_disc(self, diameter_multiplier: float = 4):
        try:
            # Convert (col, row) to Cartesian (x, y)
            discX = self.disc_x
            discY = self.disc_y
            
            if self.eye == "left":
                top_left_x = discX - (self.disc_diameter / 2)
                top_left_y = discY - (self.disc_diameter * diameter_multiplier)
                bottom_right_x = discX + (self.disc_diameter * diameter_multiplier)
                bottom_right_y = discY + (self.disc_diameter * diameter_multiplier)
            elif self.eye == "right":
                top_left_x = discX - (self.disc_diameter * diameter_multiplier)
                top_left_y = discY - (self.disc_diameter * diameter_multiplier)
                bottom_right_x = discX + (self.disc_diameter / 2)
                bottom_right_y = discY + (self.disc_diameter * diameter_multiplier)
                
            # Crop mask
            mask_pil = Image.fromarray(self.mask)
            self.mask = np.array(mask_pil.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y)))
            
        except Exception as e:
            print("Error cropping around disc:", str(e))
            print("Make sure that disc_x, disc_y and disc_diameter are specified in the database for this image.")
            
=======
    def crop_around_disc(self, disc_x, disc_y, width_after_cropped=450):
        h, w = self.mask.shape
        nasal_width = 50
        temporal_width = width_after_cropped - nasal_width
        try:
            if self.eye == "left":
                disc_x = w*0.2 if (self.disc_x > w/2) or np.isnan(self.disc_x) else self.disc_x
                x_min = round(disc_x - nasal_width)
                x_max = round(disc_x + temporal_width)
            elif self.eye == "right":
                disc_x = w*0.8 if (self.disc_x < w/2) or np.isnan(self.disc_x) else self.disc_x
                x_min = round(disc_x - temporal_width)
                x_max = round(disc_x + nasal_width)
            x_min = 0 if x_min < 0 else x_min
            x_max = w if x_max > w else x_max
        except:
            raise ValueError("disc_x, disc_y and/or width_after_cropped not specified")
        self.disc_x = disc_x
        self.disc_y = disc_y
        self.mask = self.mask[:, x_min:x_max]
        
    def pad(self, pad_width =10):
        self.mask = np.pad(self.mask, 
                           [(0, 0), (pad_width , pad_width )], 
                           mode='constant')
        
>>>>>>> Stashed changes
    def crop_to_fovea(self, disc_x, disc_y, fovea_x, fovea_y, disc_temporal_width=50):
        h, w = self.mask.shape
        try:
            if self.eye == "left":
<<<<<<< Updated upstream
                disc_x = w*0.2 if (disc_x > w/2) or np.isnan(disc_x) else disc_x
                x_min = round(disc_x - disc_temporal_width)
                x_max = round(fovea_x) if ~np.isnan(fovea_x) else round(w/2)
            elif self.eye == "right":
                disc_x = w*0.8 if (disc_x < w/2) or np.isnan(disc_x) else disc_x
                x_min = round(fovea_x) if ~np.isnan(fovea_x) else round(w/2)
=======
                disc_x = width*0.2 if (disc_x > width/2) or np.isnan(disc_x) else disc_x
                x_min = round(disc_x - disc_temporal_width)
                x_max = round(fovea_x) if ~np.isnan(fovea_x) else round(width/2)
            elif self.eye == "right":
                disc_x = width*0.8 if (disc_x < width/2) or np.isnan(disc_x) else disc_x
                x_min = round(fovea_x) if ~np.isnan(fovea_x) else round(width/2)
>>>>>>> Stashed changes
                x_max = round(disc_x + disc_temporal_width)
            x_min = 0 if x_min < 0 else x_min
            x_max = w if x_max > w else x_max     
        except:
            raise ValueError("disc_x, disc_y, fovea_x and/or fovea_y not specified")  
        self.mask = self.mask[:, x_min:x_max]    
        
    def horizontal_cut(self, top):
        if top:
            top_seg = self.mask[0:round(self.disc_y),:]
            top_seg_flipped = cv2.flip(top_seg, 0)
            return np.concatenate((top_seg,top_seg_flipped))
        else:
            bottom_seg = self.mask[round(self.disc_y):,:]
            bottom_seg_flipped = cv2.flip(bottom_seg, 0)
            return np.concatenate((bottom_seg_flipped, bottom_seg))        
        
    def rotate(self, angle):
        """Rotate mask around its center by angle in degrees (positive values mean counter-clockwise rotation)."""
        mask_ui8 = self.mask.astype(np.uint8) * 255
        image_center = tuple(np.array(self.mask.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        self.mask = cv2.warpAffine(mask_ui8, rot_mat, self.mask.shape[1::-1], flags=cv2.INTER_LINEAR).astype(bool)
        
    ############ PRIVATE METHODS ############
    
    def _compute_areas(self):
<<<<<<< Updated upstream
        mask_ui8 = self.mask.astype(np.uint8) * 255
        output = cv2.connectedComponentsWithStats(mask_ui8, cv2.CV_32S)
=======
        # apply connected component analysis
        output = cv2.connectedComponentsWithStats(self.mask, cv2.CV_32S)
>>>>>>> Stashed changes
        (numLabels, labels, stats, centroids) = output
        areas = stats[:, cv2.CC_STAT_AREA][1:] # ignore the first element which corresponds to background pixels
        return areas


class ArcadeLS(ArcadeDetector):
<<<<<<< Updated upstream
    
        def __init__(self, mask):
            self.mask = mask
=======
        def __init__(self, name: str, mask: np.ndarray, db_manager: DatabaseManager = None):
            super().__init__(name=name, mask=mask, db_manager=db_manager)
            
>>>>>>> Stashed changes
            coords = self.get_coordinates()
            self.x = coords[:,0]
            self.y = coords[:,1]
            
        def quadratic_vertex_equation(self, y, concavity, vertex_y, vertex_x):
            # Quadratic function in vertex form
            return concavity*(y - vertex_y)**2 + vertex_x
        
        def fit(self):
            self.coefficients, _ = curve_fit(self.quadratic_vertex_equation, self.y, self.x, maxfev=1000)
            self.concavity, self.vertex_y, self.vertex_x = self.coefficients
            
        def predict(self, y_input):
            return self.quadratic_vertex_equation(y_input, *self.coefficients)
    
        def display_fit(self, mark_vertex=True):
            # define a sequence of inputs between the smallest and largest known inputs
            y_input = np.arange(min(self.y), max(self.y), 1)
            # calculate the output for the range
            x_predicted = self.predict(y_input)
            # Plot
            plt.imshow(self.mask, cmap='gray')
            plt.plot(self.x, self.y, '.', markersize=1.5, color="red")              
            show_boolean = (x_predicted <= self.mask.shape[1]) & (x_predicted > 0)
            plt.plot(x_predicted[show_boolean], y_input[show_boolean], '--', color="tomato")
            if mark_vertex:
                plt.scatter(self.vertex_x, self.vertex_y, marker="x", color="tomato", s=70)
            plt.axis('off')
            
        def _compute_metrics(self, decimal_place=4, verbose=True):
            residuals = self.x - self.predict(self.y)
            median_residual = np.median(abs(residuals))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.x-np.mean(self.x))**2)
            r_squared = 1 - (ss_res / ss_tot)
            r_squared = np.round(r_squared, decimal_place)
            if verbose:
                print("#### least square parabola ####", )
                print("Concavity index: {:.4f}".format(abs(self.concavity)))
                print("Median residual: {:.4f}".format(median_residual))
                print("R2: {:.4f}".format(r_squared))
            return abs(self.concavity), median_residual, r_squared
            
            
class ArcadeRANSAC(ArcadeDetector):
    
<<<<<<< Updated upstream
    def __init__(self, 
                 name: str, 
                 mask: np.ndarray, 
                 db_manager: DatabaseManager = None,
                 seed: int = 42):
        np.random.seed(seed)
        super().__init__(name, mask, db_manager)
    
    def __call__(self):
        
        # Mask filtering to isolate arcade vessels/shape
        super().__call__()
        
        # Get vessel coordinates
=======
    def __init__(self, name: str, mask: np.ndarray, db_manager: DatabaseManager = None):
        super().__init__(name=name, mask=mask, db_manager=db_manager)
        
>>>>>>> Stashed changes
        coords = self.get_coordinates()
        self.x = coords[:,0]
        self.y = coords[:,1]
        self.x_reshaped = self.x.reshape((-1, 1))
        self.y_reshaped = self.y.reshape((-1, 1))
        
        # Model fitting
        self.fit_parabola()
        
        self.display_parabola()
        
    ############ PARABOLA ############
    
    def fit_parabola(self):
        self.quadratic = PolynomialFeatures(degree=2)
        self.y_quadratic_trans = self.quadratic.fit_transform(self.y_reshaped)
        self.parabola = RANSACRegressor(LinearRegression(fit_intercept=False), min_samples=15)
        self.parabola.fit(self.y_quadratic_trans, self.x_reshaped)
        # get coefficients
        self.c = self.parabola.estimator_.coef_[0, 0] 
        self.b = self.parabola.estimator_.coef_[0, 1] 
        self.concavity = self.parabola.estimator_.coef_[0, 2]
    
    def display_parabola(self, mark_vertex=True):
        y_input = np.arange(0, self.mask.shape[0])
        y_input = y_input.reshape((-1,1))
        y_input_trans = self.quadratic.fit_transform(y_input)
        self.x_predicted = self.parabola.predict(y_input_trans)
        # Plot
        plt.imshow(self.mask, cmap='gray')
        show_line_boolean = (self.x_predicted <= self.mask.shape[1]) & (self.x_predicted > 0)
        plt.plot(self.x[self.parabola.inlier_mask_], 
                 self.y[self.parabola.inlier_mask_], 
                 '.', markersize=1.5, color='red')
        plt.plot(self.x[~self.parabola.inlier_mask_], 
                 self.y[~self.parabola.inlier_mask_], 
                 '.', markersize=1.5, color='green') 
        plt.plot(self.x_predicted[show_line_boolean], 
                 y_input.reshape((-1,1))[show_line_boolean], 
                 '--', color='orange') 
        if mark_vertex:
            vertex_x, vertex_y = self.get_parabola_vertex()
            plt.scatter(vertex_x, vertex_y, marker="x", color="orange", s=70)
        plt.axis('off')
        plt.savefig(f"{self.name}_arcade_ransac.png", bbox_inches='tight', pad_inches=0)
        
    def get_parabola_vertex(self):
        # vertex point of the predicted parabola
        vertex_y = -self.b / (2*self.concavity)
        vertex_x = self.concavity*vertex_y**2 + self.b*vertex_y + self.c
        return vertex_x, vertex_y   

    ############ LINEAR REGRESSION ############
     
    def get_inliers(self):
        try:
            x_inliers = self.x[self.parabola.inlier_mask_] 
            y_inliers = self.y[self.parabola.inlier_mask_] 
            return x_inliers, y_inliers
        except:
            raise NameError("Optic disc coordinates not found; parabola must first be fitted before calling this function")
    
    def segment_mask(self):
        x_inliers, y_inliers = self.get_inliers()
        self.vertex_x, self.vertex_y = self.get_parabola_vertex()
        ## First segment of the linear regression (from top to disc centre)
        first_segment_boolean = y_inliers > self.vertex_y
        self.x_inliers_1 = x_inliers[first_segment_boolean].reshape((-1, 1))
        self.y_inliers_1 = y_inliers[first_segment_boolean].reshape((-1, 1))
        ## Second segment of the linear regression (from disc centre to bottom)
        sec_segment_boolean = y_inliers <= self.vertex_y
        self.x_inliers_2 = x_inliers[sec_segment_boolean].reshape((-1, 1))
        self.y_inliers_2 = y_inliers[sec_segment_boolean].reshape((-1, 1))
    
    def fit_segmented_lr(self, fit_intercept):
        self.segment_mask()
        self.centre_factor_x = 0 if fit_intercept else self.vertex_x
        self.centre_factor_y = 0 if fit_intercept else self.vertex_y
        ## Fit the first segment
        if len(self.y_inliers_1) > 0:
            self.linreg1 = LinearRegression(fit_intercept=fit_intercept)
            self.linreg1.fit(self.y_inliers_1 - self.centre_factor_y, self.x_inliers_1 - self.centre_factor_x)
        ## Fit the second segment    
        if len(self.y_inliers_2) > 0:
            self.linreg2 = LinearRegression(fit_intercept=fit_intercept)
            self.linreg2.fit(self.y_inliers_2 - self.centre_factor_y, self.x_inliers_2 - self.centre_factor_x)
        
    def display_segmented_lr(self):
        centre_point = self.vertex_y - self.centre_factor_y
        ## Plot
        plt.imshow(self.mask, cmap='gray')
        # First segment
        if len(self.y_inliers_1) > 0:
            top_point = np.max(self.y_inliers_1 - self.centre_factor_y) 
            y_input_1 = np.arange(centre_point, top_point) 
            y_input_1 = y_input_1.reshape((-1, 1))
            x_predicted_1 = self.linreg1.predict(y_input_1)
            show_line_boolean = abs(x_predicted_1 + self.centre_factor_x) <= self.mask.shape[1]
            plt.plot(x_predicted_1[show_line_boolean] + self.centre_factor_x, 
                     y_input_1[show_line_boolean] + self.centre_factor_y, 
                     "--", color="gray")            
        # Second segment
        if len(self.y_inliers_2) > 0:
            bottom_point = np.min(self.y_inliers_2 - self.centre_factor_y) 
            y_input_2 = np.arange(bottom_point, centre_point) 
            y_input_2 = y_input_2.reshape((-1, 1))
            x_predicted_2 = self.linreg2.predict(y_input_2)             
            show_line_boolean = abs(x_predicted_2 + self.centre_factor_x) <= self.mask.shape[1]
            plt.plot(x_predicted_2[show_line_boolean] + self.centre_factor_x, 
                     y_input_2[show_line_boolean] + self.centre_factor_y, 
                     "--", color="gray") 
        # Plot inliers & outliers
        plt.plot(self.x[self.parabola.inlier_mask_], self.y[self.parabola.inlier_mask_], '.', markersize=1.5, color='red')   
        plt.plot(self.x[~self.parabola.inlier_mask_], self.y[~self.parabola.inlier_mask_], '.', markersize=1.5, color='green')  
        plt.axis('off')   
        
    ############ PRIVATE METHODS ############
    
    def _compute_metrics(self, model: str = "parabola", verbose=True):
        inliers_boolean = self.parabola.inlier_mask_
        if model == "parabola":
            self.x_predicted = self.parabola.predict(self.y_quadratic_trans)
            # Compute residuals (include only inliers)
            residuals = self.x_reshaped[inliers_boolean] - self.x_predicted[inliers_boolean]
            ## Concavity
            concavity = abs(self.concavity)
            ## Median absolute residual (superior & inferior arcades)
            vertex_x, vertex_y = self.get_parabola_vertex()
            # Superior arcade (include only inliers)
            top_seg = self.mask[0:round(vertex_y),:]
            top_coords = self.get_coordinates(top_seg)
            top_residuals = residuals[0:len(top_coords)]
            top_median_residual = np.median(abs(top_residuals))
            # Inferior arcade  (include only inliers)
            bottom_seg = self.mask[round(vertex_y):,:]
            bottom_coords = self.get_coordinates(bottom_seg)
            bottom_residuals = residuals[-len(bottom_coords):]
            bottom_median_residual = np.median(abs(bottom_residuals))            
            
        elif model == "linear":
            residuals = []
            # First segment
            if len(self.y_inliers_1) > 0:
                x_predicted = self.linreg1.predict(self.y_inliers_1 - self.centre_factor_y)
                x_predicted = x_predicted.flatten() + self.centre_factor_x
                residuals = residuals + list(self.x_inliers_1.flatten() - x_predicted)
            # Second segment
            if len(self.y_inliers_2) > 0:
                x_predicted = self.linreg2.predict(self.y_inliers_2 - self.centre_factor_y)
                x_predicted = x_predicted.flatten() + self.centre_factor_x
                residuals = residuals + list(self.x_inliers_2.flatten() - x_predicted)
            residuals = np.array(residuals)
        else:
            raise ValueError("model must be one of: 'parabola' or 'linear'")
            
        median_residual = np.median(abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.x-np.mean(self.x))**2)
        r_squared = 1 - (ss_res / ss_tot)
                
        if verbose: 
            outliers_boolean = np.logical_not(inliers_boolean)
            print("#### RANSAC", model, ":", "Excluded", sum(outliers_boolean), "outliers out of", 
                  sum(inliers_boolean)+sum(outliers_boolean), "vessel pixels ####")
            try:
                print("Concavity index: {:.4f}".format(abs(concavity)))
                print("Top median residual: {:.4f}".format(abs(top_median_residual)))
                print("Bottom median residual: {:.4f}".format(abs(bottom_median_residual)))
            except:
                pass            
            print("Median residual: {:.4f}".format(median_residual))
            print("R2: {:.4f}".format(r_squared))

        
        try:
            return concavity, median_residual, top_median_residual, bottom_median_residual, r_squared, 
        except:
            return median_residual, r_squared

    def _parabola_index(self):
        try:
            _, parabola_median_residual, parabola_r2 = self._compute_metrics(model="parabola", verbose=False)
            linear_median_residual, linear_r2 = self._compute_metrics(model="linear", verbose=False)
            median_residual_ratio = linear_median_residual / parabola_median_residual
            r2_ratio = parabola_r2 / linear_r2
            return median_residual_ratio, r2_ratio
        except:
            raise NotImplementedError("Both parabola and segmented linear models must be fitted before using this")


<<<<<<< Updated upstream
=======
class ArcadeAnalyzer(ArcadeDetector):
    pass


>>>>>>> Stashed changes
