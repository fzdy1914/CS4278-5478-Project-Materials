import logging
import numpy as np
import cv2
import intelligent_robots_project.trace_skeleton.trace_skeleton as trace_skeleton

from functools import reduce

logger = logging.getLogger(__name__)

class Perception():
    def __init__(self, visualise_camera=False, visualise_topdown=False):
        self.is_camera_on = visualise_camera
        self.is_topdown_on = visualise_topdown

        # Initialise matplotlib data for visualising topdown view if needed
        if self.is_topdown_on:
            logger.warning("Perception class is no longer responsible for visualization of top-down.")

        # For reference, the OpenCV images dealt with have the following coordinate system;
        # (0, 0) ----> x / cols
        # |
        # |
        # V  y / rows
        # The image is typically of size 640 x 480, and the coordinate system is left-handed.
        # This coincides with the handedness of the egocentric coordinate system of the vehicle in gym.
        # (To be precise, the gym coordinate system is right-handed, but y is used for elevation and the
        # ground plane corresponds to the x-z plane. The right-handed x-z coordinate system is equivalent 
        # to a left-handed x-y coordinate system.)

        # Compute perspective transform from hardcoded calibration values
        src_calib1 = np.array([[155, 109], [485, 109], [639, 145], [0, 145]], dtype="float32")
        src_calib2 = np.array([[158, 109], [482, 109], [639, 144], [0, 144]], dtype="float32")
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")
        self.image_to_ground_plane = cv2.getPerspectiveTransform(src_calib2, dst)
        self.ground_plane_to_image = np.linalg.pinv(self.image_to_ground_plane)
        logger.info("===== Image to ground plane transform =====")
        logger.info(self.image_to_ground_plane)
        logger.info("===== Ground plane to image transform =====")
        logger.info(self.ground_plane_to_image)

        # Compute rigid transform from the coordinate system of ground plane (matched to OpenCV coordinate system)
        # to the egocentric coordinate system. Again, calibration values are hardcoded.
        tx = 1.96 # x is ahead
        ty = -0.49 # y is to the right
        tphi = 89.0 * np.pi / 180.0
        self.T_perspective_to_ego = np.array([[np.cos(tphi), -np.sin(tphi), tx],
                                              [np.sin(tphi), np.cos(tphi), ty],
                                              [0, 0, 1]])
        logger.info("===== Rigid transform: perspective to egocentric =====")
        logger.info(self.T_perspective_to_ego)

        # Initialise hardcoded colour ranges for detecting yellow, red and white road markings
        self.yellow_lower_hsv = (26, 75, 20)
        self.yellow_upper_hsv = (34, 255, 255)
        self.red_lower_hsv = (165, 75, 20)
        self.red_upper_hsv = (179, 255, 255)
        self.white_lower_hsv = (0, 0, 125)
        self.white_upper_hsv = (179, 75, 255)

        self.duck_yellow_lower_hsv = (24, 75, 20)
        self.duck_yellow_upper_hsv = (36, 255, 255)

        # Initialise structuring elements for morphological ops
        self.yellow_structuring_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.white_structuring_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Initialise a lower bound for the ROI. The ROI is the region from the middle of the image to the
        # very bottom (so all near details are not missed), and the lower bound is where in the middle
        # we want to begin extracting the ROI from.
        self.roi_lower_bound = 120

        self.roi_hsv = None
        self.debug_features = None
        self.debug_image = None

        self.saved_white_mask = None
        self.init_features = None

    def _visualise_camera(self):
        cv2.imshow("Camera view", self.annotated_img)

    # Helper function; input is a contour, output is the centroid of the contour
    def _get_centroid(self, cont):
        M = cv2.moments(cont)
        if M["m00"] < 1e-9:
            return [np.inf, np.inf]
        else:
            return [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

    # Takes in a list of points [[p1x, p1y], ... ], or alternatively an np array of dimension N x 2
    def _transform_points_to_egocentric(self, points):
        points = np.squeeze(np.array(points))
        points = np.array([points]) if len(points.shape) == 1 else points
        points_h = np.concatenate((points, np.ones((len(points), 1))), axis=1).T
        points_warped = np.matmul(self.image_to_ground_plane, points_h)
        points_warped = points_warped / points_warped[-1,:]
        points_transformed = np.matmul(self.T_perspective_to_ego, points_warped)
        return points_transformed[:-1, :]

    def _process(self, observation, init):
        roi_bgr = observation[self.roi_lower_bound:, :, ::-1] # Reverse channel ordering because OpenCV uses BGR instead of RGB
        self.roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(self.roi_hsv, self.yellow_lower_hsv, self.yellow_upper_hsv)
        white_mask = cv2.inRange(self.roi_hsv, self.white_lower_hsv, self.white_upper_hsv)

        # Clean up the yellow line mask and generate a polyline from it
        yellow_mask_cleaned = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.yellow_structuring_elem)
        yellow_cont, yellow_hierarchy = cv2.findContours(yellow_mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellow_cont_pruned = []
        for i in range(len(yellow_cont)):
            if yellow_hierarchy[0, i, 3] == -1:
                yellow_cont_pruned.append(yellow_cont[i])
        yellow_centroids = list(map(self._get_centroid, yellow_cont_pruned))
        
        # Generate white mask polyline
        white_mask_cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, self.white_structuring_elem)
        white_mask_smoothed = cv2.morphologyEx(white_mask_cleaned, cv2.MORPH_CLOSE, self.white_structuring_elem)
        white_polys = trace_skeleton.from_numpy(white_mask_smoothed)

        if init:
            self.saved_white_mask = white_mask_cleaned

        # Project both yellow and white lines onto the top-down viewed ground plane
        if len(yellow_centroids) > 0:
            yellow_polyline_offset = np.array(yellow_centroids) + np.array([[0, self.roi_lower_bound]]) # offset by ROI lower bound to get original height
            yellow_polyline_h = np.concatenate((yellow_polyline_offset, np.ones((len(yellow_centroids), 1))), axis=1)
            yellow_line_transformed = np.matmul(self.image_to_ground_plane, yellow_polyline_h.T)
            projected = yellow_line_transformed / yellow_line_transformed[-1,:]
            projected_transformed = np.matmul(self.T_perspective_to_ego, projected)
            yellow_projected_transformed = projected_transformed[:-1, :] # Save this value to compute initialising features
            self.projected_yellow_line = None if projected_transformed.shape[1] <= 1 else projected_transformed[:-1,:]
        else:
            self.projected_yellow_line = None
        logger.debug("Perceived yellow line in local frame %s:\n", self.projected_yellow_line)

        if len(white_polys) > 0:
            white_polys_biggest = reduce(lambda l1, l2: l1 if len(l1) > len(l2) else l2, white_polys)
            white_polyline = np.squeeze(white_polys_biggest) + np.array([[0, self.roi_lower_bound]]) # offset by ROI lower bound to get original height
            white_polyline_h = np.concatenate((white_polyline, np.ones((white_polyline.shape[0], 1))), axis=1)
            white_line_transformed = np.matmul(self.image_to_ground_plane, white_polyline_h.T)
            projected = white_line_transformed / white_line_transformed[-1,:]
            projected_transformed = np.matmul(self.T_perspective_to_ego, projected)
            simplified = np.squeeze(cv2.approxPolyDP(np.float32(projected_transformed[:-1,:].T), 0.02, False)).T
            self.projected_white_line = None if len(simplified.shape) < 2 else simplified
        else:
            self.projected_white_line = None
        logger.debug("Perceived white line in local frame %s:\n", self.projected_white_line)

        # Generate red mask and look for initialising features
        out_img = roi_bgr.copy()
        red_mask = cv2.inRange(self.roi_hsv, self.red_lower_hsv, self.red_upper_hsv)
        red_cont, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.debug_features = None

        if len(red_cont) > 0:
            red_centroids = list(map(self._get_centroid, red_cont))
            red_areas = np.array([cv2.contourArea(c) for c in red_cont])
            max_idx = np.argmax(red_areas)
            largest_red_centroid = red_centroids[max_idx]

            if not np.any(np.isinf(largest_red_centroid)):
                hull = cv2.convexHull(red_cont[max_idx])
                hull_points = np.array(hull) + np.array([[0, self.roi_lower_bound]])
                hull_ego = self._transform_points_to_egocentric(hull_points)
                hull_sides = np.hstack((hull_ego[:,1:], np.expand_dims(hull_ego[:,0], axis=1))) - hull_ego
                hull_side_lengths = np.linalg.norm(hull_sides, axis=0)
                longest_hull_side = np.argmax(hull_side_lengths)
                longest_hull_vec = hull_sides[:, longest_hull_side]
                longest_hull_vec = longest_hull_vec / np.linalg.norm(longest_hull_vec)

                if self.projected_white_line is not None:
                    longest_white_line = np.argmax(np.linalg.norm(self.projected_white_line, axis=0))
                    longest_white_vec = self.projected_white_line[:, longest_white_line]
                    longest_white_vec = longest_white_vec / np.linalg.norm(longest_white_vec)

                    if abs(np.dot(longest_hull_vec, longest_white_vec)) < np.cos(np.pi / 6):
                        red_centroid = np.array(red_centroids[max_idx]) + np.array([0, self.roi_lower_bound])
                        red_centroid_transformed = self._transform_points_to_egocentric([red_centroid])
                        self.debug_features = (red_centroid_transformed, longest_hull_vec)
                        # cv2.drawContours(out_img, [red_cont[max_idx]], -1, (0, 255, 0), 5)
                        # cv2.circle(out_img, largest, 2, (0, 255, 0), -1)

        # cv2.imshow("Init image", out_img)

    def _process_lane_markers_init(self):
        out_img = self.saved_white_mask[120:, :].copy()
        white_cont, _  = cv2.findContours(self.saved_white_mask[120:, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.init_features = None
        if len(white_cont) > 0:
            max_idx = np.argmax(np.array([cv2.contourArea(c) for c in white_cont]))
            white_hull = cv2.convexHull(white_cont[max_idx])
            cv2.drawContours(out_img, [white_hull], -1, 125, 5)
            white_hull = np.squeeze(np.array(white_hull)).T
            white_hull[1] = list(map(lambda x: 239 - x, white_hull[1]))
            white_hull = np.array(white_hull)
            hull_sides = np.hstack((white_hull[:, 1:], np.expand_dims(white_hull[:,0], axis=1))) - white_hull
            hull_side_lengths = np.linalg.norm(hull_sides, axis=0)
            # print(hull_side_lengths)
            # print(white_hull)
            for i in range(white_hull.shape[1]):
                coords = (white_hull[0, i], 239 - white_hull[1, i])
                cv2.circle(out_img, coords, 7, 175, -1)
            longest_hull_side = np.argmax(hull_side_lengths)
            if hull_side_lengths[longest_hull_side] >= 50:
                # coord1 = white_hull[:, longest_hull_side]
                # coord2 = white_hull[:, longest_hull_side + 1]
                # cv2.circle(out_img, (coord1[0], 239 - coord1[1]), 10, 50, -1)
                # cv2.circle(out_img, (coord2[0], 239 - coord2[1]), 10, 50, -1)
                longest_hull_vec = hull_sides[:, longest_hull_side]
                longest_hull_vec = longest_hull_vec / np.linalg.norm(longest_hull_vec)
                self.init_features = (longest_hull_vec, white_hull)
        # cv2.imshow("Saved", out_img)

    def _process_duck(self, observation):
        # Search near the top of the image for the duck
        roi_duck_bgr = observation[:240, :, ::-1] # Reverse channel ordering because OpenCV uses BGR instead of RGB
        out_img = roi_duck_bgr.copy()
        roi_hsv = cv2.cvtColor(roi_duck_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(roi_hsv, self.duck_yellow_lower_hsv, self.duck_yellow_upper_hsv)
        yellow_mask_cleaned = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.yellow_structuring_elem)
        yellow_cont, _ = cv2.findContours(yellow_mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        duck_features = None
        if len(yellow_cont) > 0:
            areas = [cv2.contourArea(c) for c in yellow_cont]
            max_index = np.argmax(areas)
            biggest_contour = yellow_cont[max_index]
            duck_features = cv2.boundingRect(biggest_contour)
            x,y,w,h = duck_features
            cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.imshow("Duck image", out_img)
        return duck_features

    def step(self, observation, init=False, is_final=False):
        # Generate 3-DOF (x, y, heading) waypoints from observation along with status flag.
        # Status flag: 0 - found waypoints, 1 - no waypoints found, unable to see any lines
        self._process(observation, init)
        if init:
            self._process_lane_markers_init()
        if is_final:
            return self._process_duck(observation)
        if self.is_camera_on:
            self._visualise_camera()
        return self.projected_white_line, self.projected_yellow_line, self.init_features, self.debug_features

    def get_debug_features(self):
        return self.debug_features

    def get_debug_image(self):
        return self.debug_image
