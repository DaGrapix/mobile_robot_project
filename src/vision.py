import cv2
import numpy as np
from typing import List

# Code inspired from https://www.geeksforgeeks.org/perspective-transformation-python-opencv/ for the perspective transformation.

class VisionDetector:
    def __init__(self, initial_image: np.ndarray, map_size: List[int], **kwargs):
        """
        Initialize the ArUco detector. An initial image is required to initialize the perspective correction.
        """
        self.initial_image = initial_image
        self.map_size = map_size

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, kwargs.get("aruco_dict", "DICT_6X6_250")))
        self.marker_size = kwargs.get("marker_size", 200)
        self.top_left_id = kwargs.get("top_left_id", 0)
        self.top_right_id = kwargs.get("top_right_id", 1)
        self.bottom_left_id = kwargs.get("bottom_left_id", 2)
        self.bottom_right_id = kwargs.get("bottom_right_id", 3)
        self.robot_id = kwargs.get("robot_id", 4)

        self.id_dict = {
            self.top_left_id: "top_left",
            self.top_right_id: "top_right",
            self.bottom_left_id: "bottom_left",
            self.bottom_right_id: "bottom_right",
            self.robot_id: "robot"
        }

        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)

        self.transformed_aruco_state = None
        self.perspective_transform = None

        self._init_perspective_correction(initial_image)


    def get_corrected_image(self, image, hide_aruco=False):
        """
        Get the cropped image after the perspective correction.
        """
        assert self.perspective_transform is not None, "Perspective transform is not initialized. Call correct_perspective first."
        
        corrected_image = cv2.warpPerspective(image, self.perspective_transform, (self.map_size[0], self.map_size[1]), borderValue=255)

        if hide_aruco:
            # set the corners to white
            for marker in self.transformed_aruco_state.values():
                cv2.fillPoly(corrected_image, [marker["corners"].astype(np.int32)], (255, 255, 255))

        return corrected_image


    def get_robot_pose(self, image: np.ndarray, is_corrected: bool=False):
        """
        Get the robot pose in the image.
        """
        assert self.perspective_transform is not None, "Perspective transform is not initialized. Call correct_perspective first."
        aruco_state = self._detect_aruco(image)

        robot_corners = aruco_state["robot"]["corners"].astype(np.float32)
        corner_0 = robot_corners[0]
        corner_3 = robot_corners[3]
        if not is_corrected:
            robot_corners = np.hstack([aruco_state["robot"]["corners"], np.ones((4, 1), dtype=np.float32)]).astype(np.float32)
            robot_corners = robot_corners@self.perspective_transform.T
            robot_corners = robot_corners[:, :2] / robot_corners[:, 2].reshape(-1, 1)

            corner_0 = self.perspective_transform@np.append(aruco_state['robot']['corners'][0], 1)
            corner_0 = corner_0[:2] / corner_0[2]
            corner_3 = self.perspective_transform@np.append(aruco_state['robot']['corners'][3], 1)
            corner_3 = corner_3[:2] / corner_3[2]
        
        robot_center = robot_corners.mean(0)
        robot_orientation = (corner_0 - corner_3) / np.linalg.norm(corner_0 - corner_3)

        return robot_center, robot_orientation, robot_corners
    

    def find_polygons(self, image: np.ndarray, border_size=1, min_area=200, thresh_1 = 127, thresh_2 = 255):
        # code from https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/

        # converting image into grayscale image 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        # setting threshold of gray image 
        _, threshold = cv2.threshold(gray, thresh_1, thresh_2, cv2.THRESH_BINARY) 

        # using a findContours() function 
        contours, _ = cv2.findContours( 
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        i = 0
        polygon_list = []

        # list for storing names of shapes 
        for contour in contours:
            # Removing all polygons that intersect with the aruco markers
            valid_polygon = True
            for aruco_corners in self.transformed_aruco_state.values():
                corner_points = np.int32(aruco_corners["corners"])
                for point in contour.squeeze(1):
                    if cv2.pointPolygonTest(aruco_corners["corners"].astype(np.int32), (int(point[0]), int(point[1])), False) > 0:
                        valid_polygon = False
                        break
            
            # removing all polygons that are too close to the border
            for point in contour.squeeze(1):
                if point[0] < border_size or point[0] > image.shape[1] - border_size or point[1] < border_size or point[1] > image.shape[0] - border_size:
                    valid_polygon = False
                    break

            if not valid_polygon:
                continue

            # here we are ignoring first counter because 
            # findcontour function detects whole image as shape 
            if i == 0: 
                i = 1
                continue

            # cv2.approxPloyDP() function to approximate the shape 
            approx = cv2.approxPolyDP( 
                contour, 0.01 * cv2.arcLength(contour, True), True)
            
            # remove small area polygons
            if cv2.contourArea(approx) < min_area:
                continue
            
            # only keep points in the convex hull
            polygon = cv2.convexHull(approx)

            # if the polygon has more than 10 points, it is a circle we don't want to keep it
            if len(polygon) > 7:
                continue
            polygon_list.append(polygon)
        
        return polygon_list

    def find_circles(self, image: np.ndarray, min_radius=1, max_radius=40, thresh1=300, thresh2=30):
        return find_circles(image, min_radius, max_radius, thresh1, thresh2)
    

    def find_shapes(self, image, border_size=1):
        polygons = self.find_polygons(image, border_size)
        circles = self.find_circles(image, border_size)
        return polygons, circles

    
    def _detect_aruco(self, image: np.ndarray):
        """
        Detect ArUco markers in the given perspective corrected image.
        Returns the corners, centers and orientations of each detected marker. In x,y format.
        For openCV, the origin is at the top left corner of the image.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = self.detector.detectMarkers(gray)
        corners = [corner.squeeze() for corner in corners]
        centers = [corner.mean(0) for corner in corners]
        orientations = [(corner[0] - corner[3])/np.linalg.norm(corner[0] - corner[3]) for corner in corners]

        ids = ids.squeeze().reshape(-1, 1)

        result = {
            self.id_dict[int(id)]: {
                'corners': corner,
                'center': center,
                'orientation': orientation
            } for id, corner, center, orientation in zip(ids, corners, centers, orientations)
        }

        return result
    

    def _init_perspective_correction(self, image: np.ndarray):
        """
        Correct the perspective of the image using the given corners.
        img_size has to be in the format (width, height).
        """

        img_size = self.map_size
        aruco_state = self._detect_aruco(image)

        top_left_corners = aruco_state['top_left']['corners']
        top_right_corners = aruco_state['top_right']['corners']
        bottom_right_corners = aruco_state['bottom_right']['corners']
        bottom_left_corners = aruco_state['bottom_left']['corners']

        # extract the left upper corner of the marker
        top_left = top_left_corners[0]
        top_right = top_right_corners[1]
        bottom_right = bottom_right_corners[2]
        bottom_left = bottom_left_corners[3]

        perspective_points = np.float32([top_left, top_right, bottom_left, bottom_right])
        image_points = np.float32([
            [0, 0],
            [img_size[0] - 1, 0],
            [0, img_size[1] - 1],
            [img_size[0] - 1, img_size[1] - 1]
        ])

        self.perspective_transform = cv2.getPerspectiveTransform(perspective_points, image_points)
        self._correct_aruco_state_perspective(aruco_state)

        return
    
    
    def _correct_aruco_state_perspective(self, aruco_state: dict):
        """
        Update the state of the ArUco markers after the perspective correction.
        """
        assert self.perspective_transform is not None, "Perspective transform is not initialized. Call correct_perspective first."
        self.transformed_aruco_state = {}
        for marker, dico in aruco_state.items():
            corners = np.hstack([dico["corners"], np.ones((4, 1))]).astype(np.float32)@self.perspective_transform.T
            corners = corners[:, :2] / corners[:, 2].reshape(-1, 1)
            
            center = self.perspective_transform@np.append(dico["center"], 1)
            center = center[:2] / center[2]

            orientation = (corners[0] - corners[3]) / np.linalg.norm(corners[0] - corners[3])

            self.transformed_aruco_state[marker] = {
                'corners': corners,
                'center': center,
                'orientation': orientation
            }



def find_circles(image, min_radius=1, max_radius=40, thresh1=300, thresh2=30):
    # https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, 
                                        param1 = thresh1, 
                                        param2=thresh2,
                                        minRadius=min_radius,
                                        maxRadius=max_radius) 

    return detected_circles

def find_polygons(image: np.ndarray):
    # code from https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/

    # converting image into grayscale image 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # setting threshold of gray image 
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 

    # using a findContours() function 
    # this works by using the beforedefined threshhold find the places where there is a transition from black to white 
    #(threshold to not threshhold) and then defining those as contours
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    i = 0
    polygon_list = []

    # list for storing names of shapes 
    for contour in contours: 

        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape 
        if i == 0: 
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape 
        #Douglas-Peucker algorithm is used to reduce the number of points in contours by defining the shortest distance between the points and the contour vector
        approx = cv2.approxPolyDP( 
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        # only keep points in the convex hull
        polygon = cv2.convexHull(approx)
        polygon_list.append(polygon)
    
    return polygon_list

def extend_polygons(polygons: List[np.ndarray], extension_size: int = 10) -> List[np.ndarray]:
    # draw contours on the image
    extended_polygons = []
    for polygon in polygons:
        tmp_polygon = polygon.squeeze(1) #make sure the polygon is in the right format (two dimensions)
        # use the previous and next points to calculate the extension direction
        previous_points = np.roll(tmp_polygon, 1, axis=0)
        next_points = np.roll(tmp_polygon, -1, axis=0)
        # normalize the vectors
        vector1 = (previous_points - tmp_polygon) / np.linalg.norm(previous_points - tmp_polygon, axis=1, keepdims=True)
        vector2 = (next_points - tmp_polygon) / np.linalg.norm(next_points - tmp_polygon, axis=1, keepdims=True)
        # calculate the mean vector
        mean_vector = (vector1 + vector2) / 2
        mean_vector = mean_vector / np.linalg.norm(mean_vector, axis=1, keepdims=True)
        # rotate the mean vector by 180 degrees
        angle = np.pi
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        extension_direction = mean_vector @ rotation_matrix
        extension_direction = extension_direction
        # extend the polygon by the extension size
        extended_polygon = (polygon + extension_size * extension_direction[:, None, :]).astype(np.int32)

        extended_polygons.append(extended_polygon)
    return extended_polygons
