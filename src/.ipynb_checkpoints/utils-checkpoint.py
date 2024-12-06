import cv2
import numpy as np
from tdmclient import ClientAsync, aw

def add_robot(image, robot_pos, robot_orientation, robot_corners):
    new_image = image.copy()
    cv2.circle(new_image, tuple(robot_pos.astype(np.int32)), 5, (0, 0, 255), 5)
    # draw arrow
    arrow_length = 50
    arrow_end = robot_pos + arrow_length * robot_orientation
    cv2.line(new_image, tuple(robot_pos.astype(np.int32)), tuple(arrow_end.astype(np.int32)), (0, 0, 255), 5)
    for corner in robot_corners:
        cv2.circle(new_image, tuple(corner.astype(np.int32)), 5, (0, 0, 255), 5)
    return new_image


def add_polygons(image, polygons):
    new_image = image.copy()
    for polygon in polygons:
        for point in polygon:
            cv2.circle(new_image, tuple(point[0]), 5, (0, 0, 255), -1)
    return new_image

def add_circles(image, circles):
    new_image = image.copy()
    if circles is None:
        cv2.imshow("Detected Circle", new_image) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        return
    for circle in circles[0]:
        a, b, r = tuple(circle.astype(int))

        # Draw the circumference of the circle. 
        cv2.circle(new_image, (a, b), r, (0, 255, 0), 2) 
  
        # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(new_image, (a, b), 1, (0, 0, 255), 3) 
    return new_image

def add_graph(image, points, adjacency_matrix, color=(0, 255, 0)):
    new_image = image.copy()
    for i, point in enumerate(points):
        cv2.circle(new_image, tuple(point), 5, (0, 0, 255), -1)
        for j, edge in enumerate(adjacency_matrix[i]):
            if edge != 0:
                cv2.line(new_image, tuple(points[i]), tuple(points[j]), color, 2)
                # write the edge value
                cv2.putText(new_image, str(int(edge)), tuple((points[i] + points[j]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return new_image

def add_route(image, route):
    for i in range(len(route) - 1):
        cv2.line(image, tuple(route[i]), tuple(route[i+1]), (0, 255, 0), 2)
    return image
    
def add_line(image,vec,point):
    
    pt1 = tuple(point.astype(int))
    pt2 = tuple((point + vec).astype(int))

    cv2.line(image, pt1, pt2, (0, 255, 255), 2)
    return image
    
def add_point(image, point):
    new_image = image.copy()
    cv2.circle(new_image, tuple(point), 5, (0, 0, 255), -1)
    return new_image


    