import numpy as np

#convert into matematical or cv2 coordinate
def convert_vec(vec):
    if len(vec.shape) == 1:
        return np.array([vec[0],-vec[1]])
    converted_vec = np.stack((vec[:, 0], -vec[:, 1]), axis=-1)
    return converted_vec

def convert_points(points, y_max):
    if len(points.shape) == 1:
        return np.array([points[0], y_max - points[1]])
    converted_points = np.stack((points[:, 0], y_max - points[:, 1]), axis=-1)
    return converted_points

#define a line with two points (direction vector and second point)
def define_line_p(point1,point2):
    
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    droite = {
        "v_dir" : (point2 - point1),
        "point" : point1
    }
    return droite

#Take a point and calculate the distance and the direction to the line
def dir_to_line(point, droite):
    #from point, generate the projection on the line using parametric equation of the line
    v_point_droite = point - droite["point"]
    produit_scalaire = np.dot(v_point_droite, droite["v_dir"])
    t = produit_scalaire/(np.dot(droite["v_dir"],droite["v_dir"]))
    projection = droite["point"] + t*droite["v_dir"]
    # generate the vector frome the point and the projection
    direction = projection - point

    return direction

def get_angle(v1,v2):
    angle = np.dot(v1,v2)
    denom = (np.linalg.norm(v1)*np.linalg.norm(v2))

    #avoid division by 0
    if(denom != 0):    
        angle = angle/denom
        angle = np.sign(np.linalg.det(np.array([v1, v2])))*np.arccos(angle)
        return angle
    else:
        return 0

def convert_angle_to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])