import numpy as np
import time


##############################function to make geometrical operations##############################
#convertit dans les axes de coordonnée mathématique
def convert_point(point, y_max):
    return np.array([point[0], y_max - point[1]])

def convert_vec(vec):
    return np.array([vec[0],-vec[1]])


#define a line with two points (direction vector and second point)
def define_line_p(point1,point2):
    droite = {
        "v_dir" : (point2 - point1),
        "point" : point1
    }
    return droite


#define a line with 1 point and one vector (direction vector and second point)
def define_line_v(point,vector):
    droite = {
        "v_dir" : vector,
        "point" : point
    }
    return droite


#take a line and a point, if the point belong to the line: return true
def point_in_line(droite,point):
    result = droite["v_dir"][0]*point[1] - droite["v_dir"][1]*point[0]
    result += droite["v_dir"][1]*droite["point"][0] - droite["v_dir"][0]*droite["point"][1]
    if result == 0:
        return 1
    else:
        return 0


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
        angle = np.sign(np.linalg.det(np.array([v1, v2])))*np.acos(angle)
        return angle
    else:
        return 0
