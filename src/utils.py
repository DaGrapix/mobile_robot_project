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
    cv2.circle(image, tuple(route[0]), 5, (0, 0, 255), -1)
    for i in range(len(route) - 1):
        cv2.line(image, tuple(route[i]), tuple(route[i+1]), (0, 255, 0), 2)
        cv2.circle(image, tuple(route[i+1]), 5, (0, 0, 255), -1)
    return image
    
def add_line(image,vec,point):
    
    pt1 = tuple(point.astype(int))
    pt2 = tuple((point + vec).astype(int))

    cv2.line(image, pt1, pt2, (0, 255, 255), 2)
    return image
    
def add_point(image, point):
    new_image = image.copy()
    cv2.circle(new_image, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    return new_image

    
def plot_orientation(angle_estime, variance_angle):
    import matplotlib.pyplot as plt

    sigma = np.sqrt(variance_angle)  # Écart-type
    angle = np.linspace(-np.pi, np.pi, 500)  # Intervalle [-pi, pi]
    gaussienne = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((angle - angle_estime) / sigma) ** 2)

    # Tracé
    plt.figure(figsize=(8, 4))
    plt.plot(angle, gaussienne, label="Gaussienne de l'angle")
    plt.axvline(x=angle_estime, color='red', linestyle='--', label="Angle estimé")
    plt.title("Estimation de l'orientation")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Densité de probabilité")
    plt.legend()
    plt.show()

from scipy.stats import multivariate_normal
def gauss2d(robot_pos_kalman, covariance_kalman, map_size):
    # https://gist.github.com/danoneata/75c5bbe8d651d4ec0e804995010a850d
    
    img = np.zeros(map_size)
    X = np.arange(0, map_size[0], 1)
    Y = np.arange(0, map_size[1], 1)
    X, Y = np.meshgrid(X, Y)

    XY = np.vstack((X.flatten(), Y.flatten())).T

    mu = robot_pos_kalman
    sigma = covariance_kalman
    normal_rv = multivariate_normal(mu, sigma)
    
    pdf = normal_rv.pdf(XY)
    img = pdf.reshape(*map_size, order="F")

    return np.uint8(255*img / img.max())


def draw_kalman_pos(robot_pos_kalman, covariance_kalman, map_size):
    gaussian = gauss2d(robot_pos_kalman, covariance_kalman, map_size)
    # img = np.asarray(gaussian.astype(np.uint8))
    # cv2.circle(img, tuple(np.uint8(robot_pos_kalman)), 255, -1)
    
    return gaussian


import matplotlib.pyplot as plt
def draw_kalman_angle(angle_kalman, variance_angle):
    # https://stackoverflow.com/questions/77714621/how-to-convert-a-matplotlib-figure-to-a-cv2-image
    sigma = np.sqrt(variance_angle)  # Écart-type
    angle = np.linspace(-np.pi, np.pi, 500)  # Intervalle [-pi, pi]
    gaussienne = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((angle - angle_kalman) / sigma) ** 2)
    
    fig, ax = plt.subplots()
    ax.plot(angle, gaussienne, label="Gaussienne de l'angle")
    ax.axvline(x=angle_kalman, color='red', linestyle='--', label="Angle estimé")

    plt.title("Estimation de l'orientation")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Densité de probabilité")
    plt.legend()

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    return img
