# Code inspired from    https://www.geeksforgeeks.org/detecting-aruco-markers-with-opencv-and-python-1/ for the aruco patch generation.


import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from typing import Optional
import argparse
import os
import json


def generate_aruco(marker_id: int, marker_size: int=200, aruco_dict: Optional[aruco.Dictionary] = None):
    """
    Generate an ArUco marker with the given ID, dictionary and size.
    """
    if (aruco_dict is None):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    assert marker_id >= 0 and marker_id <= aruco_dict.bytesList.shape[0], "Marker ID is out of range"
    # Create an ArUco marker
    aruco_marker = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    return aruco_marker

# to run this script, use the following command:
# python utils/aruco.py --save_path=assets/aruco
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_path', type=str, required=False, help='Path to save the output file')

    args = parser.parse_args()
    save_path: Optional[str] = args.save_path

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_size  = 200

    top_left_id     = 0
    top_right_id    = 1
    bottom_left_id  = 2
    bottom_right_id = 3
    robot_id        = 4

    top_left_marker     = generate_aruco(top_left_id, marker_size, aruco_dict)
    top_right_marker    = generate_aruco(top_right_id, marker_size, aruco_dict)
    bottom_right_marker = generate_aruco(bottom_right_id, marker_size, aruco_dict)
    bottom_left_marker  = generate_aruco(bottom_left_id, marker_size, aruco_dict)
    robot_marker        = generate_aruco(robot_id, marker_size, aruco_dict)

    aruco_info = {
        'aruco_dict': "DICT_6X6_250",
        'marker_size': marker_size,
        'top_left_id': top_left_id,
        'top_right_id': top_right_id,
        'bottom_left_id': bottom_left_id,
        'bottom_right_id': bottom_right_id,
        'robot_id': robot_id
    }

    # save the marker images and state
    if save_path is not None:
        cv2.imwrite(os.path.join(save_path, 'top_left_marker.png'), top_left_marker)
        cv2.imwrite(os.path.join(save_path, 'top_right_marker.png'), top_right_marker)
        cv2.imwrite(os.path.join(save_path, 'bottom_right_marker.png'), bottom_right_marker)
        cv2.imwrite(os.path.join(save_path, 'bottom_left_marker.png'), bottom_left_marker)
        cv2.imwrite(os.path.join(save_path, 'robot_marker.png'), robot_marker)

        with open(os.path.join(save_path, 'state.json'), 'w') as f:
            json.dump(aruco_info, f)

    # create an image viewer where i can scroll through the markers
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    axes[0, 0].imshow(top_left_marker, cmap='gray')
    axes[0, 0].set_title('Top Left Marker')
    axes[0, 1].imshow(top_right_marker, cmap='gray')
    axes[0, 1].set_title('Top Right Marker')
    axes[1, 0].imshow(bottom_left_marker, cmap='gray')
    axes[1, 0].set_title('Bottom Left Marker')
    axes[1, 1].imshow(bottom_right_marker, cmap='gray')
    axes[1, 1].set_title('Bottom Right Marker')
    axes[0, 2].imshow(robot_marker, cmap='gray')
    axes[0, 2].set_title('Robot Marker')
    axes[1, 2].axis('off')
    
    plt.show()
