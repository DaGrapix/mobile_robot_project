# Mobile Robot Path Planning and Kalman Filter Implementation

This project involves implementing a completely autonomous mobile robot using computer vision, path planning algorithms, controllers and a Kalman Filter. The robot navigates the map in order to collect all of the objective points while avoiding both global obstacles (seen by the camera) and local ones (only seen by proximity sensors when close enough), and by estimating its position and orientation in real-time using odometry and camera information.

## Demo

![Robot Navigation Demo](assets/videos/video.gif)

## Introduction

The Thymio robot has to get up every day to deliver its loaves of bread to its customers. One day, he asked Group 44 of the BOMR class for some help!

With the help of their instructor Prof. Mondada and the army of TAs, the group managed to quickly propose a solution (in 4 weeks!) to make the bread delivery as efficient as possible!


## Project Overview

The project is divided into several key components:

1. **Computer Vision and Localization**: 
ArUco markers are used to detect the environment map as well as the pose of the robot.
    Black polygons, which are global obstacles, are detected with an edge detection algorithm and red circles with the Hough circle transform.

2. **Path Planning**: A path planning algorithm retrieves the shortest path for the robot to collect all red circles. Taking into account the objective points and environment keypoints (obtained by expanding the global obstacles), the algorithm runs a graph search in the graph comprising these two sources of points with edges between two points if they are mutually visible.
    
    The search is done in a dynamic-programming sense where first the shortest path between any pair of objective points is computed, after which the shortest global path is computed by stitching together, for every possible permutation of objective points, the paths obtained by considering each successive pair of objective points.

3. **Control and Navigation**: The navigation is split into two modes: global and local navigation. While the goal of the global navigation is to reach all of the objectives, the local navigation ensures that the robot doesn't make an accident with an obstacle that is not seen by the camera.
The robot switches from the first to the second when an obstacle is detected by its proximity sensors.

4. **Pose estimation**: A Kalman filter is used to merge the information obtained by the camera and the odometry. This also allows the robot to move independtly of the camera (when it is not available) by only relying on its odometry information.

Further explanations are available in the `report.ipynb` notebook. 