# Mobile Robot Path Planning and Kalman Filter Implementation

This project involves implementing path planning and state estimation for a mobile robot using computer vision, path planning algorithms, and a Kalman Filter. The robot navigates on the map in order to collect all of the objective points while avoiding obstacles and estimates its position and orientation in real-time using odometry and camera information.

## Introduction

The Thymio robot has to get up every day to deliver his loaves of bread to his customers. One day, he asked Group 44 of the BOMR class for some help!

With the help of their instructor Prof. Mondada and the army of TAs, the group of students managed to quickly propose a solution to make the bread delivery as efficient as possible!

As part of the "Basics of Mobile Robotics" course project by Professor Mondada, this project aims at creating a complete mobile robotics pipeline on the Thymio robot. In this sense, the robot has to navigate a map to deliver his loaves to his customers, located on the red circles. During this mission, it must avoid obstacles and find the path that minimizes his delivery. To achieve this.

## Demo

![Robot Navigation Demo](assets/videos/video.gif)

## Project Overview

The project is divided into several key components:

1. **Computer Vision and Localization**: Using OpenCV and ArUco markers to detect the robot's position and orientation within the environment, detecting global obstacles in the form of black polygons as well as objective red discs.

2. **Path Planning**: Implementing path planning algorithms to compute the most efficient route to reach the target points while avoiding obstacles.

3. **Kalman Filter**: Applying a Kalman Filter for state estimation to fuse sensor data and improve the accuracy of the robot's perceived position and orientation.

4. **Control and Navigation**: Developing control strategies for the robot to follow the planned path and adjust its movements based on real-time feedback.

## Main Components and Ideas

### 1. Computer Vision and Localization

- **Camera Calibration**: Calibrating the camera to correct distortions and improve the accuracy of the position and orientation measurements.

- **ArUco Marker Detection**: Utilizing ArUco markers placed on the robot and within the environment to determine the robot's pose.

- **Coordinate Transformation**: Converting detected marker positions from image coordinates to real-world coordinates for accurate localization.

### 2. Path Planning

- **Map Representation**: Defining the environment's map, including obstacles and target points, to be used for path planning.

- **Algorithm Implementation**: Implementing path planning algorithms such as the Traveling Salesman Problem (TSP) solver to determine the optimal path visiting all target points.

- **Obstacle Avoidance**: Incorporating obstacle detection and avoidance mechanisms within the path planning to ensure safe navigation.

### 3. Kalman Filter

- **State Estimation**: Using the KF to estimate the robot's state (position and orientation) by combining the predictive model and the measurements from the sensors.

- **Predictive Model**: Defining the robot's motion model to predict its next state based on control inputs.

- **Measurement Update**: Updating the state estimate using measurements from the computer vision system.

### 4. Control and Navigation

- **Motor Control**: Sending appropriate speed commands to the robot's motors to follow the planned path.

- **Feedback Loop**: Continuously updating the robot's state estimate and adjusting control commands based on real-time feedback.

- **Error Handling**: Implementing mechanisms to handle discrepancies between the expected and observed states, including recovery strategies.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```