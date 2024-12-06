import numpy as np
import math
from src.geometry import get_angle, define_line_p, dir_to_line, convert_points
from src.path_planning import path_planning

GLOBAL_GAIN_ANGLE = 1.7
CONTROL_GAIN_PATH = 0.7
OBSTACLE_GAIN = 1.5
OBSTACLE_SPEED = 90

class FSMController():
    def __init__(self,
                 robot_pos_m,
                 all_target_m,
                 objectives_m,
                 extended_polygons,
                 img_shape,
                 first_prox = [0, 25, 0],
                 prox_obstacle_threshold = 30,
                 intensity = 10,
                 ):
        
        self.previous_target = robot_pos_m
        self.all_target_m = all_target_m
        self.objectives_m = objectives_m
        self.current_target = all_target_m[0, :]
        self.path_m = define_line_p(robot_pos_m, self.current_target)

        self.first_prox = first_prox
        self.state = 0
        self.intensity = intensity
        self.prox_obstacle_threshold = prox_obstacle_threshold


        self.img_shape = img_shape
        self.extended_polygons = extended_polygons
       
    @property
    def current_path(self):
        return np.vstack((self.previous_target.reshape(-1, 2), self.all_target_m))
        
    #cotroller using the vector director of line paths
    def global_control_angle(self, robot_orientation):

        v_dir = self.path_m["v_dir"]

        #calculate the angle between the robot and the direction of the path
        robot_to_vdir_angle = get_angle(robot_orientation, v_dir)
        
        #define speed
        speed_regulator = GLOBAL_GAIN_ANGLE * np.degrees(robot_to_vdir_angle)
        
        return speed_regulator

    #controller using distance from droite and the direction with the angle
    def global_control_path(self, vec_robot_to_path, robot_orientation):
        
        #calculation angle from the robot orientation to the projection on the path
        robot_to_path_angle = get_angle(robot_orientation, vec_robot_to_path)  
        
        controller = CONTROL_GAIN_PATH * np.sign(robot_to_path_angle) * np.linalg.norm(vec_robot_to_path)
        
        return controller

    #controller for global navigation
    def global_control(self, vec_robot_to_path, robot_orientation):
        
        speed = 150
        #mix the controller with angle of the path and distance from path
        controller = self.global_control_angle(robot_orientation)
        controller += self.global_control_path(vec_robot_to_path, robot_orientation)

        #avoid variable conversion problem with float and int
        if math.isnan(controller):
            speed_left = 0
            speed_right = 0
        else:
            speed_left = speed - int(controller)
            speed_right = speed + int(controller)
        
        return speed_left, speed_right
    
    #function that set the controller using the front prox sensors values
    def local_control(self, prox, vec_robot_to_path):
        """
            the obstacle avoidance is based on the prox sensor, so if a sensor from left is activated, 
            the robot will turn right. If the front sensor is activated the robot need to choose a direction.
            If the robot is on the left of the path, it will avoid the obstacle turning left.
            
        """
        
        local_turning_direction = np.sign(get_angle(vec_robot_to_path, self.path_m["v_dir"])) 
        
        #set weight for obstacle avoidance from prox sensors
        initial_weight = np.array([-2.5, -2, 3.0, 2, 2.5, 0, 0])
        W = initial_weight

        #change the wheight when the obstacle is first seen by the middle prox sensor
        if self.first_prox == 2:
            W[2] = OBSTACLE_GAIN*initial_weight[2] * local_turning_direction
        else:
            W = OBSTACLE_GAIN*initial_weight
        
        control = np.array(prox).T @ W

        control = control  / 110

        speed_left = OBSTACLE_SPEED - int(control)
        speed_right = OBSTACLE_SPEED + int(control)

        return speed_left, speed_right

    #create a new path from the current robot location
    def recompute_global_path(self, robot_pos_m):
        objectives_m = self.objectives_m
        objectives_cv = convert_points(objectives_m, self.img_shape[0])
        robot_position_cv = convert_points(robot_pos_m, self.img_shape[0])
        optimal_route, optimal_route_positions, points, adjacency_matrix = path_planning(robot_position_cv, objectives_cv.reshape(-1, 2), self.extended_polygons, self.img_shape)
        all_target_cv = np.array(optimal_route_positions)[1:, :]
        all_target_m = convert_points(all_target_cv, self.img_shape[0])

        self.previous_target = robot_pos_m
        self.all_target_m = all_target_m
        self.objectives_m = objectives_m
        self.current_target = all_target_m[0, :]
        self.path_m = define_line_p(robot_pos_m, self.current_target)
        self.state = 0


    #choose the right state with fsm and change some wheigt for control if needed
    def update_state(self, dist_robot_line, robot_pos, prox):
        
        #go to state 1 (obstacle avoidance) if a prox sensor is activated
        if self.state==0 and (np.sum(prox) != 0):
            self.state=1
            #define which prox sensor is activated first
            self.first_prox = np.argmax(prox)

        #go to state 0 (follow line) if the distance to path is large enough and recompute the global route
        elif self.state==1 and (dist_robot_line > self.prox_obstacle_threshold):
            self.state = 0
            self.recompute_global_path(robot_pos)
            

    #Verifiy if target is reached and change target if needed, when last point is reached, end of program
    #return stop = True if all target are reached
    def change_target(self, robot_pos_m):

        stop = 0
        target_reach_threshold = 80
        
        #calculate distance to target
        distance_to_target = np.linalg.norm(robot_pos_m - self.current_target)

        #if closed enough to target
        if(distance_to_target < target_reach_threshold):

            #if the list of target is empty, all target are reached
            if len(self.all_target_m) > 1:

                #remove the target that has just been reached
                self.all_target_m = self.all_target_m[1:, :]
                if len(self.all_target_m) == 1:
                    # si il ne reste plus que le dernier point, on réduit le threshold pour s'arrêter dessus
                    target_reach_threshold = 25

                #initialize the new target
                self.previous_target = self.current_target.copy()
                self.current_target[:] = self.all_target_m[0, :]
                
                # modifiy circles (objectives) list if one reached
                id_list = np.where((self.objectives_m == self.previous_target).all(axis=1))[0]
                if len(id_list) == 1:
                    self.objectives_m = np.delete(self.objectives_m, id_list[0],0)

                
                #create path from the previous target to the new
                self.path_m["v_dir"] = np.array(self.current_target) - np.array(self.previous_target)
                self.path_m["point"] = self.previous_target
                
            else:
                print("END OF PROGRAM: the goals are reached")
                stop = 1 #===========================================================END OF PROGRAM
        
        return stop

    # choose the instructions from state of FSM and
    def get_command(self, prox, robot_pos, robot_orientation):

        vec_robot_to_path = dir_to_line(robot_pos, self.path_m)
        dist_robot_line = np.linalg.norm(vec_robot_to_path)
        
        speed_left, speed_right = 0, 0

        self.update_state(dist_robot_line, robot_pos, prox)

        if self.state==0:
            leds_top = [self.intensity, 0, self.intensity]
            #use global controller with the mix of vector of path and distance to path
            speed_left, speed_right = self.global_control(vec_robot_to_path, robot_orientation)
        if self.state==1:
            leds_top = [0, self.intensity, 0]
            #use local control using the prox values
            speed_left, speed_right = self.local_control(prox, vec_robot_to_path)

        #try to change target and verify if all the target are reached
        stop = self.change_target(robot_pos)

        return leds_top, speed_left, speed_right, stop
