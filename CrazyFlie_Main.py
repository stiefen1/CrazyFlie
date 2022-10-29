import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
import numpy as np

logging.basicConfig(level=logging.ERROR)

###################################### CONTROLLER ########################################

class P_controller:
    def __init__(self, kp=1, MAX_SPEED=0.1):
        self.kp = kp
        self.MAX_SPEED = MAX_SPEED
        self.u = np.array([0, 0, 0])  # command

    def get_u(self, pt2go, actual_pos):
        # Compute the error
        error = pt2go - actual_pos

        # Compute the Proportionnal command
        self.u = self.kp * error

        # Saturate the command
        for i in range(3):
            if self.u[i] > self.MAX_SPEED:
                self.u[i] = self.MAX_SPEED

            elif self.u[i] < -self.MAX_SPEED:
                self.u[i] = -self.MAX_SPEED

        return self.u


###################################### PLAYGROUND ########################################

class playground:
    def __init__(self):
        """
                            W
        ##########################################
        #                   L               h    #
        # l   ------------------------------   l #
        #                                   |    #
        #                                   | H  #
        #                                   |    #
        #     ------------------------------     # H3
        #    |                                   #
        #    | H                                 #
        #    |            L                      #
        #  l  ------------------------------   l #
        #                                  h     #
        ##########################################
        #                                        #
        #                          xxxx          #
        #                          xxxx          #
        #                          xxxx          #
        #        xxxx                            # H2
        #        xxxx                            #
        #        xxxx                            #
        #                                        #
        ##########################################
        #                                        #
        #                                        #
        #                                        #
        #             ooooo                      #
   x0 > #             ooooo                      # H1
        #             ooooo                      #
        ^                                        #
        |                                        #
        O-->######################################
                        ^
                        y0
        """
        self.W = 3  # m
        self.H1 = 1.5  # m
        self.H2 = 2.5  # m
        self.H3 = 1.5  # m

        self.padHeight = 0.05
        self.padMargin = 0.01
        self.padSize_x = 0.25
        self.padSize_y = 0.2
        self.padEdge = np.zeros((4, 2))
        self.padCenter = np.array([0., 0.])

        # Inital position of the platform
        self.xyz0 = np.array([1, 1, 0.1])
        # x = 1, y = 2



###################################### CHARLES AIRLINES ########################################

class Charles:
    def __init__(self):

        print("Bienvenue sur Charles Airline")
        self.uri = "radio://0/80/2M/E7E7E7E701"
        self.default_height = 0.3

        self.playground = playground()

        # Attributes for moving to landing zone
        self.border = False

        # State machine obstacle avoidance while searching
        self.move = 0  # Type of move when searching (left, right or forward)
        self.avoiding = False  # Drone is avoiding an obstacle
        self.obs_y = 0.0  # Record position of obstacle to go a bit straight to be able to detect it on the side

        # Initial position in the global frame
        self.xyz0 = self.playground.xyz0

        # Position in the "take off platform" frame
        self.xyz = np.array([0., 0., 0.])
        self.rpy = np.array([0., 0., 0.])

        self.speed_controller = P_controller()

        # Position in the global frame
        self.xyz_global = self.xyz0

        # self.range = [front, back, up, left, right, zrange]
        self.range = np.array([0, 0, 0, 0, 0, 0])
        self.xyz_rate_cmd = np.array([0, 0, 0])
        self.rpy_rate_cmd = np.array([0, 0, 0])

        self.state = 0

        # variable related to landing pad
        self.stateCentering = 0
        self.edgeDetected = False
        self.edgeFound = 0  # 0:not found, 1:rising edge, 2:falling edge
        self.edgeThresholdUp = 0.015
        self.edgeThresholdDown = 0.01
        self.edgeTimeDelay = 0.5
        self.centerReached = False
        self.queueZ = 50 * [0.]
        self.diffZ = 0.
        self.meanZ = 0.

        # Sensors list for callbacks
        self.pos_var_list = ['stateEstimate.x',
                             'stateEstimate.y',
                             'stateEstimate.z']

        self.multi_var_list = ['range.front',
                               'range.back',
                               'range.up',
                               'range.left',
                               'range.right',
                               'range.zrange']

        # Searching path variables
        self.waypoints = None
        self.keep_searching = False

        # Constants :
        self.min_dist = 300  # Distance to stop flying

        self.l = 0.1  # y margin
        self.L = self.playground.W - 2 * self.l  # y width
        self.h = 0.1  # x margin
        self.N = 6  # Number of way
        self.H = (self.playground.H3 - 2 * self.h) / (self.N - 1)  # Distance between two ways

        self.N_spiral = 5  # Number of spirals to plan
        self.H_spiral = 0.3  # Distance between each spiral

        self.Te_loop = 0.01  # frequency of main loop in !!! SECONDS !!!
        self.Te_log = 10  # frequency of data transmission in !!! MILLISECONDS !!!

        print("Driver initialisation ..")
        cflib.crtp.init_drivers()

        print("Log Configuration ..")
        self.setLog()

    # ----------------------------------------------------------------------------------------#

    def is_not_close(self):
        # False if an object is too close to the drone (up)
        return (self.range[2] > self.min_dist)

    # ----------------------------------------------------------------------------------------#

    def is_close_obs(self, range):  # if use of self.range, scale in mm
        MIN_DISTANCE = 300  # mm

        if range is None:
            return False
        else:
            return range < MIN_DISTANCE

    # ----------------------------------------------------------------------------------------#

    def setLog(self): 
        # Set the log configuration
        self.log_position = LogConfig(name='Position', period_in_ms=self.Te_log)
        self.log_multiranger = LogConfig(name='Multiranger', period_in_ms=self.Te_log)

        for var in self.pos_var_list:
            self.log_position.add_variable(var, 'float')

        for var in self.multi_var_list:
            self.log_multiranger.add_variable(var, 'float')

    # ----------------------------------------------------------------------------------------#

    def log_pos_callback(self, timestamp, data, logconf):
        # Get x,y,z and roll, pitch, yaw values and save it into self variables
        self.xyz = np.array([data[self.pos_var_list[0]], -data[self.pos_var_list[1]], data[self.pos_var_list[2]]])
        self.xyz_global = self.xyz + self.xyz0  # Position in the global frame
        self.queueZ.pop(0)
        self.queueZ.append(self.xyz[2] ** 3)
        self.diffZ = max(self.queueZ) - min(self.queueZ)
        self.meanZ = sum(self.queueZ) / len(self.queueZ)
        
        # When the drones take off again, its referential changes
        if self.state >= 4:
            self.xyz[0] += self.playground.padCenter[0]
            self.xyz[1] += self.playground.padCenter[1]

    # ----------------------------------------------------------------------------------------#

    def log_multi_callback(self, timestamp, data, logconf):
        # Get multiranger values and save it into self variables
        self.range = [data[self.multi_var_list[0]],
                      data[self.multi_var_list[1]],
                      data[self.multi_var_list[2]],
                      data[self.multi_var_list[3]],
                      data[self.multi_var_list[4]],
                      data[self.multi_var_list[5]]]

    # ----------------------------------------------------------------------------------------#
    # Function making the drone going from the start zone to the landing zone while avoiding obstacle
    # To tune :
    # MAX_DISTANCE : x-distance to move before being in the landing zone
    # VELOCITY_X : velocity in x-direction, increase if want to go faster to the zone, but can break the avoidance
    # VELOCITY_Y : velocity in y-direction (to avoid obstacle), increase if does not avoid obstacle well
    # MIN_Y : to avoid obstacle by the right instead of the left if risk of going out of the flying zone

    def move_to_landing_zone(self):
        self.keep_flying = True

        # Some constants to modify
        VELOCITY_X = 0.4
        VELOCITY_Y = 0.8
        MIN_Y = 0.5
        MAX_DISTANCE = self.playground.H1 + self.playground.H2 - self.xyz0[0]

        # Commands
        velocity_x = 0.0
        velocity_y = 0.0

        # There is an obstacle in front
        if self.is_close_obs(self.range[0]):

            # If near the border, avoid by the right
            if ((self.xyz[1] + self.xyz0[1]) < MIN_Y) or (self.border):
                self.border = True
                velocity_x = 0.0
                velocity_y = VELOCITY_Y

            # If not near the border, avoid by the left
            else:
                velocity_x = 0.0
                velocity_y = -VELOCITY_Y

        # If no obstacle, go forward
        else:
            velocity_x = VELOCITY_X
            velocity_y = 0.0

        # Arrived in searching zone
        if (self.xyz[0] > MAX_DISTANCE):
            self.keep_flying = False
            velocity_x = 0.0
            velocity_y = 0.0

        # Send command
        self.xyz_rate_cmd = [velocity_x, velocity_y, 0]

        # Return flase if in searching zone, true otherwise
        return self.keep_flying

    # ----------------------------------------------------------------------------------------#
    def set_spiral_waypoints(self):
        """ Creates waypoints in a spiral shape to search the landing pad in the starting zone"""
        self.waypoints = np.array([])
        self.waypoints = np.append(self.waypoints, [self.xyz[0], self.xyz[1], self.xyz[2]])

        for i in range(2 * self.N_spiral):
            self.waypoints = np.append(self.waypoints, self.waypoints[6 * i:6 * i + 3] + np.array(
                [0, ((-1) ** i) * (i + 1) * self.H_spiral, 0]))
            self.waypoints = np.append(self.waypoints, self.waypoints[6 * i + 3:6 * i + 6] + np.array(
                [((-1) ** i) * (i + 1) * self.H_spiral, 0, 0]))
        
        self.waypoints = self.waypoints[3::]

    def set_waypoints(self):
        """
        Create a list of waypoints in the GLOBAL FRAME to search the platform
                            W
        ##########################################
        #                   L               h    #
        # l   ------------------------------   l #
        #                                   |    #
        #                                   | H  #
        #                                   |    #
        #     ------------------------------     # H3
        #    |                                   #
        #    | H                                 #
        #    |            L                      #
        #  l  ------pi-----------------------  l #
        #                                  h     #
        ##########################################
        """
        # When we enter this function, drone is at position pi
        
        self.waypoints = np.array([])
        self.waypoints = np.append(self.waypoints, [self.xyz_global[0], self.l, self.default_height])

        # Direction to start obstacle avoidance
        self.move = 0

        for i in range(self.N - 1):
            self.waypoints = np.append(self.waypoints, self.waypoints[6 * i:6 * i + 3] + np.array([self.H, 0, 0]))
            self.waypoints = np.append(self.waypoints,
                                       self.waypoints[6 * i + 3:6 * i + 6] + np.array([0, self.L * (-1) ** i, 0]))

        # Correct starting direction
        if self.xyz_global[1] > self.playground.W / 2:
            # Direction to start obstacle avoidance
            self.move = 2
            # Mirroir + décalage de 2*l + L
            for i in range(int(len(self.waypoints) / 3)):
                self.waypoints[3 * i + 1] = -self.waypoints[3 * i + 1] + 2 * self.l + self.L

    # ----------------------------------------------------------------------------------------#

    def follow_waypoints(self):
        """ Follow the waypoints given in self.waypoints"""
        # Min distance to consider point as reached
        epsilon = 0.05  # m
        modulus_error = np.sum((self.waypoints[0:3] - self.xyz_global) ** 2)  # Modulus of the error [m^2]

        # Check if the waypoint has been reached
        if modulus_error < epsilon ** 2:
            # If yes, check if it was the last waypoint in the list
            if len(self.waypoints) == 3:
                # If yes stop the search
                self.waypoints = None

                return False

            # Otherwise remove the first waypoint from the list
            self.waypoints = self.waypoints[3:len(self.waypoints)]

        # Set current waypoint to reach
        current_waypoint = self.waypoints[0:3]

        # Compute speed rate command
        self.xyz_rate_cmd = self.speed_controller.get_u(current_waypoint, self.xyz_global)

        return True

    # ----------------------------------------------------------------------------------------#
    # Function making the drone following the waypoints to search the pad while avoiding the obstacles
    # To tune :
    # VELOCITY_X : velocity in x-direction (forward or avoiding obstacle, increase if not avoiding obstacle well)
    # VELOCITY_Y : velocity in y-direction, volontary slow to detect the pad

    def obstacle_avoidance_searching(self, current_waypoint):

        VELOCITY_X = 0.4
        VELOCITY_Y = 0.4 # 0.5

        velocity_x = 0.0
        velocity_y = 0.0

        x_waypoint = current_waypoint[0]
        y_waypoint = current_waypoint[1]

        y_right = 0.3
        y_left = -0.5

        reached = False

        # Case moving to the right
        if self.move == 0:

            # While obstacle detected on the right, go forward
            #
            #       #########
            #    ^  #       #
            #    |  #       #
            #    |  #########

            if self.is_close_obs(self.range[4]):
                velocity_x = 2 * VELOCITY_X
                velocity_y = 0
                self.obs_y = self.xyz[1]
                self.avoiding = True

            # When nothing anymore on the right, go a bit on the right to detect the obstacle behind
            #     -->
            #       #########
            #       #       #
            #       #       #
            #       #########

            elif self.avoiding:
                if self.xyz[1] < (self.obs_y + 1.0):
                    velocity_x = 0
                    velocity_y = VELOCITY_Y
                else:
                    self.avoiding = False

            # When nothing more behind, go back to the trajectory
            #
            #       #########   |
            #       #       #   \/
            #       #       #
            #       #########

            elif (not self.is_close_obs(self.range[1]) and self.xyz[0] > (x_waypoint + 0.1) and self.avoiding == False):
                velocity_x = -VELOCITY_X
                velocity_y = 0

            # In other cases, go the right
            #         --->
            #       #########
            #       #       #      --->
            #       #       #
            #       #########

            else:
                velocity_x = 0
                velocity_y = VELOCITY_Y

            # Waypoint is reached
            if self.xyz[1] > y_waypoint:
                velocity_x = 0
                velocity_y = 0
                reached = True

        # Case move forward
        if self.move == 1:

            # Avoiding by the left
            #
            #       #########
            #    ^  #       #
            #    |  #       #
            #    |  #########
            #        <----

            if (self.is_close_obs(self.range[0]) and (self.xyz[1] >= 0.8)): 
                velocity_y = -VELOCITY_Y
                velocity_x = 0

            # Avoiding by the right
            #
            #       #########
            #       #       #   ^
            #       #       #   |
            #       #########   |
            #        ----->

            elif (self.is_close_obs(self.range[0]) and (self.xyz[1] < 0.8)): 
                velocity_x = 0
                velocity_y = VELOCITY_Y

            # Go forward if no obstacle
            else:
                velocity_x = VELOCITY_X
                velocity_y = 0

            # Waypoint is reached
            if self.xyz[0] > x_waypoint:
                velocity_x = 0
                velocity_y = 0
                reached = True

        # Case moving to the left
        if self.move == 2:

            # While obstacle detected on the left, go forward
            #
            #       #########   ^
            #       #       #   |
            #       #       #   |
            #       #########

            if self.is_close_obs(self.range[3]):
                velocity_x = 2 * VELOCITY_X
                velocity_y = 0
                self.obs_y = self.xyz[1]
                self.avoiding = True

            # When nothing more on the left, go a bit on the left to detect the obstacle behind
            #               <---
            #       #########
            #       #       #
            #       #       #
            #       #########

            elif self.avoiding:
                if self.xyz[1] > (self.obs_y - 1.0):
                    velocity_x = 0
                    velocity_y = -VELOCITY_Y
                else:
                    self.avoiding = False

            # When nothing more behind, go back to the trajectory
            #
            #   |   #########
            #   |   #       #
            #   \/  #       #
            #       #########

            elif (not self.is_close_obs(self.range[1]) and self.xyz[0] > (x_waypoint + 0.1) and self.avoiding == False):
                velocity_x = -VELOCITY_X
                velocity_y = 0

            # In other cases, go the left
            #         <---
            #       #########
            #  <--  #       #
            #       #       #
            #       #########

            else:
                velocity_x = 0
                velocity_y = -VELOCITY_Y

            # Waypoint reached
            if self.xyz[1] < y_waypoint:
                velocity_x = 0
                velocity_y = 0
                reached = True
        
        # Case moving to the back
        if self.move == 3:

            # Avoiding by the left
            #        <----
            #       #########
            #    |  #       #
            #    |  #       #
            #    \/ #########
            #

            if (self.is_close_obs(self.range[1]) and (self.xyz[1] >= 0.8)): 
                velocity_y = -VELOCITY_Y
                velocity_x = 0

            # Avoiding by the right
            #        ----->
            #       #########
            #       #       #   |
            #       #       #   |
            #       #########   \/
            #

            elif (self.is_close_obs(self.range[1]) and (self.xyz[1] < 0.8)):  
                velocity_x = 0
                velocity_y = VELOCITY_Y

            # Go forward if no obstacle
            else:
                velocity_x = -VELOCITY_X
                velocity_y = 0

            # Waypoint is reached
            if self.xyz[0] < x_waypoint:
                velocity_x = 0
                velocity_y = 0
                reached = True

        # Send the velocity command
        self.xyz_rate_cmd = [velocity_x, velocity_y, 0]
        # If waypoint reached, return True
        return reached

    # ------------------------------------------------------------------------------------------#
    # Function the drone going back to the starting point while avoiding obstacle
    # VELOCITY_X : velocity in x-direction, increase if want to go faster to the zone, but can break the avoidance
    # VELOCITY_Y : velocity in y-direction (to avoid obstacle), increase if does not avoid obstacle well

    def back_to_start(self):
        self.keep_flying = True

        epsilon = 0.3
        VELOCITY_X = 0.4
        VELOCITY_Y = 0.8

        velocity_x = 0.0
        velocity_y = 0.0

        # x > 0, not at the start point yet
        if self.xyz[0] > self.xyz0[0]:
            # If obstacle behind
            if self.is_close_obs(self.range[1]):
                # If y > 0, avoid obstacle by the left
                if self.xyz[1] > self.xyz0[1]:
                    velocity_x = 0.0
                    velocity_y = -VELOCITY_Y

                # If y < 0, avoid obstacle by the right
                else:
                    velocity_x = 0.0
                    velocity_y = VELOCITY_Y
            else:
                velocity_x = -VELOCITY_X
                velocity_y = 0.0

        # If x = 0, move to y = 0 (on the line of the starting point)
        else:
            # y > 0 -> go left while avoiding obstacle
            if self.xyz[1] > self.xyz0[1]:

                # While obstacle detected on the left, go forward
                #
                #       #########   ^
                #       #       #   |
                #       #       #   |
                #       #########

                if self.is_close_obs(self.range[3]):
                    velocity_x = VELOCITY_X
                    velocity_y = 0
                    self.obs_y = self.xyz[1]
                    self.avoiding = True

                # When nothing more on the left, go a bit on the left to detect the obstacle behind
                #               <---
                #       #########
                #       #       #
                #       #       #
                #       #########

                elif self.avoiding:
                    if self.xyz[1] > (self.obs_y - 1.0):
                        velocity_x = 0
                        velocity_y = -VELOCITY_Y
                    else:
                        self.avoiding = False

                # When nothing more behind, go back to the trajectory
                #
                #   |   #########
                #   |   #       #
                #   \/  #       #
                #       #########

                elif (not self.is_close_obs(self.range[1]) and (self.xyz[0] > self.xyz0[0]+0.05) and self.avoiding == False):
                    velocity_x = -VELOCITY_X
                    velocity_y = 0

                # If no obstacle, go to the left
                else:
                    velocity_x = 0
                    velocity_y = -VELOCITY_Y / 2

            # y < 0 : go right while avoiding obstacle
            else:

                # While obstacle detected on the right, go forward
                #
                #       #########
                #    ^  #       #
                #    |  #       #
                #    |  #########

                if self.is_close_obs(self.range[4]):
                    # print("Obstacle in view")
                    velocity_x = VELOCITY_X
                    velocity_y = 0
                    self.obs_y = self.xyz[1]
                    self.avoiding = True

                # When nothing more on the right, go a bit on the right to detect the obstacle behind
                #     -->
                #       #########
                #       #       #
                #       #       #
                #       #########

                elif self.avoiding:
                    # print("Avoiding")
                    if self.xyz[1] < (self.obs_y + 1.0):
                        # print("Avoiding 2")
                        velocity_x = 0
                        velocity_y = VELOCITY_Y / 2
                    else:
                        self.avoiding = False

                # When nothing more behind, go back to the trajectory
                #
                #       #########   |
                #       #       #   \/
                #       #       #
                #       #########

                elif (not self.is_close_obs(self.range[1]) and self.xyz[0] > self.xyz0[0]+0.05 and self.avoiding == False):
                    velocity_x = -VELOCITY_X
                    velocity_y = 0

                # When no obstacle, go to the right
                else:
                    velocity_x = 0
                    velocity_y = VELOCITY_Y

        # Arrived in searching zone
            if (abs(self.xyz[1]) < self.xyz0[1]):

                self.keep_flying = False
                velocity_x = 0.0
                velocity_y = 0.0

        # Send command
        self.xyz_rate_cmd = [velocity_x, velocity_y, 0]

        # Return false if in searching zone, true otherwise
        return self.keep_flying

    # ----------------------------------------------------------------------------------------#
    def detectEdge(self, edgeType=0):
        """Function to detect an edge using the z-ranger"""
        self.edgeFound = 0

        if (self.diffZ > self.edgeThresholdUp) and not self.edgeDetected:
            self.edgeDetected = True
            if (self.meanZ - min(self.queueZ)) > (max(self.queueZ) - self.meanZ):
                if edgeType == 1 or edgeType == 0:
                    self.edgeFound = 1
                else:
                    self.edgeFound = -1
            else:
                if edgeType == 2 or edgeType == 0:
                    self.edgeFound = 2
                else:
                    self.edgeFound = -2
            print(self.edgeFound)
        elif (self.diffZ <= self.edgeThresholdDown) and self.edgeDetected:
            self.edgeDetected = False

    # ----------------------------------------------------------------------------------------#
    def centering3(self):
        """Function to center the drone above the platform before landing"""
        if self.stateCentering == 0:
            self.detectEdge()
            if self.edgeDetected:
                self.playground.padEdge[0] = (self.xyz_global[0], self.xyz_global[1])

                self.xyz_rate_cmd_old = np.array(self.xyz_rate_cmd)
                self.xyz_rate_cmd_old = 0.2 * self.xyz_rate_cmd_old / np.max(np.abs(self.xyz_rate_cmd_old))
                self.xyz_rate_cmd = np.array([0.,0.,0.])

                self.stateCentering += 1
                time.sleep(self.edgeTimeDelay)  # for stabilization

        elif self.stateCentering == 1:
            self.xyz_rate_cmd = self.xyz_rate_cmd_old
            self.detectEdge()
            if self.edgeFound:# == 2:
                self.playground.padEdge[1] = (self.xyz_global[0], self.xyz_global[1])
                self.playground.padCenter = np.array((self.playground.padEdge[0] + self.playground.padEdge[1]) / 2)

                self.xyz_rate_cmd = np.array([0.,0.,0.])

                self.waypoints = np.array([self.playground.padCenter[0],
                                           self.playground.padCenter[1],
                                           self.default_height+0.1])

                self.stateCentering += 1
                time.sleep(self.edgeTimeDelay)  # for stabilization

        elif self.stateCentering == 2:
            if not self.follow_waypoints():
                # move perpendicular to previous direction
                self.xyz_rate_cmd = -1*self.xyz_rate_cmd_old
                self.stateCentering += 1

        elif self.stateCentering == 3:
            self.detectEdge()
            if self.edgeFound:# == 2:
                self.playground.padEdge[0] = (self.xyz_global[0], self.xyz_global[1])
                self.playground.padCenter = np.array((self.playground.padEdge[0] + self.playground.padEdge[1]) / 2)

                self.xyz_rate_cmd = np.array([0.,0.,0.])

                self.waypoints = np.array([self.playground.padCenter[0],
                                           self.playground.padCenter[1],
                                           self.default_height+0.1])

                self.stateCentering += 1
                time.sleep(self.edgeTimeDelay)  # for stabilization

        elif self.stateCentering == 4:
            if not self.follow_waypoints():
                # move perpendicular to previous direction
                self.xyz_rate_cmd = np.array([self.xyz_rate_cmd_old[1],
                                              self.xyz_rate_cmd_old[0],
                                              self.xyz_rate_cmd_old[2]])
                self.stateCentering += 1

        elif self.stateCentering == 5:
            self.detectEdge()
            if self.edgeFound:# == 2:
                self.playground.padEdge[2] = (self.xyz_global[0], self.xyz_global[1])

                if abs(self.xyz_rate_cmd[0]) > abs(self.xyz_rate_cmd[1]):
                    self.playground.padCenter[0] = self.playground.padEdge[2, 0] - np.sign(self.xyz_rate_cmd[0])*self.playground.padSize_x
                else:
                    self.playground.padCenter[1] = self.playground.padEdge[2, 1] - np.sign(self.xyz_rate_cmd[1])*self.playground.padSize_y

                self.xyz_rate_cmd_old = self.xyz_rate_cmd
                self.xyz_rate_cmd = np.array([0.,0.,0.])


                self.waypoints = np.array([self.playground.padCenter[0],
                                           self.playground.padCenter[1],
                                           self.default_height + 0.1])

                self.stateCentering += 1
                time.sleep(self.edgeTimeDelay)  # for stabilization

        elif self.stateCentering == 6:
            if not self.follow_waypoints():
                # move perpendicular to previous direction
                self.xyz_rate_cmd = -1*self.xyz_rate_cmd_old
                self.stateCentering += 1

        elif self.stateCentering == 7:
            self.detectEdge()
            if self.edgeFound:# == 2:
                self.playground.padEdge[3] = (self.xyz_global[0], self.xyz_global[1])

                if abs(self.xyz_rate_cmd[0]) > abs(self.xyz_rate_cmd[1]):
                    self.playground.padCenter[0] = (self.playground.padEdge[2, 0] + self.playground.padEdge[3, 0]) / 2
                else:
                    self.playground.padCenter[1] = (self.playground.padEdge[2, 1] + self.playground.padEdge[3, 1]) / 2

                self.xyz_rate_cmd_old = self.xyz_rate_cmd
                self.xyz_rate_cmd = np.array([0., 0., 0.])

                self.waypoints = np.array([self.playground.padCenter[0],
                                           self.playground.padCenter[1],
                                           self.default_height+0.1])
                self.waypoints = np.append(self.waypoints,
                                           [self.playground.padCenter[0], self.playground.padCenter[1], 0.15])

                self.stateCentering += 1
                time.sleep(self.edgeTimeDelay)  # for stabilization

        elif self.stateCentering == 8:
            if not self.follow_waypoints():
                self.centerReached = True
                self.xyz_rate_cmd = np.array([0., 0., 0.])
                self.stateCentering += 1

    # ----------------------------------------------------------------------------------------#

    def stateMachine(self, scf):
        """Main function that manages the transition between each state of the drone"""
        with MotionCommander(scf, default_height=self.default_height) as mc:
            while (self.is_not_close()):

                if self.state == 0:
                    # ---- Take off ----#

                    # default height has been reached -> Next state
                    if self.xyz[2] >= self.default_height:
                        self.state += 1

                        # test centering:
                        time.sleep(.5)

                elif self.state == 1:

                    # ---- Fly to zone 2 ----#

                    self.keep_flying = self.move_to_landing_zone()

                    if not self.keep_flying:
                        self.keep_searching = True
                        waypoint_drone = 0
                        self.state += 1

                elif self.state == 2 or self.state == 6:
                    # Soit on cherche la zone de landing soit on cherche la zone de départ

                    # ---- Search landing zone ----#

                    if self.waypoints is None and self.keep_searching == True:
                        # Wait to compute waypoints (searching path)
                        self.xyz_rate_cmd = np.array([0, 0, 0])

                        # self.set_spiral_waypoints()
                        if self.state == 2:
                            self.set_waypoints()
                        elif self.state == 6:
                            self.set_spiral_waypoints()
                            self.move = 0


                    change_waypoint = False

                    # From global frame to drone frame
                    
                    if self.waypoints is not None:
                        if self.state == 2 :
                            initial_pos = [self.xyz0[0], self.xyz0[1]]
                            waypoint_drone = [self.waypoints[0] - initial_pos[0], self.waypoints[1] - initial_pos[1]]
                        elif self.state == 6 :
                            waypoint_drone = [self.waypoints[0], self.waypoints[1]]

                        change_waypoint = self.obstacle_avoidance_searching(waypoint_drone)

                    if change_waypoint:
                        self.avoiding = False
                        self.waypoints = self.waypoints[3:len(self.waypoints)]
                        if self.waypoints is not None:
                            if self.state == 2 :
                                waypoint_drone = [self.waypoints[0] - initial_pos[0], self.waypoints[1] - initial_pos[1]]
                            elif self.state == 6 :
                                waypoint_drone = [self.waypoints[0], self.waypoints[1]]

                        # If right or left before, forward now
                        if self.state == 2 :
                            if self.move != 1:
                                self.move = 1

                            # If forward before, determine right or left frome y coordinate of next waypoint in drone frame
                            else:
                                if waypoint_drone[1] < 0:
                                    self.move = 2
                                else:
                                    self.move = 0
                        
                        elif self.state == 6 :
                            # If right before, forward next
                            self.move += 1
                            if self.move == 4 :
                                self.move = 0
                                
                    # Return true if we reached last waypoint, false otherwise
                    self.keep_searching = True

                    self.detectEdge()
                    if self.edgeDetected:
                        self.keep_searching = False
                        self.state += 1

                elif self.state == 3 or self.state == 7:
                    # ---- Search center of the landing zone ----#
                    self.centering3()
                    if self.centerReached:
                        if self.state == 3:  # Si on est au state 3, on s'est posé sur la zone d'arrivée -> On passe au state suivant
                            self.state += 1
                            self.stateCentering = 0
                            self.centerReached = False
                            self.playground.padCenter[0] = self.xyz_global[0]
                            self.playground.padCenter[1] = self.xyz_global[1]
                        else:
                            mc.land()
                            break  # Si on est au state 7, on est de retour à la zone de départ -> On coupe

                elif self.state == 4:
                    mc.land()
                    time.sleep(5.)
                    mc.take_off()
                    self.keep_flying = True
                    self.state += 1

                elif self.state == 5:
                    # ---- Back to start ---------------------#
                    self.back_to_start()
                    if not self.keep_flying:
                        #mc.land()
                        time.sleep(0.5)
                        self.state += 1
                        self.keep_searching = True

                    if self.xyz[0] < 1.5:
                        self.detectEdge()
                        if self.edgeDetected:
                            self.keep_flying = False
                            self.state = 7
                    
                else:
                    print("Woooooops invalid state")

                
                mc.start_linear_motion(self.xyz_rate_cmd[0], -self.xyz_rate_cmd[1], self.xyz_rate_cmd[2],
                                       self.rpy_rate_cmd[0])

                time.sleep(self.Te_loop)

    # ----------------------------------------------------------------------------------------#

    def run(self):
        """Function that setup and run the whole program"""
        print("Connection ..")

        with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Charles connecté, Charles content")

            print("Add config ..")
            scf.cf.log.add_config(self.log_position)
            scf.cf.log.add_config(self.log_multiranger)
            print("Add Callback ..")
            self.log_position.data_received_cb.add_callback(self.log_pos_callback)
            self.log_multiranger.data_received_cb.add_callback(self.log_multi_callback)

            time.sleep(1)

            print("Start dataflow")
            self.log_position.start()
            self.log_multiranger.start()

            print("Z'eeeeeeest parti")
            self.stateMachine(scf)

            print("Goodbye :'(")
            self.log_position.stop()
            self.log_multiranger.stop()


####################################### MAIN ##############################################

test = Charles()
test.run()

