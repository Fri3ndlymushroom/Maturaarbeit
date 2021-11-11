import random
import math


from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from util.orientation import Orientation

from nn import learningAgent
from objective_module import Objective 
from target_module import Target
from path_module import Path
from controlls_module import Controlls



class MyBot(BaseAgent, Objective, Target, Path, Controlls):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

        # General
        self.frame = 0
        self.packet = None

        # Game Info
        self.goals = [0, 0]

        # Field Info
        self.packet = None
        self.my_car = None
        self.car_location = None
        self.car_rotation = None
        self.car_velocity = None
        self.car_forward_velocity = None
        self.ball_location = None
        self.ball_velocity = None
        self.ball_rotation = None

        # Objective
        self.target_index = 0

        # Target
        self.target_location_info = [Vec3(0, 0, 0), Vec3(0, 0, 0)]

        # Path
        self.path_length = 0

        # Tracked
        self.maneuver_time = 0  # time given for maneuver to execute
        self.maneuver_start = 0  # maneuver start time
        self.since_maneuver_start = 0  # passed time

        self.last_prediction = None
        self.last_time = None

        self.min_rad = 700

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.renderer.begin_rendering()
        self.packet = packet
        self.my_car = packet.game_cars[self.index]
        self.car_location = Vec3(self.my_car.physics.location)
        self.car_rotation = self.my_car.physics.rotation
        self.car_velocity = Vec3(self.my_car.physics.velocity)
        self.car_forward_velocity = Vec3.dot(
            self.car_velocity, Vec3.normalized(Orientation(self.car_rotation).forward))
        self.ball_location = Vec3(packet.game_ball.physics.location)
        self.ball_velocity = Vec3(packet.game_ball.physics.velocity)
        self.ball_rotation = packet.game_ball.physics.rotation

        self.frame += 1

        # rendering
        self.chat()
        self.predictBallPath()

        # if a sequence is running further execute it
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Track
        self.trackGame()

        # decision
        got_new_objective = self.setObjective()

        if(got_new_objective):
            self.setTarget()

        [target_location, target_direction] = self.target_location_info
        path = self.setPath(target_location, target_direction)

        # execution & controls
        controls = SimpleControllerState()

        # no possible path
        if(path.name == ""):
            controls.throttle = 1
            controls.steer = 1
            return(controls)

        self.renderArcLineArcPath(path)
        self.path_length = path.length
        controls.steer = self.getArcLineArcControllerState(path)
        controls = self.getThrottle(controls)
        if(Vec3.length(self.car_location - self.ball_location) < 185):
            return self.begin_front_flip(self.packet)

        self.renderer.end_rendering()




        # to remove ball from unpractical edges
        if self.isNearWall():
            controls.steer = steer_toward_target(self.my_car, self.ball_location)
            controls.throttle = 1


        return controls

    def trackGame(self):
        # Time
        self.since_maneuver_start = -1 * \
            (self.maneuver_start*10 - self.packet.game_info.seconds_elapsed*10)








    #==============================|==============================#
    #=====================Situation assessment====================#
    #==============================|==============================#

    def unforseenAction(self):

        if self.last_prediction == None or self.since_maneuver_start > self.maneuver_time:
            self.last_prediction = self.get_ball_prediction_struct().slices
            self.last_time = self.packet.game_info.seconds_elapsed
            return True

        time = self.packet.game_info.seconds_elapsed
        delta_time = round(359/60*(time - self.last_time) * 10)

        if(delta_time > 1):
            
            prediction = self.get_ball_prediction_struct().slices
            last_prediction = self.last_prediction

            new_prediction = prediction[100].physics.location
            old_prediction = last_prediction[100-delta_time].physics.location

            deviation = Vec3.length(Vec3(new_prediction.x, new_prediction.y, new_prediction.z) - Vec3(
                old_prediction.x, old_prediction.y, old_prediction.z))

            self.last_prediction = self.get_ball_prediction_struct().slices
            self.last_time = self.packet.game_info.seconds_elapsed
            if(deviation > 35):
                return True


        # not reachable

        v = self.car_forward_velocity
        l = self.path_length
        t = round(self.maneuver_time - self.since_maneuver_start)



        return False

    #==============================|==============================#
    #=====================Target determining======================#
    #==============================|==============================#



    #==============================|==============================#
    #========================Arc Line Arc=========================#
    #==============================|==============================#



    def isNearWall(self):
        x = self.ball_location.x
        y = self.ball_location.y

        goals = [Vec3(0, 5213, 0), Vec3(0, -5213, 0)]
        near = x > 3800 or x < -3800 or y > 4800 or y < -4800
        near_goal = Vec3.length(self.ball_location - goals[0]) < 1500 or Vec3.length(self.ball_location - goals[1]) < 1500

        if not near_goal:
            return near
        else: return False



    #==============================|==============================#
    #==========================Movement===========================#
    #==============================|==============================#

    def begin_front_flip(self, packet):
        # Send some quickchat just for fun

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.02,
                        controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.02,
                        controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(
                jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

    def begin_speed_flip(self, packet):
        # Send some quickchat just for fun

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(
                duration=0.7, controls=SimpleControllerState(boost=True)),
            ControlStep(duration=0.05,
                        controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.2, controls=SimpleControllerState(
                jump=False, yaw=-1, pitch=-1)),
            ControlStep(duration=0.1, controls=SimpleControllerState(
                jump=True, yaw=-1, pitch=-1, )),
            ControlStep(duration=0.05,
                        controls=SimpleControllerState(yaw=-1, pitch=1)),
            ControlStep(duration=0.8, controls=SimpleControllerState(yaw=0.5)),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

    #==============================|==============================#
    #==========================Game Info==========================#
    #==============================|==============================#

    def predictBallPath(self):
        ball_prediction = self.get_ball_prediction_struct()

        loc1 = None
        loc2 = None

        if ball_prediction is not None:
            for i in range(0, ball_prediction.num_slices):
                prediction_slice = ball_prediction.slices[i]
                location = prediction_slice.physics.location

                if (i-1) % 2 == 0:
                    loc2 = location
                else:
                    loc1 = location

                if loc1 is not None and loc2 is not None:
                    self.renderer.draw_line_3d(
                        loc1, loc2, self.renderer.yellow())

    def predictBallLocation(self, time):
        if time > 60:
            time = 60

        time = round(359/60*time)

        ball_prediction = self.get_ball_prediction_struct()

        ball_prediction_time = ball_prediction.slices[time].physics.location

        return Vec3(ball_prediction_time.x, ball_prediction_time.y, ball_prediction_time.z)

    def getSteeringRadius(self):
        velocity = self.car_forward_velocity
        """
        curvature = 0.0069
        if velocity > 2300: curvature = 0.0008
        elif velocity > 1750: curvature = 0.0011
        elif velocity > 1500: curvature = 0.001375
        elif velocity > 1000: curvature = 0.00235
        elif velocity > 500: curvature = 0.00598
        """
        val1 = 0
        val2 = 0
        val3 = 0
        val4 = 0

        if (velocity <= 0):
            return 1/0.0069
        if(velocity >= 1750 and velocity <= 2300):
            val1 = 1750
            val2 = 2300
            val3 = 0.0011
            val4 = 0.00088
        elif(velocity >= 1500 and velocity <= 1750):
            val1 = 1500
            val2 = 1750
            val3 = 0.001375
            val4 = 0.0011
        elif(velocity >= 1000 and velocity <= 1500):
            val1 = 1000
            val2 = 1500
            val3 = 0.00235
            val4 = 0.001375
        elif(velocity >= 500 and velocity <= 1000):
            val1 = 500
            val2 = 1000
            val3 = 0.00398
            val4 = 0.00235
        elif(velocity >= 0 and velocity <= 500):
            val1 = 0
            val2 = 500
            val3 = 0.0069
            val4 = 0.00398

        percentage = 1 / (val2 - val1) * (velocity - val1)
        curvature = val3 - (val3 - val4) * percentage

        radius = 1/curvature

        return(radius)

    def getAcceleration(self, car_velocity):

        acceleration = 0
        vals = []

        if(car_velocity < 1400):
            vals = [[0, 1600],
                    [1400, 160]]
        elif(car_velocity < 1410):
            vals = [[1400, 160],
                    [1410, 0]]
        elif(car_velocity < 2300):
            vals = [[1410, 0],
                    [2300, 0]]

        percentage = 1 / (vals[1][0] - vals[0][0]) * \
            (car_velocity - vals[0][0])

        acceleration = vals[0][1] - (vals[0][1] - vals[1][1]) * percentage

        return acceleration

    def checkIfOutOfMap(self, locs):
        possible = True
        for loc in locs:
            # general bounds
            if(loc.x > 4096 or loc.x < -4096 or loc.y > 5120 + 880 or loc.y < -5120 - 880):
                possible = False
            # goals
            if(loc.y > 5120 or loc.y < -5120):
                if(loc.x > 893 or loc.x < -893):
                    possible = False
            # edges
            if(
                # upper left
                (loc.y > 5120 - 1152 and loc.x > 4096 - 1152) or
                # upper right
                (loc.y > 5120 - 1152 and loc.x < -4096 + 1152) or
                # down left
                (loc.y < -5120 + 1152 and loc.x > 4096 - 1152) or
                # down right
                (loc.y < -5120 + 1152 and loc.x < -4096 + 1152)
            ):
                possible = False
        return not possible

    #==============================|==============================#
    #==========================Rendering==========================#
    #==============================|==============================#

    def renderText(self, text):
        text = str(text)
        self.renderer.draw_string_2d(4, 4, 2, 2, text, self.renderer.white())



    def chat(self):
        message_index = round(random.random()*6)

        if(self.my_car.score_info.goals > self.goals[0]):
            if(message_index == 0):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Toxic_WasteCPU)
            if(message_index == 1):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Compliments_Pro)
            if(message_index == 2):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Exclamation_Yeet)
            if(message_index == 3):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Information_TakeTheShot)
            if(message_index == 4):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Compliments_WhatASave)
            if(message_index == 5):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Reactions_Calculated)
            if(message_index == 6):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Apologies_NoProblem)
            self.goals[0] = self.my_car.score_info.goals
        if(self.packet.game_cars[1].score_info.goals > self.goals[1]):

            if(message_index == 0):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Useful_Bumping)
            if(message_index == 1):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Compliments_SkillLevel)
            if(message_index == 2):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Excuses_Lag)
            if(message_index == 3):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Apologies_Whoops)
            if(message_index == 4):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Compliments_Thanks)
            if(message_index == 5):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Reactions_NoWay)
            if(message_index == 6):
                self.send_quick_chat(
                    team_only=False, quick_chat=QuickChatSelection.Custom_Compliments_TinyChances)
            self.goals[1] = self.packet.game_cars[1].score_info.goals


