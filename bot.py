import random
import numpy as np
import math
import tensorflow as tf
from collections import deque


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


class MyBot(BaseAgent):
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


    def setTarget(self):

        target = self.target_index
        target_location_info = None

        if target == 0:
            self.renderText("attack")
            target_location_info = self.shootBallTowardsTarget(
                Vec3(800, 5213, 321.3875), Vec3(-800, 5213, 321.3875))
        elif target == 1:
            self.renderText("defend")
            target_location_info = self.shootBallTowardsTarget(
                Vec3(10000, self.ball_location.y - 2000, self.ball_location.z),
                Vec3(-10000, self.ball_location.y - 2000, self.ball_location.z),
            )
        elif target == 100:
            self.renderText("random")

            height = 5120 - 1152
            width = 4096 - 1152

            lx = random.randint(-width, width)
            ly = random.randint(-height, height)
            location = Vec3(lx, ly, 0)
            dx = random.randint(-100, 100)
            dy = random.randint(-100, 100)
            dz = random.randint(-100, 100)
            direction = Vec3.normalized(Vec3(dx, dy, 0.01))

            target_location_info = [location, direction]

        elif target == 101:
            self.renderText("random driection")

            d1 = Vec3(random.randint(-100, 100),
                      random.randint(-100, 100), random.randint(-100, 100))
            d2 = Vec3(random.randint(-100, 100),
                      random.randint(-100, 100), random.randint(-100, 100))

            target_location_info = self.shootBallTowardsTarget(d1, d2)

        self.target_location_info = target_location_info

    def setPath(self, target_location, target_direction):

        self.renderer.draw_line_3d(Vec3(target_location.x, target_location.y, 0), Vec3(
            target_location.x, target_location.y, target_location.z+100), self.renderer.red())
        self.renderer.draw_line_3d(target_location, Vec3(
            target_location.x, target_location.y+100, target_location.z), self.renderer.red())
        self.renderer.draw_line_3d(target_location, Vec3(
            target_location.x+100, target_location.y, target_location.z), self.renderer.red())

        size = 7

        self.renderer.draw_rect_3d(Vec3(
            target_location.x, target_location.y, target_location.z), size, size, True, self.renderer.red())

        self.renderer.draw_line_3d(
            target_location, target_location + target_direction * 500, self.renderer.purple())

        path = self.computePossibleArcLineArcDrivePaths(
            target_location, target_direction)
        return path

    def setObjective(self):

        if(self.unforseenAction()):
            # new_target_index = learningAgent.getAction(packet)
            new_target_index = 0
            self.target_index = new_target_index
            self.maneuver_start = self.packet.game_info.seconds_elapsed
            self.createNewManeuver()
            return True
        else: return False

    def getThrottle(self, controls):

        path_length = self.path_length


        time_left = (self.maneuver_time - self.since_maneuver_start) / 10
        needed_speed = path_length / (time_left + 0.1)
        speed = self.car_forward_velocity
        diff = needed_speed - speed

        throttle = diff/1000 * 1.7

        if(throttle > 1):
            throttle = 1

        if(throttle < -1):
            throttle = -1

        controls.throttle = throttle
        return controls

    def createNewManeuver(self):

        v0 = self.car_forward_velocity
        t = 0
        d = Vec3.length(self.car_location - self.ball_location) 

        short_coefficient = 500
        long_coefficient = 2

        d = d * long_coefficient + short_coefficient / d


        max_speed = 1500

        while d > 0:
            d -= v0 / 10
            v0 += self.getAcceleration(v0) / 100

            if(v0 > max_speed):
                v0 = max_speed
            t += 1

        t = t * 2

        if(t > 60):
            t = 60


        self.maneuver_time = t



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


        return False

    #==============================|==============================#
    #=====================Target determining======================#
    #==============================|==============================#

    def shootBallTowardsTarget(self, left_most_target, right_most_target):



        ball_location = self.predictBallLocation(self.maneuver_time)

        car_to_ball = ball_location - self.car_location
        car_to_ball_direction = Vec3.normalized(car_to_ball)

        ball_to_left_target_direction = Vec3.normalized(
            left_most_target - ball_location)
        ball_to_right_target_direction = Vec3.normalized(
            right_most_target - ball_location)
        direction_of_approach = Vec3.clamp2D(
            direction=car_to_ball_direction, start=ball_to_left_target_direction, end=ball_to_right_target_direction)
        # offset would be 92.75 but is better with a greater value for arc line arc
        offset_ball_location = ball_location - direction_of_approach * 100


        return [offset_ball_location, direction_of_approach]

    #==============================|==============================#
    #========================Arc Line Arc=========================#
    #==============================|==============================#

    def computePossibleArcLineArcDrivePaths(self, target_location, target_direction):
        steering_radius = self.getSteeringRadius()

        if(steering_radius < self.min_rad):
            steering_radius = self.min_rad

        # self.renderer.draw_line_3d(target_location,target_location+ target_direction*600, self.renderer.red())

        # car circles
        car_direction = Orientation(self.car_rotation).forward
        # self.renderer.draw_line_3d(self.car_location,self.car_location+ car_direction*600, self.renderer.red())

        # car circle 1
        Mc1 = Circle()
        Mc1.location = Vec3.normalized(Vec3.cross(
            car_direction, Vec3(0, 0, 1))) * steering_radius + self.car_location
        Mc1.radius = steering_radius
        Mc1.rotation = -1
        # render
        Mc1.points = self.getPointsInSircle(11, Mc1.radius, Mc1.location)
        # self.renderer.draw_polyline_3d(Mc1.points, self.renderer.white())

        # car circle 2
        Mc2 = Circle()
        Mc2.location = Vec3.normalized(Vec3.cross(car_direction, Vec3(
            0, 0, 1))) * -steering_radius + self.car_location
        Mc2.radius = steering_radius
        Mc2.rotation = 1
        # render
        Mc2.points = self.getPointsInSircle(11, Mc2.radius, Mc2.location)
        # self.renderer.draw_polyline_3d(Mc2.points, self.renderer.white())

        # target circles
        # self.renderer.draw_line_3d(target_location,target_location+ target_direction*100, self.renderer.red())

        # target circle 1
        Mt1 = Circle()
        Mt1.location = Vec3.normalized(Vec3.cross(
            target_direction, Vec3(0, 0, 1))) * steering_radius + target_location
        Mt1.radius = steering_radius
        Mt1.rotation = -1
        # render
        Mt1.points = self.getPointsInSircle(11, Mt1.radius, Mt1.location)
        # self.renderer.draw_polyline_3d(Mt1.points, self.renderer.white())

        # target circle 2
        Mt2 = Circle()
        Mt2.location = Vec3.normalized(Vec3.cross(
            target_direction, Vec3(0, 0, 1))) * -steering_radius + target_location
        Mt2.radius = steering_radius
        Mt2.rotation = 1
        # render
        Mt2.points = self.getPointsInSircle(11, Mt2.radius, Mt2.location)
        # self.renderer.draw_polyline_3d(Mt2.points, self.renderer.white())

        possibleTangents = []

        # left to right
        possibleTangents.append(self.getCrossTangents(
            Mc1, Mt2, self.car_location, target_direction, target_location)[0])
        possibleTangents[0].name = "lr"
        # right to left
        possibleTangents.append(self.getCrossTangents(
            Mc2, Mt1, self.car_location, target_direction, target_location)[1])
        possibleTangents[1].name = "rl"
        # left to left
        possibleTangents.append(self.getStraightTangents(
            Mc1, Mt1, self.car_location, target_direction, target_location)[0])
        possibleTangents[2].name = "ll"
        # right to right
        possibleTangents.append(self.getStraightTangents(
            Mc2, Mt2, self.car_location, target_direction, target_location)[1])
        possibleTangents[3].name = "rr"

        best_path = ArcLineArcPath()

        for tangent in possibleTangents:
            # self.renderer.draw_line_3d(tangent.start, tangent.end, self.renderer.white())
            # self.renderer.draw_line_3d(tangent.start, tangent.circle1_center, self.renderer.white())
            # self.renderer.draw_line_3d(car_location, tangent.circle1_center, self.renderer.white())
            # if(tangent.possible):
            # self.renderer.draw_line_3d(tangent.start, tangent.end, self.renderer.white())
            c1_arc_angle = Vec3.angle(Vec3.flat(tangent.start - tangent.circle1_center), Vec3.flat(
                self.car_location - tangent.circle1_center)) * 180/math.pi
            c1_radius = Vec3.length(
                Vec3.flat(tangent.start - tangent.circle1_center))
            c2_arc_angle = Vec3.angle(Vec3.flat(tangent.end - tangent.circle2_center), Vec3.flat(
                target_location - tangent.circle2_center)) * 180/math.pi
            c2_radius = Vec3.length(
                Vec3.flat(tangent.end - tangent.circle2_center))

            if (tangent.start.x - tangent.circle1_center.x)*(self.car_location.y - tangent.circle1_center.y) - (tangent.start.y - tangent.circle1_center.y)*(self.car_location.x - tangent.circle1_center.x) > 0:
                if(tangent.name == "rl" or tangent.name == "rr"):
                    c1_arc_angle = 360 - c1_arc_angle
            else:
                if(tangent.name == "lr" or tangent.name == "ll"):
                    c1_arc_angle = 360 - c1_arc_angle

            if (tangent.end.x - tangent.circle2_center.x)*(target_location.y - tangent.circle2_center.y) - (tangent.end.y - tangent.circle2_center.y)*(target_location.x - tangent.circle2_center.x) > 0:
                if(tangent.name == "rl" or tangent.name == "rr"):
                    c2_arc_angle = 360 - c2_arc_angle
            else:
                if(tangent.name == "lr" or tangent.name == "ll"):
                    c2_arc_angle = 360 - c2_arc_angle

            c1_arc_length = c1_arc_angle/360 * 2*math.pi * c1_radius
            c2_arc_length = c2_arc_angle/360 * 2*math.pi * c2_radius

            tangent_length = Vec3.length(tangent.end - tangent.start)
            if(Vec3.length(self.car_location - tangent.end) < 200):
                c1_arc_angle = 0
                c1_arc_length = 0
            elif(Vec3.length(tangent.circle1_center - tangent.circle2_center) < 200):
                c2_arc_angle = 0
                c2_arc_length = 0

            arc_line_arc_length = c1_arc_length + tangent_length + c2_arc_length

            if(self.checkIfOutOfMap([tangent.start, tangent.end])):
                tangent.possible = False

            if best_path.length > arc_line_arc_length and tangent.possible:
                best_path.length = arc_line_arc_length

                best_path.start = self.car_location
                best_path.tangent_start = tangent.start
                best_path.tangent_end = tangent.end
                best_path.tangent_length = Vec3.length(
                    tangent.end - tangent.start)
                best_path.end = target_location

                best_path.name = tangent.name
                best_path.c1_radius = c1_radius
                best_path.c2_radius = c2_radius
                best_path.c1_center = tangent.circle1_center
                best_path.c2_center = tangent.circle2_center
                best_path.c1_angle = c1_arc_angle
                best_path.c2_angle = c2_arc_angle
                best_path.c1_length = c1_arc_length
                best_path.c2_length = c2_arc_length

        # self.renderer.draw_polyline_3d([best_path.start, best_path.tangent_start, best_path.tangent_end, best_path.end], self.renderer.red())

        return(best_path)

    def getCrossTangents(self, C1, C2, car_location, target_direction, target_location):

        # middle circle
        C3 = Circle()
        C3.location = C1.location + (C2.location - C1.location)*0.5
        C3.radius = Vec3.length((C2.location - C1.location)*0.5)

        # bigger car circle
        C4 = Circle()
        C4.location = C1.location
        C4.radius = C1.radius + C2.radius

        # bigger target circle
        C5 = Circle()
        C5.location = C2.location
        C5.radius = C1.radius + C2.radius

        C4intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C4.location.x, C4.location.y, C4.radius)
        C5intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C5.location.x, C5.location.y, C5.radius)

        C1g1 = Vec3(C4intersections[0], C4intersections[1], 0)
        C1g2 = Vec3(C4intersections[2], C4intersections[3], 0)
        C2g1 = Vec3(C5intersections[0], C5intersections[1], 0)
        C2g2 = Vec3(C5intersections[2], C5intersections[3], 0)

        C1t1 = Vec3.normalized(C1g1 - C1.location)*C1.radius + C1.location
        C2t1 = Vec3.normalized(C2g1 - C2.location)*C2.radius + C2.location

        C1t2 = Vec3.normalized(C1g2 - C1.location)*C1.radius + C1.location
        C2t2 = Vec3.normalized(C2g2 - C2.location)*C2.radius + C2.location

        C1t1.z = 4
        C1t2.z = 4
        C2t1.z = 4
        C2t2.z = 4

        tangent1 = Tangent()
        tangent1.start = C1t1
        tangent1.end = C2t1
        tangent1.circle1_center = C1.location
        tangent1.circle2_center = C2.location

        tangent1.possible = C4intersections[4]

        tangent2 = Tangent()
        tangent2.start = C1t2
        tangent2.end = C2t2
        tangent2.circle1_center = C1.location
        tangent2.circle2_center = C2.location

        tangent2.possible = C5intersections[4]
        return[tangent1, tangent2]

    def getStraightTangents(self, C1, C2, car_location, target_direction, target_location):

        C1.location.z = 0
        C2.location.z = 0

        # middle circle
        C3 = Circle()
        C3.location = C1.location + (C2.location - C1.location)*0.5
        C3.radius = Vec3.length((C2.location - C1.location)*0.5)

        # bigger car circle
        C4 = Circle()
        C4.location = C1.location
        C4.radius = C1.radius - C2.radius + 1

        # bigger target circle
        C5 = Circle()
        C5.location = C2.location
        C5.radius = C2.radius - C1.radius + 1

        C4intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C4.location.x, C4.location.y, C4.radius)
        C5intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C5.location.x, C5.location.y, C5.radius)

        C1g1 = Vec3(C4intersections[0], C4intersections[1], 0)
        C1g2 = Vec3(C4intersections[2], C4intersections[3], 0)
        C2g1 = Vec3(C5intersections[0], C5intersections[1], 0)
        C2g2 = Vec3(C5intersections[2], C5intersections[3], 0)

        C1t1 = Vec3.normalized(C1g1 - C1.location)*C1.radius + C1.location
        C2t2 = Vec3.normalized(C2g2 - C2.location)*C2.radius + C2.location

        C2t1 = Vec3.normalized(C2g1 - C2.location)*C2.radius + C2.location
        C1t2 = Vec3.normalized(C1g2 - C1.location)*C1.radius + C1.location

        C1t1.z = 4
        C1t2.z = 4
        C2t1.z = 4
        C2t2.z = 4

        tangent1 = Tangent()
        tangent1.start = C1t1
        tangent1.end = C2t2
        tangent1.circle1_center = C1.location
        tangent1.circle2_center = C2.location
        tangent1.possible = True

        tangent2 = Tangent()
        tangent2.start = C1t2
        tangent2.end = C2t1
        tangent2.circle1_center = C1.location
        tangent2.circle2_center = C2.location
        tangent2.possible = True

        return[tangent1, tangent2]

    def getIntersections(self, x0, y0, r0, x1, y1, r1):

        d = math.sqrt((x1-x0)**2 + (y1-y0)**2)

        a = (r0**2-r1**2+d**2)/(2*d)
        if(r0**2-a**2 < 0):
            return (0, 0, 0, 0, False)
        h = math.sqrt(r0**2-a**2)
        x2 = x0+a*(x1-x0)/d
        y2 = y0+a*(y1-y0)/d
        x3 = x2+h*(y1-y0)/d
        y3 = y2-h*(x1-x0)/d

        x4 = x2-h*(y1-y0)/d
        y4 = y2+h*(x1-x0)/d

        return (x3, y3, x4, y4, True)

    def isNearWall(self):
        x = self.ball_location.x
        y = self.ball_location.y

        goals = [Vec3(0, 5213, 0), Vec3(0, -5213, 0)]
        near = x > 3800 or x < -3800 or y > 4800 or y < -4800
        near_goal = Vec3.length(self.ball_location - goals[0]) < 1500 or Vec3.length(self.ball_location - goals[1]) < 1500

        print(near_goal)

        if not near_goal:
            return near
        else: return False

    def getArcLineArcControllerState(self, path):





        steer = 0
        if(path.c1_length < 100 and path.tangent_length < 100):
            steer = steer_toward_target(self.my_car, path.end)
        elif(path.c1_length < 100):
            steer = steer_toward_target(self.my_car, path.tangent_end)
        else:
            steer = steer_toward_target(self.my_car, path.tangent_start)

        mult = 1
        rad = self.getSteeringRadius()
        if(rad < self.min_rad):
            mult = 1 / self.min_rad * rad

        return(steer * mult)

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

    def getPointsInSircle(self, every, radius, center):
        circle_positions = []
        for i in range(every + 1):

            angle = 2 * math.pi / every * (i+1)
            location = Vec3(radius * math.sin(angle),
                            radius * -math.cos(angle), 0) + center
            location.z = 4
            circle_positions.append(location)
        return(circle_positions)

    def renderArcLineArcPath(self, path):

        p = self.getPointsInSircle(20, path.c1_radius, path.c1_center)
        self.renderer.draw_polyline_3d(p, self.renderer.purple())
        self.renderer.draw_line_3d(
            path.tangent_start, path.tangent_end, self.renderer.purple())
        p = self.getPointsInSircle(20, path.c2_radius, path.c2_center)
        self.renderer.draw_polyline_3d(p, self.renderer.purple())

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


class Circle:
    def __init__(self):
        self.location = Vec3(0, 0, 0)
        self.radius = 0
        self.points = []
        self.rotation = 0


class Tangent:
    def __init__(self):
        self.name = ""
        self.circle1_center = Vec3(0, 0, 0)
        self.start = Vec3(0, 0, 0)

        self.circle2_center = Vec3(0, 0, 0)
        self.end = Vec3(0, 0, 0)

        self.possible = False


class ArcLineArcPath:
    def __init__(self):
        self.length = 10000000
        self.tangent_length = 0
        self.c1_length = 0
        self.c2_length = 0
        self.c1_radius = 0
        self.c2_radius = 0
        self.c1_angle = 0
        self.c2_angle = 0
        self.name = ""

        self.start = Vec3(0, 0, 0)
        self.tangent_start = Vec3(0, 0, 0)
        self.tangent_end = Vec3(0, 0, 0)
        self.end = Vec3(0, 0, 0)
