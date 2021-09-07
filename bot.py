from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from util.orientation import Orientation

import math


from tkinter import Tk, Text, END
#root = Tk()
#text = Text(root)
#text.pack()

#debugLog("start")

import sys
#debugLog( "python: " + sys.version)
import tensorflow as tf
#debugLog( "tensorflow: " + tf.__version__)

#debugLog( "keras: " + tf.keras.__version__)

import time

import logging
tf.get_logger().setLevel(3)

from collections import deque
import numpy as np
import random




class ModelAgent():
    def __init__ (self):
        self.REPLAY_MEMORY_SIZE = 50000 # mit wie vielen der letzten actions das hauptmodel gefitted werden soll
        self.MIN_REPLAY_MEMORY_SIZE = 1000
        self.UPDATE_TARGET_EVERY = 5
        self.MODEL_NAME = "model"
        self.MINIBATCH_SIZE = 64
        self.DISCOUNT = 0.99
        self.ACTION_SPACE_SIZE = 4
        # possible actions
        # throttle forwards
        # throttle backwards
        # steer right
        # steer left
        # pich down
        # pitch up
        # yaw left
        # yaw right
        # roll left
        # roll right
        # jump
        # boost
        # handbrake
        # no action

        # es gibt zwei models weil das model sonst für jeden schritt overfitted. Desswegen wird das eine immer nur alle x schritte an das andere angepasst. Man predicted aber immer am hauptmodel
        # hauptmodel
        self.model = self.create_model()

        # target model dass die gleichen weights hat # das ist das model dass alle x schritte gefitted wird
        self.target_model = self.create_model()


        self.target_model.set_weights(self.model.get_weights())

        # erstellt eine liste für die letzten x inputs um overfitting weiter zu verhindern
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)   

        # zählt wann das target network mit dem haupt network synchronisieren soll
        self.target_update_counter = 0

    def create_model(self):
        model = tf.keras.models.Sequential()

        # inputs sind: xDelta, yDelta, zDelta, car speed
        model.add(tf.keras.layers.Dense(9, input_shape=(9,), activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(self.ACTION_SPACE_SIZE, activation="linear"))# anzahl möglicher output controlls
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        return model
    

    
    # updated das reply mermory
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    # q values berechnen
    def get_qs(self, state):
        qs= self.model.predict(np.array(state).reshape(-1, *np.array(state).shape)/1)[0]
        return qs
    
    # jeder step soll das netzwerk trainiert werdenfg
    def train(self, terminal_state, step):

        
        # soll abbrechen wenn das replay memory zu klein ist
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # batch mit der grösse MINIBATCH_SIZE vom replay memory
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        

        
        # get q valuess
        current_states = np.array([transition[0] for transition in minibatch])/1
        current_qs_list = self.model.predict(current_states)

        # zukünftige q values
        new_current_states = np.array([transition[3] for transition in minibatch])/1
        future_qs_list = self.target_model.predict(new_current_states)


        
        


        # update model
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/1, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[] if terminal_state else None)
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



agent = ModelAgent()



class QLearningAgent:
    def __init__(self):
        self.step = 1
        self.total_step = 1
        self.STEPS_PER_EPISODE = 200
        self.done_accuracy = []
        self.last_distance = 0
        self.done = False

        # epsilon
        self.epsilon = 1
        self.EPSILON_DECAY = 0.99975
        self.MIN_EPSILON = 0.001

        # action space size
        self.ACTION_SPACE_SIZE = 4
        # attack
        # defend
        # get boost

        #penaltys
        self.MOVE_PENALTY = 1
        self.DISTANCE_PENALTY_MULTIPLICATOR = 5


    
    def getAction(self, packet):

        packet = packet
        self_car = packet.game_cars[0]
        enemy_car = packet.game_cars[1]
        self_car_location = Vec3(self_car.physics.location)
        enemy_car_location = Vec3(enemy_car.physics.location)
        ball_location = Vec3(packet.game_ball.physics.location)

        # Mit dem Score im Spiel wird ermittelt wie gut der bot ist.
        #  Wenn also zb. 3/7 steht ist die scoreRatio -4 und somit kann der bot gut trainiert werdne
        scoreRatio = packet.teams[0].score - packet.teams[1].score

        # wenn self.step == STEPS_PER_EPISODE ist sollen variabeln zurückgesetz werden
        self.manageEpisodes()

        self.state_now = [
            self_car_location.x, self_car_location.y, self_car_location.z,
            enemy_car_location.x, enemy_car_location.y, enemy_car_location.z,
            ball_location.x, ball_location.y, ball_location.z
        ]
        
        # den letzten schritt beurteilen
        if self.should_train:

            self.step_reward -= self.MOVE_PENALTY

            self.step_reward += scoreRatio * 1000
            self.episode_reward += self.step_reward


            if self.step == self.STEPS_PER_EPISODE-1:
                self.done = True

        
            agent.update_replay_memory((self.old_state, self.action, self.step_reward, self.state_now, self.done))
            agent.train(self.done, self.step)
        
        
        # neuen schritt machen
        if np.random.random() > self.epsilon:
            # Get action from Q table
            self.action = np.argmax(agent.get_qs(self.state_now))
        else:
            # Get random action
            self.action = np.random.randint(0, self.ACTION_SPACE_SIZE)

        
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon = max(self.MIN_EPSILON, self.epsilon)
        
        #self.action = np.argmax(agent.get_qs(self.state_now))
        self.old_state = self.state_now

        self.step += 1
        self.total_step += 1

        #if (self.total_step % 200 == 0): 
            # step





        return self.action



    def manageEpisodes(self):
        if self.step == self.STEPS_PER_EPISODE or self.done:
            self.step = 1

        if self.step == 1:
            self.episode_reward = 0
            self.done = False

            # verhindert, dass beim ersten schritt trainiert wird
            self.should_train = False
        else:
            self.should_train = True
        
        self.step_reward = 0



    
        



learningAgent = QLearningAgent()







class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

        self.first_call = True




        self.maneuver = Maneuver()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.renderer.begin_rendering()

        self.packet = packet
        
        # get info
        self.index = 0
        # field info
        field_info = self.get_field_info()
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_rotation = my_car.physics.rotation
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_rotation = packet.game_ball.physics.rotation

        self.predictBallPath()


        
        # execution & controls
        controls = SimpleControllerState()


        if not self.checkIfManeuverFinished() and not self.checkIfUnforseenAction() and not self.first_call:
            # further execute maneuver
            controls.steer = self.getArcLineArcControllerState(self.maneuver.path, my_car)
            controls.throttle = 1
            controls.boost = False
            self.renderArcLineArcPath(self.maneuver.path, 0, 0)
            
        else:
            #get new maneuver
            #self.maneuver.target = learningAgent.getAction(packet)
            self.maneuver.target = 0
            self.getTargetLocation()
            self.maneuver.path = self.computePossibleArcLineArcDrivePaths(self.packet, self.maneuver.target_location[0], self.maneuver.target_location[1])




        self.renderer.end_rendering()
        self.first_call = False
        return controls



    def getTargetLocation(self):
        if(self.maneuver.target == 0):
            #attack
            self.maneuver.target_location = self.shootBallTowardsTarget(self.packet, Vec3(800, 5213, 321.3875), Vec3(-800, 5213, 321.3875))


    def checkIfManeuverFinished(self):
        if(self.maneuver.path):
            if(self.maneuver.path.phase == 3):
                return True
        
        return False
    def checkIfUnforseenAction(self):
        return False
    #==============================|==============================#
    #=====================Target determining======================#
    #==============================|==============================#

    def shootBallTowardsTarget(self, packet, left_most_target, right_most_target):
        ball_location = Vec3(packet.game_ball.physics.location)
        car_location = Vec3(packet.game_cars[self.index].physics.location)


        distance = Vec3.length(ball_location - car_location) * 2

        ball_location = self.predictBallLocation(60)




        """
        car_velocity = Vec3.dot(packet.game_cars[self.index].physics.velocity, Vec3.normalized(Orientation(packet.game_cars[self.index].physics.rotation).forward))
        target_reached = False
        time = 0
        while not target_reached:

            distance -= car_velocity / 10
            car_velocity += self.getAcceleration(car_velocity) / 100

            if(car_velocity > 1410): car_velocity = 1410

            time += 1
            if(distance < 0):target_reached = True
        
        """


        self.renderer.draw_line_3d(Vec3(ball_location.x, ball_location.x, 10000), ball_location, self.renderer.blue())
        # max speed = 1410


        

        car_to_ball = ball_location - car_location
        car_to_ball_direction = Vec3.normalized(car_to_ball)

        ball_to_left_target_direction = Vec3.normalized(left_most_target - ball_location)
        ball_to_right_target_direction = Vec3.normalized(right_most_target - ball_location)
        direction_of_approach = Vec3.clamp2D(direction=car_to_ball_direction, start=ball_to_left_target_direction, end=ball_to_right_target_direction)
        #offset would be 92.75 but is better with a greater value for arc line arc
        offset_ball_location = ball_location - direction_of_approach * 150
        return [offset_ball_location, direction_of_approach]








    #==============================|==============================#
    #========================Arc Line Arc=========================#
    #==============================|==============================#

    def computePossibleArcLineArcDrivePaths(self, packet, target_location, target_direction):
        #my_car = packet.game_cars[self.index]
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_rotation = my_car.physics.rotation
        car_velocity = Vec3(my_car.physics.velocity)
        #steering_radius = self.getSteeringRadius(car_velocity, car_rotation)
        steering_radius = 1 / 0.001375

        self.renderer.draw_line_3d(target_location,target_location+ target_direction*600, self.renderer.red())


        # car circles
        car_direction = Orientation(car_rotation).forward
        self.renderer.draw_line_3d(car_location,car_location+ car_direction*600, self.renderer.red())
        

        # car circle 1
        Mc1 = Circle()
        Mc1.location = Vec3.normalized(Vec3.cross(car_direction, Vec3(0, 0, 1))) * steering_radius + car_location
        Mc1.radius = steering_radius
        Mc1.rotation = -1
        # render
        Mc1.points = self.getPointsInSircle(11, Mc1.radius, Mc1.location)
        #self.renderer.draw_polyline_3d(Mc1.points, self.renderer.white())

        # car circle 2
        Mc2 = Circle()
        Mc2.location = Vec3.normalized(Vec3.cross(car_direction, Vec3(0, 0, 1))) * -steering_radius + car_location
        Mc2.radius = steering_radius
        Mc2.rotation = 1
        # render
        Mc2.points = self.getPointsInSircle(11, Mc2.radius, Mc2.location)
        #self.renderer.draw_polyline_3d(Mc2.points, self.renderer.white())


        # target circles
        self.renderer.draw_line_3d(target_location,target_location+ target_direction*100, self.renderer.red())


        # target circle 1
        Mt1 = Circle()
        Mt1.location = Vec3.normalized(Vec3.cross(target_direction, Vec3(0, 0, 1))) * steering_radius + target_location
        Mt1.radius = steering_radius   
        Mt1.rotation = -1
        # render
        Mt1.points = self.getPointsInSircle(11, Mt1.radius, Mt1.location)
        #self.renderer.draw_polyline_3d(Mt1.points, self.renderer.white())

        # target circle 2
        Mt2 = Circle()
        Mt2.location = Vec3.normalized(Vec3.cross(target_direction, Vec3(0, 0, 1))) * -steering_radius + target_location
        Mt2.radius = steering_radius
        Mt2.rotation = 1
        # render
        Mt2.points = self.getPointsInSircle(11, Mt2.radius, Mt2.location)
        #self.renderer.draw_polyline_3d(Mt2.points, self.renderer.white())

        possibleTangents = []



        # left to right
        possibleTangents.append(self.getCrossTangents(Mc1, Mt2, car_location, target_direction, target_location)[0])
        possibleTangents[0].name = "lr"
        # right to left
        possibleTangents.append(self.getCrossTangents(Mc2, Mt1, car_location, target_direction, target_location)[1])
        possibleTangents[1].name = "rl"
        # left to left
        possibleTangents.append(self.getStraightTangents(Mc1, Mt1, car_location, target_direction, target_location)[0])
        possibleTangents[2].name = "ll"
        # right to right
        possibleTangents.append(self.getStraightTangents(Mc2, Mt2, car_location, target_direction, target_location)[1])
        possibleTangents[3].name = "rr"

        

        best_path = ArcLineArcPath()

        for tangent in possibleTangents:
            #self.renderer.draw_line_3d(tangent.start, tangent.end, self.renderer.white())
            #self.renderer.draw_line_3d(tangent.start, tangent.circle1_center, self.renderer.white())
            #self.renderer.draw_line_3d(car_location, tangent.circle1_center, self.renderer.white())
            #if(tangent.possible):
                #self.renderer.draw_line_3d(tangent.start, tangent.end, self.renderer.white())
            c1_arc_angle = Vec3.angle(Vec3.flat(tangent.start - tangent.circle1_center),Vec3.flat(car_location - tangent.circle1_center)) * 180/math.pi
            c1_radius = Vec3.length(Vec3.flat(tangent.start - tangent.circle1_center))
            c2_arc_angle = Vec3.angle(Vec3.flat(tangent.end -  tangent.circle2_center), Vec3.flat(target_location - tangent.circle2_center))* 180/math.pi
            c2_radius = Vec3.length(Vec3.flat(tangent.end - tangent.circle2_center))

            if (tangent.start.x - tangent.circle1_center.x)*(car_location.y - tangent.circle1_center.y) - (tangent.start.y - tangent.circle1_center.y)*(car_location.x - tangent.circle1_center.x)>0:
                if(tangent.name == "rl" or tangent.name == "rr"):   
                    c1_arc_angle = 360 - c1_arc_angle
            else:
                if(tangent.name == "lr" or tangent.name == "ll"):   
                    c1_arc_angle = 360 - c1_arc_angle


            if (tangent.end.x - tangent.circle2_center.x)*(target_location.y -tangent.circle2_center.y) - (tangent.end.y - tangent.circle2_center.y)*(target_location.x - tangent.circle2_center.x)>0:
               if(tangent.name == "rl" or tangent.name == "rr"):   
                    c2_arc_angle = 360 - c2_arc_angle
            else:
                if(tangent.name == "lr" or tangent.name == "ll"):   
                    c2_arc_angle = 360 - c2_arc_angle

            

            c1_arc_length = c1_arc_angle/360* 2*math.pi * c1_radius
            c2_arc_length = c2_arc_angle/360* 2*math.pi * c2_radius
           
            

            tangent_length = Vec3.length(tangent.end - tangent.start)

            arc_line_arc_length = c1_arc_length + c2_arc_length + tangent_length


            if best_path.length > arc_line_arc_length and tangent.possible:
                best_path.length = arc_line_arc_length

                best_path.start = car_location
                best_path.tangent_start = tangent.start
                best_path.tangent_end = tangent.end
                best_path.tangent_length = Vec3.length(tangent.end - tangent.start)
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
        

        return(best_path)

            

    def getCrossTangents(self, C1, C2, car_location, target_direction, target_location):

        # middle circle
        C3 = Circle()
        C3.location = C1.location +  (C2.location - C1.location)*0.5
        C3.radius = Vec3.length((C2.location - C1.location)*0.5)

        # bigger car circle
        C4 = Circle()
        C4.location = C1.location
        C4.radius = C1.radius + C2.radius

        # bigger target circle
        C5 = Circle()
        C5.location = C2.location
        C5.radius = C1.radius + C2.radius

        C4intersections = self.getIntersections(C3.location.x, C3.location.y, C3.radius, C4.location.x, C4.location.y, C4.radius)
        C5intersections = self.getIntersections(C3.location.x, C3.location.y, C3.radius, C5.location.x, C5.location.y, C5.radius)

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
        C3.location = C1.location +  (C2.location - C1.location)*0.5
        C3.radius = Vec3.length((C2.location - C1.location)*0.5)

        # bigger car circle
        C4 = Circle()
        C4.location = C1.location
        C4.radius = C1.radius - C2.radius+ 1

        # bigger target circle
        C5 = Circle()
        C5.location = C2.location
        C5.radius = C2.radius - C1.radius + 1

        C4intersections = self.getIntersections(C3.location.x, C3.location.y, C3.radius, C4.location.x, C4.location.y, C4.radius)
        C5intersections = self.getIntersections(C3.location.x, C3.location.y, C3.radius, C5.location.x, C5.location.y, C5.radius)

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


        d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
        

        a=(r0**2-r1**2+d**2)/(2*d)
        if(r0**2-a**2<0): 
            return (0, 0, 0, 0, False)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return (x3, y3, x4, y4, True)

    def getArcLineArcControllerState(self, path, car):
        car_location = Vec3(car.physics.location)
        car_rotation = car.physics.rotation
        car_velocity = Vec3(car.physics.velocity)

        radius = 1 / 0.001375
        needed_radius = self.getSteeringRadius(car_velocity, car_rotation)

        turn_force = 1 / radius * needed_radius

        print(turn_force)

        distance_to_next_point = None
        target = None

        if(path.phase == 0):
            distance_to_next_point = Vec3.length(car_location - path.tangent_start)
            target = path.tangent_start






            if(distance_to_next_point < 100):
                path.phase += 1
        
        if(path.phase == 1):
            distance_to_next_point = Vec3.length(car_location - path.tangent_end)
            target = path.tangent_end
            if(distance_to_next_point < 100):
                path.phase += 1

        if(path.phase == 2):
            distance_to_next_point = Vec3.length(car_location - path.end)
            target = path.end
            if(distance_to_next_point < 100):
                path.phase += 1


        if target != None:
            steer = steer_toward_target(car, target)
            print(steer)
        else:
            steer = 0
        return(steer * turn_force)








    #==============================|==============================#
    #==========================Movement===========================#
    #==============================|==============================#

    def begin_front_flip(self, packet):
        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05,
                        controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05,
                        controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(
                jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
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
                    self.renderer.draw_line_3d(loc1, loc2, self.renderer.yellow())
    


    def predictBallLocation(self, time):
        ball_prediction = self.get_ball_prediction_struct()
        if(time > 59): time = 59
        elif(time < 1): time = 1
        ball_prediction_time = ball_prediction.slices[time].physics.location

        return Vec3(ball_prediction_time.x, ball_prediction_time.y, ball_prediction_time.z) 


    def getSteeringRadius(self, car_velocity, car_rotation):
        velocity = Vec3.dot(car_velocity, Vec3.normalized(Orientation(car_rotation).forward))
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

        if (velocity<=0): return 1/0.0069
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
        curvature = val3 - (val3 - val4)* percentage

        
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

        percentage = 1 / (vals[1][0] - vals[0][0]) * (car_velocity - vals[0][0])

        acceleration = vals[0][1] - (vals[0][1] - vals[1][1]) * percentage

        return acceleration







    #==============================|==============================#
    #==========================Rendering==========================#
    #==============================|==============================#
    def renderText(self, text):
        text = str(text)
        self.renderer.draw_string_2d(4, 4, 2, 2,text, self.renderer.white())



    def getPointsInSircle(self, every, radius, center):
        circle_positions = []
        for i in range(every + 1):
            
            angle = 2 * math.pi / every * (i+1)
            location = Vec3(radius * math.sin(angle), radius * -math.cos(angle), 0) + center
            location.z = 4
            circle_positions.append(location)
        return(circle_positions)



    def renderArcLineArcPath(self, path, car_rotation, target_rotation):
        """
        every = 10
        circle_positions1 = []
        circle_positions2 = []
        spots = [path.name[0], path.name[1]]
        # circle 1
        start_rotation = Vec3.angle((path.tangent_start - path.c1_center), Vec3(0, -1, 0))
        end_rotation = Vec3.angle((path.start - path.c1_center), Vec3(0, -1, 0))
        rotation_snippet = path.c1_angle / every * math.pi / 180
        if(path.start.y > 0): start_rotation = -start_rotation
        for i in range(every + 1):
            angle = 0
            if(spots[0] == "r"):
                angle = start_rotation - rotation_snippet * i 
            if(spots[0] == "l"):
                angle = -start_rotation + rotation_snippet * i 
            location = Vec3(path.c1_radius * math.sin(angle), path.c1_radius * -math.cos(angle), 0) + path.c1_center
            location.z = 4
            circle_positions1.append(location)
        self.renderer.draw_polyline_3d(circle_positions1, self.renderer.red())
        # circle 2
        start_rotation = Vec3.angle((path.tangent_end - path.c2_center), Vec3(0, -1, 0))
        end_rotation = Vec3.angle((path.end - path.c2_center), Vec3(0, -1, 0))
        rotation_snippet = path.c2_angle / every * math.pi / 180
        if(path.end.y > 0): start_rotation = -start_rotation
        for i in range(every + 1):
            angle = 0
            if(spots[1] == "l"):
                angle = start_rotation - rotation_snippet * i 
            if(spots[1] == "r"):
                angle = -start_rotation + rotation_snippet * i 
            location = Vec3(path.c2_radius * math.sin(angle), path.c2_radius * -math.cos(angle), 0) + path.c2_center
            location.z = 4
            circle_positions2.append(location)
        self.renderer.draw_polyline_3d(circle_positions2, self.renderer.red())
        """

        p = self.getPointsInSircle(20, path.c1_radius, path.c1_center)
        self.renderer.draw_polyline_3d( p , self.renderer.red())
        p = self.getPointsInSircle(20, path.c2_radius, path.c2_center)
        self.renderer.draw_polyline_3d( p , self.renderer.red())
        self.renderer.draw_line_3d(path.tangent_start, path.tangent_end, self.renderer.red())
        """
        # circle 1
        
        start_rotation = Vec3.angle((path.tangent_start - path.c1_center), Vec3(0, -1, 0))
        end_rotation = Vec3.angle((path.start - path.c1_center), Vec3(0, -1, 0))
        rotation_snippet = ((start_rotation**2)**0.5 + (end_rotation**2)**0.5) / every
        indicator = start_rotation - end_rotation 
        for i in range(every + 1):
            angle = 0
            if(path.name == "rl" or path.name == "rr"):
                angle = -start_rotation - rotation_snippet * i 
            if(path.name == "ll" or path.name == "lr" ):
                angle = start_rotation + rotation_snippet * i 
            location = Vec3(path.c1_radius * math.sin(angle), path.c1_radius * -math.cos(angle), 0) + path.c1_center
            location.z = 4
            circle_positions1.append(location)
        self.renderer.draw_polyline_3d(circle_positions1, self.renderer.red())
        
        # circle 2
        start_rotation = Vec3.angle((path.tangent_end - path.c2_center), Vec3(0, -1, 0))
        rotation_snippet = path.c2_angle * math.pi*2/360 / every * -1
    
        for i in range(every + 1):
            angle = 0
            if(path.name == "rl" or path.name == "rr"):
                angle = -start_rotation - rotation_snippet * i 
            if(path.name == "ll" or path.name == "lr" ):
                angle = start_rotation + rotation_snippet * i 
            location = Vec3(path.c2_radius * math.sin(angle), path.c2_radius * -math.cos(angle), 0) + path.c2_center
            location.z = 4
            circle_positions2.append(location)
        self.renderer.draw_polyline_3d(circle_positions2, self.renderer.red())
        """

    
    

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
        self.phase = 0

class Maneuver:
    def __init__(self):
        # target
        self.target = None

        # target Location
        self.target_location = None


        # generated path
        self.path = None
        self.forseen_ball_locations = None


