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


def debugLog(content):
    text.insert(END, str(content)+"\n")
    text.see(END)
    text.update()

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
        model.add(tf.keras.layers.Dense(3, input_shape=(4,), activation="relu"))
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
        qs= self.model.predict(np.array(state).reshape(-1, *np.array(state).shape)/6000)[0]
        return qs
    
    # jeder step soll das netzwerk trainiert werdenfg
    def train(self, terminal_state, step):

        
        # soll abbrechen wenn das replay memory zu klein ist
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # batch mit der grösse MINIBATCH_SIZE vom replay memory
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        

        
        # get q valuess
        current_states = np.array([transition[0] for transition in minibatch])/6000
        current_qs_list = self.model.predict(current_states)

        # zukünftige q values
        new_current_states = np.array([transition[3] for transition in minibatch])/6000
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

        self.model.fit(np.array(X)/6000, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[] if terminal_state else None)
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

        #penaltys
        self.MOVE_PENALTY = 1
        self.DISTANCE_PENALTY_MULTIPLICATOR = 5


    
    def getAction(self, field_info, car_location, game_packet, car_rotation):

        self.field_info = field_info
        self.car_location = car_location
        self.game_packet = game_packet
        self.car_rotation = car_rotation



        # wenn self.step == STEPS_PER_EPISODE ist sollen variabeln zurückgesetz werden
        self.manageEpisodes()


        

        # wichtige parameter berechnen
        target_vector = self.car_location - self.target_location
        self.state_now = [target_vector.x, target_vector.y, target_vector.z, car_rotation.yaw]

        if self.step == 1:
            self.normalizer = car_location.dist(self.target_location)
        
        # den letzten schritt beurteilen
        if self.should_train:
            # erstmal soll der Bot anhand der distanz zum ziel beurteilt werden
            distance = car_location.dist(self.target_location)


            self.step_reward -= self.MOVE_PENALTY
            reward_normalized = distance / self.normalizer

            self.step_reward -= reward_normalized * self.DISTANCE_PENALTY_MULTIPLICATOR
            self.episode_reward += self.step_reward

            if distance < 100:
                debugLog("done")
                self.done = True
                self.done_accuracy.append(1)
            elif self.step == self.STEPS_PER_EPISODE-1:
                self.done = True
                self.done_accuracy.append(0)
                

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

        if (self.total_step % 200 == 0): 
            # step
            debugLog("step")
            debugLog(self.total_step)

            # accuracy
            sum = 0
            i = 0
            for done in self.done_accuracy[-1000:]:
                sum += done
                i += 1
            accuracy = sum / i
            debugLog("accuracy")
            debugLog(accuracy)
            if agent.target_update_counter > agent.UPDATE_TARGET_EVERY:
                debugLog("update Target")


            debugLog("-------------")

        return self.action



    def manageEpisodes(self):
        if self.step == self.STEPS_PER_EPISODE or self.done:
            self.step = 1

        if self.step == 1:
            self.episode_reward = 0
            self.done = False
            self.target_location = self.getRandomBoost()

            # verhindert, dass beim ersten schritt trainiert wird
            self.should_train = False
        else:
            self.should_train = True
        
        self.step_reward = 0



    
        
    
    def getNearestFullBoost(self):
        nearest_boost_location = None


        
        i = 0
        for boost in self.field_info.boost_pads:
            if boost.is_full_boost and self.game_packet.game_boosts[i].is_active:
                if not nearest_boost_location:
                    nearest_boost_location = boost.location
                else:
                    if self.car_location.dist(Vec3(boost.location)) < self.car_location.dist(nearest_boost_location):
                        nearest_boost_location = boost.location
            i += 1

        return nearest_boost_location
    
    def getRandomBoost(self):
        random_boost = None
        index = random.randint(0, 5)

        
        i = 0
        for boost in self.field_info.boost_pads:
            if boost.is_full_boost and self.game_packet.game_boosts[i].is_active:
                if index == i:
                    random_boost = boost.location
                i += 1

        return random_boost


learningAgent = QLearningAgent()







class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

        self.step = 1
        self.epsiolon = 1

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.renderer.begin_rendering()
        
        #============[get info]============    

        # field info
        field_info = self.get_field_info()
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_rotation = my_car.physics.rotation
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        self.predictBallPath()
        

        

       

        #============[decision]============
        target = "shoot_towards_goal"
        
        #============[execution & controls]============
        controls = SimpleControllerState()
        if target == "boost":
            target_location = Vec3(self.getNearestFullBoost(packet))

            controls = self.steerTowardsTarget(my_car, car_velocity, target_location)
        elif target == "steerTowardsTarget_train":
            action = 10
            #action = learningAgent.getAction(field_info, car_location, packet, car_rotation)
        elif target == "shoot_towards_goal":
            target_location_info = self.shootBallTowardsTarget(packet, Vec3(800, 5213, 321.3875), Vec3(-800, 5213, 321.3875))
            self.computePossibleArcLineArcDrivePaths(packet, target_location_info[0], target_location_info[1])

            #controls.steer = steer_toward_target(my_car, target)
            controls.steer =1
            controls.throttle = 0.1

        self.renderer.end_rendering()
        return controls

    def renderText(self, text):
        self.renderer.begin_rendering()
        self.renderer.draw_string_2d(4, 4, 2, 2,text, self.renderer.white())
        self.renderer.end_rendering()


    
    def shootBallTowardsTarget(self, packet, left_most_target, right_most_target):
        ball_location = Vec3(packet.game_ball.physics.location)
        car_location = Vec3(packet.game_cars[self.index].physics.location)

        

        car_to_ball = ball_location - car_location
        car_to_ball_direction = Vec3.normalized(car_to_ball)

        ball_to_left_target_direction = Vec3.normalized(left_most_target - ball_location)
        ball_to_right_target_direction = Vec3.normalized(right_most_target - ball_location)
        direction_of_approach = Vec3.clamp2D(direction=car_to_ball_direction, start=ball_to_left_target_direction, end=ball_to_right_target_direction)

        offset_ball_location = ball_location - direction_of_approach * 92.75

        """
        side_of_approach_direction = Vec3.dot(Vec3.cross(direction_of_approach, Vec3(0, 0, 1)), ball_location - car_location)
        if side_of_approach_direction > 0: side_of_approach_direction = 1
        elif side_of_approach_direction < 0: side_of_approach_direction = -1
        else: side_of_approach_direction = 0

        if Vec3.cross(car_to_ball, Vec3(0, 0, side_of_approach_direction)) != round(0):
            car_to_ball_perpendicular = Vec3.normalized(Vec3.cross(car_to_ball, Vec3(0, 0, side_of_approach_direction)))
        else:
            car_to_ball_perpendicular = car_to_ball

        adjustment = Vec3.angle(Vec3.flat(car_to_ball), Vec3.flat(direction_of_approach)) * 2560
        if (adjustment < 0):adjustment = -adjustment
        final_target = offset_ball_location + (car_to_ball_perpendicular * adjustment)
        """

        self.renderer.draw_line_3d(ball_location, ball_location+ball_to_left_target_direction*1000, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, ball_location+ball_to_right_target_direction*1000, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, ball_location+ball_to_left_target_direction*-1000, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, ball_location+ball_to_right_target_direction*-1000, self.renderer.white())

        

        return [offset_ball_location, direction_of_approach]

        
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
    
    def computePossibleArcLineArcDrivePaths(self, packet, target_location, target_direction):
        #my_car = packet.game_cars[self.index]
        my_car = packet.game_cars[1]
        car_location = Vec3(my_car.physics.location)
        car_rotation = my_car.physics.rotation
        car_velocity = Vec3(my_car.physics.velocity)
        steering_radius = self.getSteeringRadius(car_velocity, car_rotation)

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
        self.renderer.draw_polyline_3d(Mc1.points, self.renderer.white())

        # car circle 2
        Mc2 = Circle()
        Mc2.location = Vec3.normalized(Vec3.cross(car_direction, Vec3(0, 0, 1))) * -steering_radius + car_location
        Mc2.radius = steering_radius
        Mc2.rotation = 1
        # render
        Mc2.points = self.getPointsInSircle(11, Mc2.radius, Mc2.location)
        self.renderer.draw_polyline_3d(Mc2.points, self.renderer.white())


        # target circles
        self.renderer.draw_line_3d(target_location,target_location+ target_direction*100, self.renderer.red())


        # target circle 1
        Mt1 = Circle()
        Mt1.location = Vec3.normalized(Vec3.cross(target_direction, Vec3(0, 0, 1))) * steering_radius + target_location
        Mt1.radius = steering_radius   
        Mt1.rotation = -1
        # render
        Mt1.points = self.getPointsInSircle(11, Mt1.radius, Mt1.location)
        self.renderer.draw_polyline_3d(Mt1.points, self.renderer.white())

        # target circle 2
        Mt2 = Circle()
        Mt2.location = Vec3.normalized(Vec3.cross(target_direction, Vec3(0, 0, 1))) * -steering_radius + target_location
        Mt2.radius = steering_radius
        Mt2.rotation = 1
        # render
        Mt2.points = self.getPointsInSircle(11, Mt2.radius, Mt2.location)
        self.renderer.draw_polyline_3d(Mt2.points, self.renderer.white())

        possibleTangents = []




        possibleTangents.append(self.getCrossTangents(Mc1, Mt2, car_location, target_direction, target_location)[0])
        possibleTangents.append(self.getCrossTangents(Mc2, Mt1, car_location, target_direction, target_location)[1])

        possibleTangents.append(self.getStraightTangents(Mc1, Mt1, car_location, target_direction, target_location)[0])
        possibleTangents.append(self.getStraightTangents(Mc2, Mt2, car_location, target_direction, target_location)[1])

        
        best_path = ArcLineArcPath()

        for tangent in possibleTangents:
            #self.renderer.draw_line_3d(tangent.start, tangent.end, self.renderer.white())


            #self.renderer.draw_line_3d(tangent.start, tangent.circle_center1, self.renderer.white())
            #self.renderer.draw_line_3d(car_location, tangent.circle_center1, self.renderer.white())


            c1_arc_angle = Vec3.angle(tangent.start - tangent.circle_center1,car_location - tangent.circle_center1) * 180/math.pi
            c1_radius = Vec3.length(tangent.start - tangent.circle_center1)
            c2_arc_angle = Vec3.angle(tangent.end -  tangent.circle_center2, target_location - tangent.circle_center2)* 180/math.pi
            c2_radius = Vec3.length(tangent.end - tangent.circle_center2)

            c1_arc_length = c1_arc_angle/360* 2*math.pi * c1_radius
            c2_arc_length = c2_arc_angle/360* 2*math.pi * c2_radius






            if(tangent.start.y - tangent.circle_center1.y*(tangent.start.x - tangent.circle_center1.x)+tangent.start.y - tangent.circle_center1.y*(tangent.start.x - tangent.circle_center1.x))


            tangent_length = Vec3.length(tangent.end - tangent.start)

            arc_line_arc_length = c1_arc_length + c2_arc_length + tangent_length

            if best_path.length < arc_line_arc_length:
                best_path.length = arc_line_arc_length
                best_path.start = car_location
                best_path.tangent_start = tangent.start
                best_path.tangent_end = tangent.end
                best_path.end = target_location
        
        self.renderer.draw_polyline_3d([best_path.start, best_path.tangent_start, best_path.tangent_end, best_path.end], self.renderer.white())

        







    
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
        tangent1.circle_center1 = C1.location
        tangent1.circle_center2 = C2.location


    
        tangent2 = Tangent()
        tangent2.start = C1t2
        tangent2.end = C2t2
        tangent2.circle_center1 = C1.location
        tangent2.circle_center2 = C2.location

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
        tangent1.circle_center1 = C1.location
        tangent1.circle_center2 = C2.location


    
        tangent2 = Tangent()
        tangent2.start = C1t2
        tangent2.end = C2t1
        tangent2.circle_center1 = C1.location
        tangent2.circle_center2 = C2.location

        return[tangent1, tangent2]




        

    def getIntersections(self, x0, y0, r0, x1, y1, r1):


        d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
        

        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
            
        return (x3, y3, x4, y4)
        
    
    def getSteeringRadius(self, car_velocity, car_rotation):
        velocity = Vec3.dot(car_velocity, Vec3.normalized(Orientation(car_rotation).forward))
        curvature = 0.0069
        if velocity > 2300: curvature = 0.0008
        elif velocity > 1750: curvature = 0.0011
        elif velocity > 1500: curvature = 0.001375
        elif velocity > 1000: curvature = 0.00235
        elif velocity > 500: curvature = 0.00598

        radius = 1/curvature

        return(radius)

    def getPointsInSircle(self, every, radius, center):
        circle_positions = []
        for i in range(every + 1):

            angle = 2 * math.pi / every * (i+1)
            location = Vec3(radius * math.sin(angle), radius * -math.cos(angle), 0) + center
            location.z = 4
            circle_positions.append(location)
        return(circle_positions)



class Circle:
    def __init__(self):
        self.location = Vec3(0, 0, 0)
        self.radius = 0
        self.points = []
        self.rotation = 0

        
class Tangent:
    def __init__(self):
        self.circle_center1 = Vec3(0, 0, 0)
        self.start = Vec3(0, 0, 0)

        self.circle_center2 = Vec3(0, 0, 0)
        self.end = Vec3(0, 0, 0)


class ArcLineArcPath:
    def __init__(self):
        self.length = 0



        self.start = Vec3(0, 0, 0)
        self.tangent_start = Vec3(0, 0, 0)
        self.tangent_end = Vec3(0, 0, 0)
        self.end = Vec3(0, 0, 0)

