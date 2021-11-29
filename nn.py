from collections import deque
import random
import numpy as np
import math
import tensorflow as tf
from util.vec import Vec3
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


tf.get_logger().setLevel(3)

ACTION_SPACE_SIZE = 5
OBSERVATION_SPACE_SIZE = 6


class ModelAgent():
    def __init__(self):
        self.UPDATE_TARGET_EVERY = 5
        self.MODEL_NAME = "model"
        self.MINIBATCH_SIZE = 64
        self.DISCOUNT = 0.99


        try:
            self.model = tf.keras.models.load_model("src/training/target")
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())
        except:
            # --Models-- # Two models to get better consistency and not fit main model to values with to high epsilon
            # Main model # gets trained every step
            self.model = self.create_model()
            # Target model # gets predicted against every step
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

        # --Replay memory-- # The network is trained with a random batch of the replay memory
        # # Deque is a list that has a max len and pushes out the oldest value
        self.REPLAY_MEMORY_SIZE = 5000
        self.MIN_REPLAY_MEMORY_SIZE = 100
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # --target update counter-- # keeps track of when to update target model
        self.target_update_counter = 0

    def create_model(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(OBSERVATION_SPACE_SIZE, input_shape=(
            OBSERVATION_SPACE_SIZE,), activation="relu"))

        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.Dense(8, activation="relu"))

        model.add(tf.keras.layers.Dense(
            ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(
            lr=0.01), metrics=['accuracy']) #0.001
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape))[0]

    def train(self, terminal_state, step):

        # if replay memory is to small do not train
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # get batch
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # get q values
        current_states = np.array([transition[0]
                                  for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # future q values
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
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

        self.model.fit(np.array(X), np.array(y), batch_size=self.MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[] if terminal_state else None)
        if terminal_state:
            self.target_update_counter += 1
            print("------terminal state------")

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            print("-------update target-------")
            self.model.save("src/training/target")
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


agent = ModelAgent()


class QLearningAgent:
    def __init__(self):
        self.step = 1
        self.episode = 1
        self.episode_reward = 0
        self.episode_rewards = []
        self.done = False
        self.total_step = 1
        self.STEPS_PER_EPISODE = 50

        # epsilon
        self.epsilon = 1
        self.EPSILON_DECAY = 0.99975
        self.MIN_EPSILON = 0.001

        # penaltys
        self.MOVE_PENALTY = 1

        self.reward_info = rewardInfo()

    def getAction(self, packet):

        self_car = packet.game_cars[0]
        enemy_car = packet.game_cars[1]
        self_car_location = Vec3(self_car.physics.location)
        enemy_car_location = Vec3(enemy_car.physics.location)
        ball_location = Vec3(packet.game_ball.physics.location)
        game_time = packet.game_info.seconds_elapsed


        if self.step == self.STEPS_PER_EPISODE or self.done:

            #print("episode: ", self.episode, " reward: ", self.episode_reward)
            #addDatapoint([self.epsilon, self.episode_reward])
            if not (self.episode == 0):
                self.episode_rewards.append(self.episode_reward)

            print("===================================Episode {self.episode} evaluation===================================")
            print(f"Last: {self.episode_reward}, Average: {sum(self.episode_rewards) / len(self.episode_rewards)}, Max: {np.argmax(self.episode_rewards)}, Min: {np.argmin(self.episode_rewards)}")
            print("=============================================================================================")

            self.episode += 1
            self.step = 1
            self.episode_reward = 0
            self.done = False

        self.state_now = [
            self_car_location.x / 4096, self_car_location.y / 5120,
            enemy_car_location.x / 4096, enemy_car_location.y / 5120,
            ball_location.x / 4096, ball_location.y / 5120
        ]



        # den letzten schritt beurteilen
        if not self.step == 1:
            self.step_reward = self.getReward(self_car, packet)

            self.episode_reward += self.step_reward

            agent.update_replay_memory(
                    (self.old_state, self.action, self.step_reward, self.state_now, self.done))


            if self.step + 1 == self.STEPS_PER_EPISODE: self.done = True
            
            agent.train(self.done, self.step)

        # neuen schritt machen
        if np.random.random() > self.epsilon:
            # Get action from Q table
            self.action = np.argmax(agent.get_qs(self.state_now))
        else:
            # Get random action
            self.action = np.random.randint(0, ACTION_SPACE_SIZE)

        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon = max(self.MIN_EPSILON, self.epsilon)

        self.old_state = self.state_now

        self.step += 1
        self.total_step += 1
        self.last_call = game_time



        print(f"Step: {self.step}/{self.STEPS_PER_EPISODE}, Total: {self.total_step}, ")

        return self.action


    def getReward(self, self_car, packet):

        # 0.1*touch + shot + save + 10*goal - 10*concede - shot_on_own_goal



        score_info = self_car.score_info
        # shots
        self.reward_info.made_shots = score_info.shots - self.reward_info.shots
        self.reward_info.shots = score_info.shots
        # saves
        self.reward_info.made_saves = score_info.saves - self.reward_info.saves
        self.reward_info.saves = score_info.saves
        # goals
        self.reward_info.made_goals = score_info.goals - self.reward_info.goals
        self.reward_info.goals = score_info.goals
        # got goals
        self.reward_info.made_got_goals = packet.teams[1].score - \
            self.reward_info.got_goals
        self.reward_info.got_goals = packet.teams[1].score

        reward = self.reward_info.made_shots + 10 * self.reward_info.made_goals + \
                self.reward_info.made_saves - 10 * self.reward_info.made_got_goals  - 100
        
        return reward



class rewardInfo:
    def __init__(self):
        self.touches = 0
        self.shots = 0
        self.saves = 0
        self.goals = 0
        self.got_goals = 0
        self.own_goals = 0

        self.made_touches = 0
        self.made_shots = 0
        self.made_saves = 0
        self.made_goals = 0
        self.made_got_goals = 0
        self.made_own_goals = 0


learningAgent = QLearningAgent()


y_norm = []
y_plot = []
plt.ion()

def addDatapoint(data):
    global y_plot
    y_norm.append(data)
    y_plot = np.array(y_norm).reshape(len(y_norm[0]), len(y_norm))


    if(len(y_plot[0]) % 10 == 0):
        getNewChart(y_plot)


index = 0
def getNewChart(data):
    plt.cla()
    plt.plot(y_norm)
    plt.draw()
    plt.pause(0.01)




