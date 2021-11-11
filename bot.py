


from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence
from util.vec import Vec3
from util.orientation import Orientation

from nn import learningAgent
from objective_module import Objective 
from target_module import Target
from path_module import Path
from controlls_module import Controlls
from renderer_module import Renderer
from helpers_module import Helpers



class MyBot(BaseAgent, Objective, Target, Path, Controlls, Renderer, Helpers):
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

        self.min_rad = 400

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








