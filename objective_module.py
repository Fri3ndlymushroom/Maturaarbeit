from util.vec import Vec3


class Objective():
    def setObjective(self):

        if(self.unforseenAction()):
            # new_target_index = learningAgent.getAction(packet)
            new_target_index = 0
            self.target_index = new_target_index
            self.maneuver_start = self.packet.game_info.seconds_elapsed
            self.createNewManeuver()
            return True
        else: return False

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