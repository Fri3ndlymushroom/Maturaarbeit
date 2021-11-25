from util.vec import Vec3
from nn import learningAgent


class Objective():
    def setObjective(self):

        if(self.unforseenAction()):
            new_target_index = self.target_index
            if(self.since_maneuver_start > 1):
                new_target_index = 0
                new_target_index = learningAgent.getAction(self.packet)
            self.target_index = new_target_index
            self.maneuver_start = self.packet.game_info.seconds_elapsed

            self.createNewManeuver()
            return True
        else: return False

    def createNewManeuver(self):

        # generate a point the bot can surely reach
        best_path = [59, 1000000]
        for i in range(60):
            self.maneuver_time = i

            [target_location, target_direction] = self.setTarget()

            possible_path_length = self.setPath(
                    target_location, target_direction).length



            v = self.car_forward_velocity
            l = possible_path_length
            t = round(i)


            while t > 0:
                l -= v / 10
                v += self.getAcceleration(v) /10
                t -= 1

            reachable = (l + 500  < 0)

            if(i == 59): reachable= True

            if(possible_path_length < best_path[1] and reachable):
                best_path = [i, possible_path_length]

        self.maneuver_time = best_path[0] + 10

        v0 = self.car_forward_velocity
        t = 0
        d = best_path[1]


        max_speed = 1500

        while d > 0:
            d -= v0 / 10
            v0 += self.getAcceleration(v0) / 100

            if(v0 > max_speed):
                v0 = max_speed
            t += 1

        t = t * 0.7

        if(t > 60):
            t = 60

        self.maneuver_time = t

    
    def unforseenAction(self):

        # not inited
        if self.last_prediction == None or self.since_maneuver_start > self.maneuver_time:
            self.last_prediction = self.get_ball_prediction_struct().slices
            self.last_time = self.packet.game_info.seconds_elapsed
            return True


        # no time left
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


        while t > 0:
            l -= v / 10
            v += self.getAcceleration(v) /10
            t -= 1

        #if(l - 500 > 0): return True


        return False
