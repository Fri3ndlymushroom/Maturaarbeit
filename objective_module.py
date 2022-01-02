from util.vec import Vec3
from nn import learningAgent


class Objective():
    def setObjective(self):

        if(self.unforseenAction()):
            new_target_index = self.target_index
            if(self.since_maneuver_start > 1):
                new_target_index = 0
                if(self.index == 0):
                    new_target_index = learningAgent.getAction(self.packet)
            self.target_index = new_target_index
            self.maneuver_start = self.packet.game_info.seconds_elapsed
            self.since_maneuver_start = 0

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

            vend = self.car_forward_velocity
            l = possible_path_length
            t = round(i)

            while t >= 0:
                l -= vend / 10
                vend += self.getAcceleration(vend) / 100
                t -= 1

            possible_path_length_extended = self.computePossibleArcLineArcDrivePaths(
                    target_location, target_direction, self.getSteeringRadius(vend)).length


            v1 = self.car_forward_velocity
            lext = possible_path_length_extended
            tcons = 0

            while(lext > 0):
                lext -= v1 / 10
                v1 += self.getAcceleration(v1) / 100
                tcons += 1


            reachable = (tcons <= i)

            if(i == 59): reachable = True

            if(i < best_path[0] and reachable):
                best_path = [i, possible_path_length_extended]

        if(best_path[0] > 60):
            best_path[0] = 60

        self.maneuver_time = best_path[0]

    def unforseenAction(self):

        

        # not inited
        if self.last_prediction == None or self.since_maneuver_start > self.maneuver_time:
            self.last_prediction = self.get_ball_prediction_struct().slices
            self.last_time = self.packet.game_info.seconds_elapsed
            return True


        # no time left
        time = self.packet.game_info.seconds_elapsed
        delta_time = round(359/60*(time - self.last_time) * 10)

        if(delta_time > 10):

            prediction = self.get_ball_prediction_struct().slices
            last_prediction = self.last_prediction

            new_prediction = prediction[100 - delta_time].physics.location
            old_prediction = last_prediction[100].physics.location

            deviation = Vec3.length(Vec3(new_prediction.x, new_prediction.y, new_prediction.z) - Vec3(
                old_prediction.x, old_prediction.y, old_prediction.z))

            self.last_prediction = self.get_ball_prediction_struct().slices
            self.last_time = self.packet.game_info.seconds_elapsed
            print(deviation)
            if(deviation > 100):
                return True
        return False
