from util.drive import steer_toward_target

class Controlls():
    def getThrottle(self, controls):

        path_length = self.path_length


        time_left = (self.maneuver_time - self.since_maneuver_start) / 10
        needed_speed = path_length / (time_left + 0.1)
        speed = self.car_forward_velocity
        diff = needed_speed - speed

        throttle = diff/1000 * 3

        if(throttle > 1):
            throttle = 1

        if(throttle < -1):
            throttle = -1
        controls.throttle = throttle

        return controls

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