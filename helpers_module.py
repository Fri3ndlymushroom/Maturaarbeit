from util.vec import Vec3



class Helpers():
    def trackGame(self):
        # Time
        self.since_maneuver_start = -1 * \
            (self.maneuver_start*10 - self.packet.game_info.seconds_elapsed*10)

    def isNearWall(self):
        x = self.ball_location.x
        y = self.ball_location.y

        goals = [Vec3(0, 5213, 0), Vec3(0, -5213, 0)]
        near = x > 3800 or x < -3800 or y > 4800 or y < -4800
        near_goal = Vec3.length(self.ball_location - goals[0]) < 1500 or Vec3.length(self.ball_location - goals[1]) < 1500

        if not near_goal:
            return near
        else: return False


    def predictBallLocation(self, time):
        if time > 60:
            time = 60

        time = round(359/60*time)

        ball_prediction = self.get_ball_prediction_struct()

        ball_prediction_time = ball_prediction.slices[time].physics.location

        return Vec3(ball_prediction_time.x, ball_prediction_time.y, ball_prediction_time.z)

    def getSteeringRadius(self, v = None):

        
        velocity = self.car_forward_velocity
        if(v): velocity = v
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