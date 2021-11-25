from util.vec import Vec3
import random

class Target():
    def setTarget(self):

        target = self.target_index
        target_location_info = None

        if target == 0:
            self.renderText("attack")
            target_location_info = self.shootBallTowardsTarget(
                Vec3(100, 5213, 321.3875), Vec3(-100, 5213, 321.3875))
        elif target == 1:
            self.renderText("defend")
            target_location_info = self.shootBallTowardsTarget(
                Vec3(10000, self.ball_location.y - 2000, self.ball_location.z),
                Vec3(-10000, self.ball_location.y - 2000, self.ball_location.z),
            )
        elif target == 2:
            self.renderText("base line")
            target_location_info = [Vec3(0, -4200, 0), Vec3(0,1,0)]












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
        return target_location_info

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
        offset_ball_location = ball_location - direction_of_approach * 90


        return [offset_ball_location, direction_of_approach]
