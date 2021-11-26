import math
from collections import deque
from util.vec import Vec3
from util.orientation import Orientation


class Path():
    def setPath(self, target_location, target_direction):
        
        self.renderer.draw_line_3d(Vec3(target_location.x, target_location.y, 0), Vec3(
            target_location.x, target_location.y, target_location.z+100), self.renderer.red())
        self.renderer.draw_line_3d(target_location, Vec3(
            target_location.x, target_location.y+100, target_location.z), self.renderer.red())
        self.renderer.draw_line_3d(target_location, Vec3(
            target_location.x+100, target_location.y, target_location.z), self.renderer.red())
        
        size = 7

        self.renderer.draw_rect_3d(Vec3(
            target_location.x, target_location.y, target_location.z), size, size, True, self.renderer.red())

        self.renderer.draw_line_3d(
            target_location, target_location + target_direction * 500, self.renderer.purple())

        path = self.computePossibleArcLineArcDrivePaths(
            target_location, target_direction)
        return path

    def computePossibleArcLineArcDrivePaths(self, target_location, target_direction):
        steering_radius = self.getSteeringRadius()

        if(steering_radius < self.min_rad):
            steering_radius = self.min_rad

        # self.renderer.draw_line_3d(target_location,target_location+ target_direction*600, self.renderer.red())

        # car circles
        car_direction = Orientation(self.car_rotation).forward
        # self.renderer.draw_line_3d(self.car_location,self.car_location+ car_direction*600, self.renderer.red())

        # car circle 1
        Mc1 = Circle()
        Mc1.location = Vec3.normalized(Vec3.cross(
            car_direction, Vec3(0, 0, 1))) * steering_radius + self.car_location
        Mc1.radius = steering_radius
        Mc1.rotation = -1
        # render
        Mc1.points = self.getPointsInSircle(11, Mc1.radius, Mc1.location)
        # self.renderer.draw_polyline_3d(Mc1.points, self.renderer.white())

        # car circle 2
        Mc2 = Circle()
        Mc2.location = Vec3.normalized(Vec3.cross(car_direction, Vec3(
            0, 0, 1))) * -steering_radius + self.car_location
        Mc2.radius = steering_radius
        Mc2.rotation = 1
        # render
        Mc2.points = self.getPointsInSircle(11, Mc2.radius, Mc2.location)
        # self.renderer.draw_polyline_3d(Mc2.points, self.renderer.white())

        # target circles
        # self.renderer.draw_line_3d(target_location,target_location+ target_direction*100, self.renderer.red())

        # target circle 1
        Mt1 = Circle()
        Mt1.location = Vec3.normalized(Vec3.cross(
            target_direction, Vec3(0, 0, 1))) * steering_radius + target_location
        Mt1.radius = steering_radius
        Mt1.rotation = -1
        # render
        Mt1.points = self.getPointsInSircle(11, Mt1.radius, Mt1.location)
        # self.renderer.draw_polyline_3d(Mt1.points, self.renderer.white())

        # target circle 2
        Mt2 = Circle()
        Mt2.location = Vec3.normalized(Vec3.cross(
            target_direction, Vec3(0, 0, 1))) * -steering_radius + target_location
        Mt2.radius = steering_radius
        Mt2.rotation = 1
        # render
        Mt2.points = self.getPointsInSircle(11, Mt2.radius, Mt2.location)
        # self.renderer.draw_polyline_3d(Mt2.points, self.renderer.white())

        possibleTangents = []

        # left to right
        possibleTangents.append(self.getCrossTangents(
            Mc1, Mt2, self.car_location, target_direction, target_location)[0])
        possibleTangents[0].name = "lr"
        # right to left
        possibleTangents.append(self.getCrossTangents(
            Mc2, Mt1, self.car_location, target_direction, target_location)[1])
        possibleTangents[1].name = "rl"
        # left to left
        possibleTangents.append(self.getStraightTangents(
            Mc1, Mt1, self.car_location, target_direction, target_location)[0])
        possibleTangents[2].name = "ll"
        # right to right
        possibleTangents.append(self.getStraightTangents(
            Mc2, Mt2, self.car_location, target_direction, target_location)[1])
        possibleTangents[3].name = "rr"

        best_path = ArcLineArcPath()

        for tangent in possibleTangents:
            # self.renderer.draw_line_3d(tangent.start, tangent.end, self.renderer.white())
            # self.renderer.draw_line_3d(tangent.start, tangent.circle1_center, self.renderer.white())
            # self.renderer.draw_line_3d(car_location, tangent.circle1_center, self.renderer.white())
            # if(tangent.possible):
            # self.renderer.draw_line_3d(tangent.start, tangent.end, self.renderer.white())
            c1_arc_angle = Vec3.angle(Vec3.flat(tangent.start - tangent.circle1_center), Vec3.flat(
                self.car_location - tangent.circle1_center)) * 180/math.pi
            c1_radius = Vec3.length(
                Vec3.flat(tangent.start - tangent.circle1_center))
            c2_arc_angle = Vec3.angle(Vec3.flat(tangent.end - tangent.circle2_center), Vec3.flat(
                target_location - tangent.circle2_center)) * 180/math.pi
            c2_radius = Vec3.length(
                Vec3.flat(tangent.end - tangent.circle2_center))

            if (tangent.start.x - tangent.circle1_center.x)*(self.car_location.y - tangent.circle1_center.y) - (tangent.start.y - tangent.circle1_center.y)*(self.car_location.x - tangent.circle1_center.x) > 0:
                if(tangent.name == "rl" or tangent.name == "rr"):
                    c1_arc_angle = 360 - c1_arc_angle
            else:
                if(tangent.name == "lr" or tangent.name == "ll"):
                    c1_arc_angle = 360 - c1_arc_angle

            if (tangent.end.x - tangent.circle2_center.x)*(target_location.y - tangent.circle2_center.y) - (tangent.end.y - tangent.circle2_center.y)*(target_location.x - tangent.circle2_center.x) > 0:
                if(tangent.name == "rl" or tangent.name == "rr"):
                    c2_arc_angle = 360 - c2_arc_angle
            else:
                if(tangent.name == "lr" or tangent.name == "ll"):
                    c2_arc_angle = 360 - c2_arc_angle

            c1_arc_length = c1_arc_angle/360 * 2*math.pi * c1_radius
            c2_arc_length = c2_arc_angle/360 * 2*math.pi * c2_radius

            tangent_length = Vec3.length(tangent.end - tangent.start)


            if(Vec3.length(tangent.circle1_center - tangent.circle2_center) < 200):
                sidevector = Vec3.normalized(Orientation(self.car_rotation).right)

                a = self.car_location
                b = a + sidevector
                c = best_path.end


                infront = ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)) < 0
                
                
                c1_arc_angle = 0
                c1_arc_length = 0

                angle = Vec3.angle(Vec3.flat(a - tangent.circle2_center), Vec3.flat(target_location - tangent.circle2_center)) * 180/math.pi

                if infront:
                    c2_arc_angle = angle
                else:
                    c2_arc_angle = 360 - angle

                c2_arc_length = c2_arc_angle/360 * 2*math.pi * c2_radius


            arc_line_arc_length = c1_arc_length + tangent_length + c2_arc_length

            if(self.checkIfOutOfMap([tangent.start, tangent.end])):
                tangent.possible = False

            if best_path.length > arc_line_arc_length and tangent.possible:
                best_path.length = arc_line_arc_length

                best_path.start = self.car_location
                best_path.tangent_start = tangent.start
                best_path.tangent_end = tangent.end
                best_path.tangent_length = Vec3.length(
                    tangent.end - tangent.start)
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

        # self.renderer.draw_polyline_3d([best_path.start, best_path.tangent_start, best_path.tangent_end, best_path.end], self.renderer.red())
        return(best_path)

    def getCrossTangents(self, C1, C2, car_location, target_direction, target_location):

        # middle circle
        C3 = Circle()
        C3.location = C1.location + (C2.location - C1.location)*0.5
        C3.radius = Vec3.length((C2.location - C1.location)*0.5)

        # bigger car circle
        C4 = Circle()
        C4.location = C1.location
        C4.radius = C1.radius + C2.radius

        # bigger target circle
        C5 = Circle()
        C5.location = C2.location
        C5.radius = C1.radius + C2.radius

        C4intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C4.location.x, C4.location.y, C4.radius)
        C5intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C5.location.x, C5.location.y, C5.radius)

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
        C3.location = C1.location + (C2.location - C1.location)*0.5
        C3.radius = Vec3.length((C2.location - C1.location)*0.5)

        # bigger car circle
        C4 = Circle()
        C4.location = C1.location
        C4.radius = C1.radius - C2.radius + 1

        # bigger target circle
        C5 = Circle()
        C5.location = C2.location
        C5.radius = C2.radius - C1.radius + 1

        C4intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C4.location.x, C4.location.y, C4.radius)
        C5intersections = self.getIntersections(
            C3.location.x, C3.location.y, C3.radius, C5.location.x, C5.location.y, C5.radius)

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

        d = math.sqrt((x1-x0)**2 + (y1-y0)**2)

        a = (r0**2-r1**2+d**2)/(2*d)
        if(r0**2-a**2 < 0):
            return (0, 0, 0, 0, False)
        h = math.sqrt(r0**2-a**2)
        x2 = x0+a*(x1-x0)/d
        y2 = y0+a*(y1-y0)/d
        x3 = x2+h*(y1-y0)/d
        y3 = y2-h*(x1-x0)/d

        x4 = x2-h*(y1-y0)/d
        y4 = y2+h*(x1-x0)/d

        return (x3, y3, x4, y4, True)



    def getPointsInSircle(self, every, radius, center):
        circle_positions = []
        for i in range(every + 1):

            angle = 2 * math.pi / every * (i+1)
            location = Vec3(radius * math.sin(angle),
                            radius * -math.cos(angle), 0) + center
            location.z = 4
            circle_positions.append(location)
        return(circle_positions)

    def renderArcLineArcPath(self, path):
        
        p = self.getPointsInSircle(20, path.c1_radius, path.c1_center)
        self.renderer.draw_polyline_3d(p, self.renderer.purple())
        self.renderer.draw_line_3d(
            path.tangent_start, path.tangent_end, self.renderer.purple())
        p = self.getPointsInSircle(20, path.c2_radius, path.c2_center)
        self.renderer.draw_polyline_3d(p, self.renderer.purple())
        

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
