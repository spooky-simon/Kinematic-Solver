import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm
from numpy import dot, sin, cos, sign, cross
from typing import Tuple, List
from dataclasses import dataclass
import sys


class Point:
    def __init__(self, coords):
        """
        The potatoes of the meat and potates of the code
        """
        self.coords = np.array(coords)  # to track coordinates of the point
        self.links = {}  # a dictionary used to link points to each other
        self.jhist = []  # jounce history
        self.rhist = []  # rebound history
        self.hist = []  # total travel history
        self.friends = []  # list of points that this point is linked to
        self.origin = np.array(coords)  # to keep track of the static position

    def jr_combine(self):
        """
        Function used to combine jounce and rebound history
        :return: Full History
        """
        self.rhist.reverse()
        self.hist = self.rhist + self.jhist
        return self.hist

    def __repr__(self) -> str:
        return self.coords.__str__()

    def Ass_Fcn(self):
        """
        Associative Function
        
        This describes how much longer each link is than it should be

        :return: Gx
        """
        Gx = []
        for friend in self.friends:
            v = self.coords - friend.coords  # link vector
            v_norm = norm(v)  # link length
            l = self.links[friend]  # link target
            Gx.append(v_norm - l)
        return Gx

    def Obj_Fcn(self):
        """
        Objective Function (cost fcn)
        
        Lower is better

        :return: Cost
        """
        Fx = []
        for friend in self.friends:
            v = self.coords - friend.coords  # link vector
            v_norm = norm(v)  # link length
            l = self.links[friend]  # link target
            Fx.append(v_norm - l)
        Fx = [(i ** 2) * 0.5 for i in Fx]
        return Fx

    def Jacobian(self):
        """
        Jacobian of Objective Function

        :return: Jacobian
        """
        Jcb = []
        for friend in self.friends:
            df = 2 * (self.coords - friend.coords)
            Jcb.append(df)
        return np.array(Jcb)


def angle(v1: List[float], v2: List[float]):
    """
    Angle between two vectors
    Arccos of the dot product of two unit vectors

    :param v1: Vector 1
    :param v2: Vector 2
    :return: Angle
    """
    uv1 = v1 / norm(v1)
    uv2 = v2 / norm(v2)
    dot12 = dot(uv1, uv2)
    ang = np.degrees(np.arccos(dot12))
    return (ang)


def pt_to_ln(pt, a, b):
    """
    Shortest vector between a point and a line defined by points 'a' and 'b'
    This vector is also perpendicular to the line ab
    Used in this code to determine geometric steering arm

    :param pt: A point
    :param a: A point that defines line ab, also used as the arbitrary point on the line in this algorithm
    :param b: A point that defines line ab
    :return: Shortest vector from pt to line ab
    """
    # This function calculates the shortest vector from point pt to a line
    # that goes from a to b
    # The vector that is returned is point to line or ptl
    ln_ab = [a - b for a, b in zip(a, b)]
    ln_ap = [a - b for a, b in zip(a, pt)]
    ab_u = ln_ab / norm(ln_ab)
    ptl = ln_ap - dot(ln_ap, ab_u) * ab_u
    return ptl


def perp(a):
    """
    Creates a perpendicular line segement of a 2-vector

    :param a: line, 2D vector
    :return: perpendicular vector
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    """
    Determines the instersection of lines A and B
    A = a2 - a1
    B = b2 - b1
    where a1, a2, b1, b2 are points on a plane
    """
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = dot(dap, db)
    num = dot(dap, dp)
    return (num / denom) * db + b1


@dataclass()
class StrutSolve:
    """
    Kinematics Solver for MacPherson Strut
    """
    # Suspension Points
    wheel_center: Point
    lower_wishbone: Tuple[Point, Point, Point]  # Fore_Inner, Aft_Inner, Upright_Point
    strut_mount: Point
    wheel_center_offset: Point
    tie_rod: Tuple[Point, Point]  # Inner, Outer

    # Suspension Setup
    full_jounce: float
    full_rebound: float

    # Points of the suspension that will move
    moving_points: List[Point]

    unit: str = "mm"

    # Solved values
    sa = None
    bump_steer: List[float] = None
    camber_gain: List[float] = None
    caster_gain: List[float] = None
    roll_angle = None
    bump_zs = None
    roll_center = None
    instant_center = None

    def solve(self,
              steps: int = 5,
              happy: float = 10 ** -3,
              learning_rate: float = 10 ** -3,
              offset_toe: float = 0,
              offset_camber: float = 0,
              offset_caster: float = 0,
              ):
        """
        Solves stuff

        :param steps: Number of steps in each direction. (e.g. 10 -> 20 datapoints)
        :param happy: Error margin for gradient descent to be considered complete
        :param learning_rate: Learning rate
        :param offset_toe: Static offset
        :param offset_camber: Static offset
        :param offset_caster: Static offset
        """
        
        pt1 = pt_to_ln(self.wheel_center.coords,
                       self.strut_mount.coords,
                       self.lower_wishbone[2].coords) + self.wheel_center.origin
        wcl = norm(self.lower_wishbone[2].origin-pt1)
        v = self.strut_mount.coords - self.lower_wishbone[2].coords
        u = v / norm(v)
        self.wheel_center_offset.coords = u * wcl + self.lower_wishbone[2].coords
        self.wheel_center_offset.origin = self.wheel_center_offset.coords

        """ Gradient Descent """
        # Derive link lengths for use in Grad Descent
        for pt in self.moving_points:
            for friend in pt.friends:
                link = norm(np.array(pt.coords - friend.coords))
                pt.links.update({friend: link})
                
        # Error Checking
        for pt in self.moving_points:
            for friend in pt.friends:
                if pt == friend:
                    sys.exit("Point can't be it's own friend... :(")
            if len(pt.friends) < 1:
                sys.exit("Trying to move a point with no friends")
        
        print("Solving for Jounce Kinematics...")
        t0 = time.time_ns()
        err = []
        v_move = [0, 0, self.full_jounce / steps]
        # This is the good stuff
        # I'm straight up not explaining gradient decsent in code comments
        # but it loops through each step and solves the whole shibang muy bueno
        for i in range(0, steps):
            for pt in self.moving_points:
                pt.jhist.append(pt.coords)
                pt.coords = pt.coords + v_move
            self.wheel_center_offset.jhist.append(self.wheel_center_offset.coords)
            window = 1
            while window > happy:
                for pt in self.moving_points:
                    if pt == self.wheel_center:
                        v = self.strut_mount.coords - self.lower_wishbone[2].coords
                        u = v / norm(v)
                        self.wheel_center_offset.coords = u * wcl + self.lower_wishbone[2].coords
                    J = pt.Jacobian()
                    G = pt.Ass_Fcn()
                    E = pt.Obj_Fcn()
                    JT = J.T
                    step = learning_rate * JT @ G
                    pt.coords = pt.coords - step
                err.append(sum(E))
                window = sum(E)
        t1 = time.time_ns()
        print("Solved Jounce in", (t1 - t0) / 10 ** 6, "ms")

        # Reset to do rebound
        for pt in self.moving_points:
            pt.coords = pt.origin
        mid_pt_index = 1
        err = []
        v_move = [0, 0, self.full_rebound / steps]

        print("Solving for Rebound Kinematics...")
        t0 = time.time_ns()
        for i in range(0, steps):
            window = 1
            for pt in self.moving_points:
                pt.rhist.append(pt.coords)
                pt.coords = pt.coords + v_move
            self.wheel_center_offset.rhist.append(self.wheel_center_offset.coords)
            while window > happy:
                for pt in self.moving_points:
                    if pt == self.wheel_center:
                        v = self.strut_mount.coords - self.lower_wishbone[2].coords
                        u = v / norm(v)
                        self.wheel_center_offset.coords = u * wcl + self.lower_wishbone[2].coords
                    J = pt.Jacobian()
                    G = pt.Ass_Fcn()
                    E = pt.Obj_Fcn()
                    JT = J.T
                    step = learning_rate * JT @ G
                    pt.coords = pt.coords - step
                err.append(sum(E))
                window = sum(E)
        t1 = time.time_ns()
        print("Solved rebound in", (t1 - t0) / 10 ** 6, "ms")

        # Combine Jounce and Rebound into a single list
        for pt in self.moving_points:
            pt.jr_combine()
        self.wheel_center_offset.jr_combine()

        print("Calculating kinematic changes over wheel travel:")
        print("* Bump Steer")
        # Projecting the steeing arm into the XY plane and measuring the angle
        sa = [pt_to_ln(tro, self.strut_mount.coords, lo) for tro, lo in
              zip(self.tie_rod[1].hist, self.lower_wishbone[2].hist)]
        # sa_xy = [[x, y] for x, y, z in sa]  # project into xy plane (top view)
        # sa = [[x1y1[0]-x2y2[0],x1y1[1]-x2y2[1]] for x1y1,x2y2 in zip(self.wheel_center_offset.hist,self.tie_rod[1].hist)]
        sa_xy = [[x, y] for x, y, z in sa]  # project into xy plane (top view)
        bmp_str = [-angle(v, [1, 0]) for v in sa_xy]  # angle bw v1 and x axis
        bmp_str = [i - bmp_str[steps] + offset_toe for i in bmp_str]

        print("* Camber Gain")
        # Projects the kingpin into the YZ plane to meaure camber
        # by measuring the angle between the kingpin and the Z axis
        kp = [self.strut_mount.coords - b for b in self.lower_wishbone[2].hist]
        kp_yz = [[y, z] for x, y, z in kp]  # project into YZ plane (front view)
        cbr_gn = [-angle([0, 1], v) for v in kp_yz]  # compare to z axis
        cbr_gn = [i - cbr_gn[steps] + offset_camber for i in cbr_gn]  # compares to static
        # print(cbr_gn)

        print("* Caster changes")
        # Projects the kingpin into the YZ plane to meaure caster
        # by measuring the angle between the kingpin and the Z axis
        kp_xz = [[x, z] for x, y, z in kp]  # project into XZ plane (side view)
        cstr_gn = [-angle([0, 1], v) for v in kp_xz]  # compare to z axis
        cstr_gn = [i - cstr_gn[steps] + offset_caster for i in cstr_gn]  # compares to static

        # bump_zs is a list of the z height for each iterable in the code compared to static
        # roll_ang is a list of the body roll of the vehicle for each iterable in the code compared to static

        bump_zs = [z - self.wheel_center.origin[2] for x,y,z in self.wheel_center.hist]
        roll_ang = [np.degrees(np.sin(x / self.wheel_center.origin[1])) for x in bump_zs]

        print("* Roll center")
        # line intersecting functions taken from
        # https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/%7Erod/2500/notes/numpy-arrays/numpy-arrays.html
        # I don't really get the whole thing but it works so I don't need to think about it

        # lwr_mid finds average point of anti-squat/dive geometry
        # project to yz plane
        # project to yz plane
        lwr_mid = (self.lower_wishbone[0].coords+self.lower_wishbone[1].coords)/2
        lwr = [lwr_mid[1:] for i in self.lower_wishbone[2].hist]
        lo_yz = [i[1:] for i in self.lower_wishbone[2].hist]
        # print(self.wheel_center.hist[-1][2]-self.wheel_center.origin[2])
        # print(self.wheel_center.hist[-1][1:])
        # print(lwr[-1])
        # print(lo_yz[-1])
        
        strut_yz = [self.strut_mount.coords[1:] for i in self.lower_wishbone[2].hist]
        
        strut = [a-b for a,b in zip(strut_yz,lo_yz)]
        strut_perp = [perp(v) for v in strut]
        b2 = [a+b for a,b in zip(strut_perp,strut_yz)]
        ic_pts = zip(lwr, lo_yz, strut_yz, b2)
        
        ic = [seg_intersect(a1, a2, b1, b2) for a1, a2, b1, b2 in ic_pts]
        op_ic = [np.array([-y,z]) for y,z in ic]
        
        # Find vector from wc to cp at static (v_0)
        # this vector will orignate at the origin which makes rotation easy
        # rotate it by camber gain in yz plane
        # move new vector back to the wc
        # this is now the contact patch
        v_0 = [0,-self.wheel_center.origin[2]]
        # rotate method found here:
        # https://matthew-brett.github.io/teaching/rotation_2d.html
        v_y = [cos(np.radians(a))*v_0[0] - sin(np.radians(a))*v_0[1] for a in cbr_gn]
        v_z = [sin(np.radians(a))*v_0[0] + cos(np.radians(a))*v_0[1] for a in cbr_gn]
        v_yz = [[y,z] for y,z in zip(v_y,v_z)]
        cp_yz = [wc[1:] + v for wc,v in zip(self.wheel_center.hist,v_yz)]
        # print(ic[-1])
        # fa_y = [y for x,y,z in self.wheel_center.hist]
        # fa_z = [z - self.wheel_center.origin[2] for x,y,z in self.wheel_center.hist]
        # fa_gd = [[y,z] for y,z in zip(fa_y,fa_z)]
        op_cp_yz = [np.array([-y,z]) for y,z in cp_yz]
        # Roll Center in Roll
        op_ic.reverse()
        op_cp_yz.reverse()
        rcr_pts = zip(cp_yz,ic,op_cp_yz,op_ic)
        rcr = [seg_intersect(a1, a2, b1, b2) for a1,a2,b1,b2 in rcr_pts]
        
        gnd_pln_mid_pt_v = [(a-b)/2 for a,b in zip(cp_yz,op_cp_yz)]
        gnd_pln_mid_pt = [pt-v for pt,v in zip(cp_yz,gnd_pln_mid_pt_v)]  # off by 0.05mm when checked against solidworks
        # To find distances to this point
        # Can just find the magnitude of the perp vector for Z
        # then find the distance between the projected pt and the mid pt
        rc_pt_to_ln = [pt_to_ln(pt,cp,opp_cp) for pt,cp,opp_cp in zip(rcr,cp_yz,op_cp_yz)]
        rcr_z = [norm(v) for v in rc_pt_to_ln] # off by 0.06mm when checked in solidworks
        rc_projected = [rc+v for rc,v in zip(rcr,rc_pt_to_ln)]
        rcr_y = [norm(rc_proj-mid_pt) for rc_proj,mid_pt in zip(rc_projected,gnd_pln_mid_pt)]
        # this now gives only positive values
        # creating indexes to see what is positive and what is negative
        # Use cross product to find angle
        z_ind = [-sign(cross(a-b,a-c)) for a,b,c in zip(rcr,cp_yz,op_cp_yz)]
        # Need a midpoint vertical line to say left or right
        # We already have the mid point, only need another point perpendicular
        # Luckily we have a perp v to gnd
        c = [pt + v for pt,v in zip(gnd_pln_mid_pt,rc_pt_to_ln)]
        y_ind = [-sign(cross(a-b,a-c)) for a,b,c in zip(rcr,gnd_pln_mid_pt,c)]
        rcr = [[y*y_ind,z*z_ind] for y,z,y_ind,z_ind in zip(rcr_y,rcr_z,y_ind,z_ind)]

        # Save calculated values
        self.sa = sa
        self.camber_gain = cbr_gn
        self.caster_gain = cstr_gn
        self.roll_angle = roll_ang
        self.bump_zs = bump_zs
        self.bump_steer = bmp_str
        self.roll_center = rcr
        self.instant_center = ic
        self.offset_camber = offset_camber

    def plot(self,
             suspension: bool = True,
             bump_steer: bool = True,
             camber_gain: bool = True,
             caster_gain: bool = True,
             roll_center_in_roll=True,
             bump_steer_in_deg: bool = False,
             camber_gain_in_deg: bool = False,
             caster_gain_in_deg: bool = False,
             ):
        """
        Put %matplotlib into the kernel to get pop-out graphs that are interactive
        if you fix my bad graphs, send me your fix pls

        :param suspension: Plot suspension graph
        :param bump_steer: Plot bump steer vs vertical travel
        :param camber_gain: Plot camber gain vs vertical traval
        :param caster_gain: Plot caster gain
        :param roll_center_in_roll: Path of roll center as the car rolls
        :param bump_steer_in_deg: Sets y-axis of bump steer plot to roll angle in deg
        :param camber_gain_in_deg: Sets y-axis of camber gain plot to roll angle in deg
        :param caster_gain_in_deg: Sets y-axis of caster gain plot to roll angle in deg
        """

        if suspension:
            print("Plotting suspension graph(?)...")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=18., azim=-51)
            for pt in [self.wheel_center,
                       self.strut_mount,
                       self.lower_wishbone[0],
                       self.lower_wishbone[1],
                       self.lower_wishbone[2],
                       self.tie_rod[0],
                       self.tie_rod[1],
                       self.wheel_center_offset]:
                ax.scatter(pt.origin[0],pt.origin[1],pt.origin[2])
            for pt in self.moving_points:
                xs,ys,zs = zip(*pt.hist)
                ax.plot(xs,ys,zs)
            xs,ys,zs = zip(self.lower_wishbone[0].origin,
                           self.lower_wishbone[2].origin)
            ax.plot(xs,ys,zs)
            xs,ys,zs = zip(self.lower_wishbone[1].origin,
                           self.lower_wishbone[2].origin)
            ax.plot(xs,ys,zs)
            xs,ys,zs = zip(self.tie_rod[0].origin,self.tie_rod[1].origin)
            ax.plot(xs,ys,zs)
            xs,ys,zs = zip(self.lower_wishbone[2].origin,self.strut_mount.origin)
            ax.plot(xs,ys,zs)
            print(self.wheel_center_offset.hist)
            xs,ys,zs = zip(*self.wheel_center_offset.hist)
            ax.plot(xs,ys,zs)

        if camber_gain:
            fig, ax = plt.subplots()
            ax.axhline(y= 0,color='k', linestyle ='dashed', alpha = 0.25)
            if camber_gain_in_deg:
                print("Plotting Camber Gain vs Vehicle Roll...")
                # annotation
                (x,y) = (self.roll_angle[-1], self.camber_gain[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 50 if y < self.offset_camber else -50
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (10,ann_y), textcoords = 'offset points',
                            horizontalalignment='right',
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
                ax.plot(self.roll_angle, self.camber_gain, color = 'k')
                # axis label
                ax.set_xlabel('Vehicle Roll [deg]')
            else:
                print("Plotting Camber Gain vs Vertical Travel...")
                ax.set_xlabel('Vertical Wheel Center Travel [' + self.unit + ']')
                ax.plot(self.bump_zs, self.camber_gain, color = 'k')
                # annotation
                (x,y) = (self.bump_zs[-1], self.camber_gain[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 50 if y < self.offset_camber else -50
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (10,ann_y), textcoords = 'offset points',
                            horizontalalignment='right',
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            ax.set_ylabel('Camber Change [deg]')
            ax.set_title('Camber Gain', pad = 15)

        if bump_steer:     
            fig, ax = plt.subplots()
            ax.axhline(y= 0,color='k', linestyle ='dashed', alpha = 0.25)
            if bump_steer_in_deg:
                print("Plotting Bump Steer vs Vehicle Roll...")
                ax.set_xlabel('Vehicle Roll [deg]')
                ax.plot(self.roll_angle, self.bump_steer, color = 'k')
                # annotation
                (x,y) = (self.roll_angle[-1], self.bump_steer[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 50 if y < 0 else -50
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (10,ann_y), textcoords = 'offset points',
                            horizontalalignment='right',
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            else:
                print("Plotting Bump Steer vs Vertical Travel...")
                ax.plot(self.bump_zs, self.bump_steer, color = 'k')
                ax.set_xlabel('Vertical Wheel Center Travel [' + self.unit + ']')
                # annotation
                (x,y) = (self.bump_zs[-1], self.bump_steer[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 50 if y < 0 else -50
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (10,ann_y), textcoords = 'offset points',
                            horizontalalignment='right',
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            ax.set_ylabel('Toe [deg]\n<- toe in     toe out->', multialignment='center')
            ax.set_title('Bump Steer', pad = 15)

        if caster_gain:
            fig, ax = plt.subplots()
            ax.axhline(y= 0,color='k', linestyle ='dashed', alpha = 0.25)
            if caster_gain_in_deg:
                print("Plotting Caster Gain vs Vehicle Roll...")
                ax.plot(self.roll_angle, self.caster_gain, color = 'k')
                ax.set_xlabel('Vehicle Roll [deg]')
                # annotation
                (x,y) = (self.roll_angle[-1], self.caster_gain[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 50 if y < 0 else -50
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (10,ann_y), textcoords = 'offset points',
                            horizontalalignment='right',
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            else:
                print("Plotting Caster Gain vs Vertical Travel...")
                ax.plot(self.bump_zs, self.caster_gain, color = 'k')
                ax.set_xlabel('Vertical Wheel Center Travel [' + self.unit + ']')
                # annotation
                (x,y) = (self.bump_zs[-1], self.caster_gain[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 50 if y < 0 else -50
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (10,ann_y), textcoords = 'offset points',
                            horizontalalignment='right',
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            ax.set_ylabel('Caster Change [deg]')
            ax.set_title('Caster Gain', pad = 15)

        if roll_center_in_roll:
            print("Plotting Path of Roll Center as Car Rolls...")
            fig, ax = plt.subplots()
            xs = [xyz[0] for xyz in self.roll_center]
            ys = [xyz[1] for xyz in self.roll_center]
            ax.scatter(xs, ys)
            # i = len(self.upper_wishbone[2].hist)//2
            # x1,y1,z1 = self.upper_wishbone[2].hist[i]
            # y2, z2 = self.instant_center[i]
            # ax.plot((y1,y2),(z1,z2))
            ax.set_ylabel('Vertical Roll Center Travel [' + self.unit + ']')
            ax.set_xlabel('Horizontal Roll Center Travel [' + self.unit + ']')
            ax.set_title('Dynamic Roll Center')

        plt.show()
