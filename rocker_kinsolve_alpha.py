# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 14 # just to make titles more biggerer
import time
from numpy.linalg import norm
from numpy import dot, sin, cos, cross, sign
from typing import Tuple, List
from dataclasses import dataclass
import matplotlib.collections as mcoll
import sys

class Point:
    def __init__(self, coords):
        """
        The potatoes of the meat and potates of the code
        
        A "Point" is used to keep track of the location history for all moving points
        as well as what points it is linked to which I call its "friends"
        and how far it needs to be from its friends to be "happy"
        """
        self.coords = np.array(coords)  # to track coordinates of the point
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
        # Last Value in rhist is the origin and we dont need a second copy of it
        self.hist = self.rhist[:-1] + self.jhist
        return self.hist

    def __repr__(self) -> str:
        return self.coords.__str__()

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
    return ang


def pt_to_ln(pt, a, b):
    """
    Shortest vector between a point and a line defined by points 'a' and 'b'
    This vector is also perpendicular to the line ab
    Used in this code to determine geometric steering arm and in roll center calcs

    :param pt: A point
    :param a: A point that defines line ab, also used as the arbitrary point on the line in this algorithm
    :param b: A point that defines line ab
    :return: Shortest vector from pt to line ab
    """
    # This function calculates the shortest vector from point pt to a line
    # that goes from a to b
    # The vector that is returned is point to line or ptl
    ln_ab = [a - b for a, b in zip(a, b)]
    ln_ap = [a - p for a, p in zip(a, pt)]
    ab_u = ln_ab / norm(ln_ab)
    ptl = ln_ap - dot(ln_ap, ab_u) * ab_u
    return ptl


def perp(a):
    """
    Creates a perpendicular vector for a given 2-vector
    Uses rule (y,-x) _|_ (x,y)

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
class KinSolve:
    """
    Kinematics Solver
    """
    # Suspension Points
    wheel_center: Point
    lower_wishbone: Tuple[Point, Point, Point]  # Fore_Inner, Aft_Inner, Upright_Point
    upper_wishbone: Tuple[Point, Point, Point]  # Fore_Inner, Aft_Inner, Upright_Point
    tie_rod: Tuple[Point, Point]  # Inner, Outer
    p_rod: [Point, Point] # Inner, Outer
    rocker: Point # center of rocker rotation
    shock: [Point, Point] # Lower, Upper

    # Suspension Setup
    full_jounce: float
    full_rebound: float

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
        
        moving_points = [self.wheel_center,
                          self.tie_rod[1],
                          self.upper_wishbone[2],
                          self.lower_wishbone[2]]

        """ Gradient Descent """
        # Derive link lengths for use in Grad Descent
        linked_pairs = []
        for pt in moving_points:
            for friend in pt.friends:
                if [friend,pt] not in linked_pairs:
                    linked_pairs.append([pt,friend])
        link_lens = [norm(a.origin-b.origin) for [a,b] in linked_pairs]
        # link_lens = (norm(a.origin-b.origin) for [a,b] in linked_pairs)
        # self.p_rod[0]nt([i for i in link_lens])
                
        # Error Checking
        for pt in moving_points:
            for friend in pt.friends:
                if pt == friend:
                    sys.exit("Point can't be it's own friend... :(")
            if len(pt.friends) < 1:
                sys.exit("Trying to move a point with no friends")
        
        print("Solving for Jounce Kinematics...")
        # This is the good stuff
        # This is the "meat" of the meat and potatoes of the code
        # I'm straight up not explaining gradient decsent in code comments
        # but it loops through each step and solves the whole shibang muy bueno
        
        # Step 0: Initialize the clock for timing,
        # the err list for error tracking which I dont have turned on in this code, but if you graph it, its very cool,
        # and the initial vector that is starting the next grad descent problem
        t0 = time.time_ns()
        err = []
        v_move = [0, 0, self.full_jounce / steps]
        
        # As much as I hate for loops, this has to be done sequentially, as subsequent steps
        # rely on the previous step for its initial condition
        # this isnt that slow because.. i dont know how computers work so idk
        for i in range(0, steps):
            
            # Step 1: Loop through all movable points and append the current
            #         location to the points' respective jounce history
            for pt in moving_points:
                pt.jhist.append(pt.coords)
                pt.coords = pt.coords + v_move
            
            # Step 2: Initialize the error value at a value higher that "happy"
            # 1 was chosen because I have a simple mind
            error = 1
            err.append([error])
            
            # Step 3: while loop that contains the grad descent work for each step
            # The while loop does grad descent steps until the error value is low enough
            # The grad descent basically adjusts each movable point until the link lengths are close enough to the original link lengths to call it good
            # If all the link lengths are very very close to the static link lengths, you have found a feasible suspension articulation
            # This whole script relies on the fact that this grad descent algo will just find the closest local miminum and chill
            while error > happy:               
                link_lens2 = [norm(a.coords-b.coords) for [a,b] in linked_pairs]
                ass_func = [a-b for a,b in zip(link_lens2,link_lens)]
                obj_func = [(i**2)*0.5 for i in ass_func]
                # Jacobian used here  is the matrix derivative of the linear vector norm
                jcbn = [2 * (a.coords - b.coords) for a,b in linked_pairs]
                step = [a*b*learning_rate for a,b in zip(jcbn,ass_func)]
                error = sum(obj_func)
                for pair,step_ in zip(linked_pairs,step):
                    pair[0].coords = pair[0].coords - step_
                err[i].append(error)
            
            # Step 4: Once the suspension has moved, we can update the move vector to follow the arc traced out by the wheel center
            # This enables us to start each grad descent problem closer to a local minimum speeding up the whole thing mega fast
            # We move each point by the amount the wheel center moved despite them not moving the same
            # This is one of those "good enough" things. I didn't want to bother dithering each point by a different vector
            # keeping track of more than one vector and trying to match it seemed like more effort than it was worth
            v_move[1] = (self.wheel_center.coords[1] - self.wheel_center.jhist[-1][1])
            
            # Repeat Steps 1-4 for all points requested
        
        # Step 5: append the final points' locations to their jounce histories
        for pt in moving_points:
            pt.jhist.append(pt.coords)
        
        # Stop the clock and report time
        t1 = time.time_ns()
        print("Solved Jounce in", (t1 - t0) / 10 ** 6, "ms")

        # Reset to do rebound
        # All comments in Jounce apply here for Rebound
        for pt in moving_points:
            pt.coords = pt.origin
        err = []
        v_move = [0, 0, self.full_rebound / steps]

        print("Solving for Rebound Kinematics...")
        t0 = time.time_ns()
        for i in range(0, steps): 
            for pt in moving_points:
                pt.rhist.append(pt.coords)
                pt.coords = pt.coords + v_move
            error = 1
            err.append([])
            while error > happy:
                link_lens2 = [norm(a.coords-b.coords) for [a,b] in linked_pairs]
                ass_func = [a-b for a,b in zip(link_lens2,link_lens)]
                obj_func = [(i**2)*0.5 for i in ass_func]
                jcbn = [2 * (a.coords - b.coords) for a,b in linked_pairs]
                step = [a*b*learning_rate for a,b in zip(jcbn,ass_func)]
                error = sum(obj_func)
                for pair,step_ in zip(linked_pairs,step):
                    pair[0].coords = pair[0].coords - step_
                err[i].append(error)
            v_move[1] = self.wheel_center.coords[1] - self.wheel_center.rhist[-1][1]
        for pt in moving_points:
            pt.rhist.append(pt.coords)
        t1 = time.time_ns()
        print("Solved rebound in", (t1 - t0) / 10 ** 6, "ms")

        # Combine Jounce and Rebound into a single list
        for pt in moving_points:
            pt.jr_combine()
        
        # Now we have all of our suspension kinematics
        # We can use these to do math and figure out kinematic behavior
        print("Calculating kinematic changes over wheel travel:")
        
        # Bump steer is first up
        print("* Bump Steer")
        # Calculate the 'gemoetric steering arm' by finding the vector from tie rod pickup point to king pin
        # Projecting the steeing arm into the XY plane
        # Measuring the angle change from the static position
        sa = [pt_to_ln(tro, uo, lo) for tro, uo, lo in
              zip(self.tie_rod[1].hist, self.upper_wishbone[2].hist, self.lower_wishbone[2].hist)]
        sa_xy = [[x, y] for x, y, z in sa]  # project into xy plane (top view)
        bmp_str = [angle(v, [1, 0]) for v in sa_xy]  # angle bw v1 and x axis
        bmp_str = [i - bmp_str[steps] + offset_toe for i in bmp_str] # campare to static

        print("* Camber Gain")
        # Projects the kingpin into the YZ plane to meaure camber
        # by measuring the angle between the kingpin and the Z axis
        kp = [a - b for a, b in zip(self.upper_wishbone[2].hist, self.lower_wishbone[2].hist)]
        kp_yz = [[y, z] for x, y, z in kp]  # project into YZ plane (front view)
        cbr_gn = [-angle(v,[0, 1]) for v in kp_yz]  # angle between kp and z-axis in 2d
        cbr_gn = [i - cbr_gn[steps] + offset_camber for i in cbr_gn]  # compares to static

        print("* Caster changes")
        # Projects the kingpin into the YZ plane to meaure caster
        # by measuring the angle between the kingpin and the Z axis
        kp_xz = [[x, z] for x, y, z in kp]  # project into XZ plane (side view)
        cstr_gn = [-angle([0, 1], v) for v in kp_xz]  # angle between kp and z-axis in 2d
        cstr_gn = [i - cstr_gn[steps] + offset_caster for i in cstr_gn]  # compares to static

        print("* Roll center")
        # line intersecting functions taken from
        # https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/%7Erod/2500/notes/numpy-arrays/numpy-arrays.html
        # I don't really get the whole thing but it works so I don't need to think about it

        # ui_mid finds average point of anti-squat/dive geometry
        # project to yz plane
        ui_mid = (self.upper_wishbone[0].coords+self.upper_wishbone[1].coords)/2
        upr = [ui_mid[1:] for i in self.upper_wishbone[2].hist]
        lwr = [self.lower_wishbone[0].coords[1:] for i in self.lower_wishbone[2].hist]
        uo_yz = [i[1:] for i in self.upper_wishbone[2].hist]
        lo_yz = [i[1:] for i in self.lower_wishbone[2].hist]
        ic_pts = zip(upr, uo_yz, lwr, lo_yz)
        ic = [seg_intersect(a1, a2, b1, b2) for a1, a2, b1, b2 in ic_pts]
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
        cp_yz = [wheel_center[1:] + v for wheel_center,v in zip(self.wheel_center.hist,v_yz)]
        # Roll Center in Heave
        # opp variables are for opposite side
        opp_ic = [np.array([-y, z]) for y, z in ic]
        opp_cp_yz = [np.array([-y, z]) for y, z in cp_yz]
        
        # Roll Center in Roll
        opp_ic_r = opp_ic
        opp_ic_r.reverse()
        opp_cp_yz.reverse()
        # rcr is roll center in roll
        # Global Coordinates, not in rolled coordinates
        rc_pts = zip(cp_yz, ic, opp_cp_yz, opp_ic)
        rc = [seg_intersect(a1, a2, b1, b2) for a1, a2, b1, b2 in rc_pts]
        # This will give us roll center in global coordinates, not relative to the new ground plane the car is on after it rolled
        # the new 0,0,0 will be the mid point of the line between the two contact points
        gnd_pln_mid_pt_v = [(a-b)/2 for a,b in zip(cp_yz,opp_cp_yz)]
        gnd_pln_mid_pt = [pt-v for pt,v in zip(cp_yz,gnd_pln_mid_pt_v)]  # off by 0.05mm when checked against solidworks
        # To find distances to this point
        # Can just find the magnitude of the perp vector for Z
        # then find the distance between the projected pt and the mid pt
        rc_pt_to_ln = [pt_to_ln(pt,cp,opp_cp) for pt,cp,opp_cp in zip(rc,cp_yz,opp_cp_yz)]
        rcr_z = [norm(v) for v in rc_pt_to_ln] # off by 0.06mm when checked in solidworks
        rc_projected = [rc+v for rc,v in zip(rc,rc_pt_to_ln)]
        rcr_y = [norm(rc_proj-mid_pt) for rc_proj,mid_pt in zip(rc_projected,gnd_pln_mid_pt)]
        # this now gives only positive values
        # creating indexes to see what is positive and what is negative
        # Use cross product to find angle
        z_ind = [-sign(cross(a-b,a-c)) for a,b,c in zip(rc,cp_yz,opp_cp_yz)]
        # Need a midpoint vertical line to say left or right
        # We already have the mid point, only need another point perpendicular
        # Luckily we have a perp v to gnd
        c = [pt + v for pt,v in zip(gnd_pln_mid_pt,rc_pt_to_ln)]
        y_ind = [-sign(cross(a-b,a-c)) for a,b,c in zip(rc,gnd_pln_mid_pt,c)]
        rcr = [[y*y_ind,z*z_ind] for y,z,y_ind,z_ind in zip(rcr_y,rcr_z,y_ind,z_ind)]
        
        print("* Scrub Radius changes")
        # Intersects kingpin_yz with line [0,0]
        # then gets norm from where the king pin intersects to contact patch
        # ground line is found using contact patches found in roll calc
        sr_pts = zip(uo_yz,lo_yz,cp_yz,opp_cp_yz)
        kpi_int = [seg_intersect(a1, a2, b1, b2) for a1,a2,b1,b2 in sr_pts]
        sr = [norm(a-b) for a,b in zip(kpi_int,cp_yz)]
        
        # bump_zs is a list of the z height for each iterable in the code compared to static
        # roll_ang is a list of the body roll of the vehicle for each iterable in the code compared to static
        bump_zs = [z - self.wheel_center.origin[2] for x,y,z in self.wheel_center.hist]
        roll_ang = [-np.degrees(sin(z / (self.wheel_center.origin[1]))) for z in bump_zs]
        
        # Rocker calcs
        print("* Shock Travel")

        # wheel_side_lever = norm(pri-rkr)
        wheel_side_lever = norm(pt_to_ln(self.rocker.origin,self.p_rod[0].origin,self.p_rod[1].origin))
        shock_side_lever = norm(pt_to_ln(self.rocker.origin,self.shock[1].origin,self.shock[0].origin))
        self.static_motion_ratio = shock_side_lever / wheel_side_lever
        #print("static motion ratio based on rocker geomtery is:", static_motion_ratio)

        # Figure out where push rod pickup point is
        # Find equation of circle that push rod rotates through
        # by intersection of spheres
        # Step 1: figure out if pickup point of push rod is on upper or lower a-arm
        dist_to_upper = norm(self.p_rod[1].origin - self.upper_wishbone[2].origin)
        dist_to_lower = norm(self.p_rod[1].origin - self.lower_wishbone[2].origin)
        if dist_to_upper > dist_to_lower:
            # Sphere 1 is centered around lfi
            # Sphere 2 is centered around lai
            c1 = self.lower_wishbone[0].origin
            c2 = self.lower_wishbone[1].origin
        else:
            # Sphere 1 is centered around ufi
            # Sphere 2 is centered around uai
            c1 = self.upper_wishbone[0].origin
            c2 = self.upper_wishbone[1].origin

        def intersection_of_spheres(center_1,center_2,intersection_pt):
            #intersection of two spheres
            d  = norm(center_1 - center_2)
            r1 = norm(center_1 - intersection_pt)
            r2 = norm(center_2 - intersection_pt)
            h = 0.5 + (r1*r1 - r2*r2) / (2 * d*d)
            r_i = np.sqrt(r1*r1 - h*h * d*d)
            n_i = (center_2 - center_1)/d
            c_i = center_1 + h * (center_2 - center_1)
            return n_i, c_i, r_i

        (n_i,c_i,r_i) = intersection_of_spheres(c1, c2, self.p_rod[1].origin)
        # Intersect with sphere 3 which is cenetered around lo and has radius lo-pro
        # third sphere
        def intersection_sphere_circle(c_s,r_s,n_i,c_i,r_i):
            dp = dot(n_i, c_i - c_s) # distance of plane to sphere center
            c_p = c_s + dp*n_i # center of circle that is sphere section cut by plane
            r_p = np.sqrt(r_s*r_s - dp*dp) # radius of circle that is sphere section cut by plane
            if abs(dp) > r_s:
                print("distance between centers:", abs(dp))
                print("radius of sphere:", r_s)
                print("ruh roh, sphere does not intersect circle")
                return
            d = norm(c_i-c_p) # distance between centers
            if d > r_i + r_p:
                print("ruh roh, circles do not intersect?")
                return
            if d + min(r_i, r_p) < max(r_i, r_p):
                print("ruh roh, one circle is inside the other")
                return
            h = 0.5 + (r_i*r_i - r_p*r_p) / (2 * d*d) # ratio of circle sizes
            r_j = np.sqrt(r_i*r_i - h*h * d*d) # distance from center line 
            c_j = c_i + h * (c_p - c_i) # point along center line
            t = (np.cross(c_p - c_i, n_i))/norm(np.cross(c_p - c_i, n_i))
            p_0 = c_j - t * r_j
            p_1 = c_j + t * r_j
            return p_0,p_1
            
        def higher_pt(pt_0,pt_1):
            if pt_1[2] > pt_0[2]:
                return pt_1
            return pt_0

        # Populate pro.hist
        # print("Populating pro.hist")
        for c_s in self.lower_wishbone[2].hist:
            r_s = norm(self.lower_wishbone[2].origin-self.p_rod[1].origin)
            (p1,p2) = intersection_sphere_circle(c_s, r_s, n_i, c_i, r_i)
            p = higher_pt(p1,p2)
            self.p_rod[1].hist.append(p)

        # print("Pro.hist has been filled")

        # Circle of Rocker
        c_i = self.rocker.origin
        r_i = norm(self.rocker.origin - self.p_rod[0].origin)
        n_i = np.cross((self.rocker.origin - self.p_rod[0].origin),(self.rocker.origin - self.shock[0].origin))/norm(
            np.cross((self.rocker.origin - self.p_rod[0].origin),(self.rocker.origin - self.shock[0].origin)))
        
        # Populate pri.hist
        # print("Populating pri.hist")
        for c_s in self.p_rod[1].hist:
            r_s = norm(self.p_rod[0].origin-self.p_rod[1].origin)
            (p1,p2) = intersection_sphere_circle(c_s, r_s, n_i, c_i, r_i)
            p = higher_pt(p1,p2)
            self.p_rod[0].hist.append(p)

        # print("Pri.hist has been filled")

        # find rotation of rocker
        # theta  = cos-1 [ (a · b) / (|a| |b|) ]
        rkr_ang_r = np.zeros(steps)
        for i in range(1,steps+1):
            a = self.rocker.origin-self.p_rod[0].hist[i-1]
            b = self.rocker.origin-self.p_rod[0].hist[i]
            rkr_ang_r[i-1] = angle(a, b)

        rkr_ang_j = np.zeros(steps)
        for i in range(steps+1,steps+steps+1):
            a = self.rocker.origin-self.p_rod[0].hist[i-1]
            b = self.rocker.origin-self.p_rod[0].hist[i]
            rkr_ang_j[i-steps-1] = angle(a, b)
            
        # print(rkr_ang_r,rkr_ang_j)
        
        # def rotation_about_axis(pt,axis_a,axis_b,theta):
        def rotation_about_axis(pt,ctr,u_axis,theta):
            cs = np.cos(theta)
            sn = np.sin(theta)
            
            a = (pt-ctr)/norm(pt-ctr)
            b = np.cross(u_axis,a)
            a_r = cs*a + sn*b

            rot_pt = ctr+a_r*norm(pt-ctr)
            
            return rot_pt

        self.shock[1].jhist.append(self.shock[1].origin)
        for theta in rkr_ang_j:
            pt = rotation_about_axis(self.shock[1].jhist[-1], c_i, n_i, np.radians(theta))
            self.shock[1].jhist.append(pt)

        self.shock[1].rhist.append(self.shock[1].origin)
        # Rebound angles are stored counting from full rebound in rkr _ang_r for some reason
        # not going to try to understand what I did or why a few years ago but it's easy enough
        # to just chug through the array backwards
        for theta in rkr_ang_r[::-1]:
            pt = rotation_about_axis(self.shock[1].rhist[-1], c_i, -n_i, np.radians(theta))
            self.shock[1].rhist.append(pt)
        
        self.shock[1].jr_combine()

        

        # find deltas in wheel travel and shock travel
        wc_z = [z for x,y,z in self.wheel_center.hist]
        self.wheel_travel = np.diff(wc_z)
        # wheel_travel is 1 value shorter than wc_z
        self.shock_len = [norm(a-self.shock[0].origin) for a in self.shock[1].hist]
        self.shock_travel = [-l for l in np.diff(self.shock_len)]
        self.shock_travel2 = [a-b for a,b in zip(self.shock_len[:-1],self.shock_len[1:])]
        if any(t<0 for t in self.shock_travel):
            print("Ruh roh, your shock went over center")
        
        print("* Dynamic Motion Ratio")
        # Ok things get a little philosophical here
        # techinically this is the average motion ration between successive 
        # pairs of points that we have sampled. If our number of steps is 100
        # then we should have 201 points in the history of each moved point
        # but our motion ratio is calculated on the movement between points
        # so there would only be 200 values of "between points"
        # and none of them would be the tru value at any sampled point.
        # So I will do another round of "good enough" kinematics, where the 2
        # end points will just be the extremes of the "200" motion ratio values
        # and the other middle ones will be an average of the "198" other points
        # which should give us 2 + 199 = 201
        # and the ones ar the end will just be a little off
        avg_mr = [a/b for a,b in zip(self.shock_travel,self.wheel_travel) if b > 0]
        self.mr = [avg_mr[0]] + [(mr1+mr2)/2 for mr1,mr2 in zip(avg_mr[:-1],avg_mr[1:])] + [avg_mr[-1]]
        # print(self.mr)

        # Save calculated values
        self.steps = steps
        self.sa = sa
        self.camber_gain = cbr_gn
        self.caster_gain = cstr_gn
        self.roll_angle = roll_ang
        self.bump_zs = bump_zs
        self.bump_steer = bmp_str
        self.roll_center = rcr
        self.instant_center = ic
        self.contactpatch_yz = cp_yz
        self.scrub_radius = sr
        self.moving_points = moving_points
        return  self.sa,self.camber_gain, self.caster_gain, self.roll_angle, self.bump_zs, self.bump_steer, self.roll_center, self.instant_center, self.scrub_radius, self.moving_points

    def plot(self,
             suspension: bool = True,
             
             bump_steer: bool = True,
             bump_steer_in_deg: bool = False,

             camber_gain: bool = True,
             camber_gain_in_deg: bool = False,

             caster_gain: bool = True,
             caster_gain_in_deg: bool = False,

             scrub_gain: bool = True,
             scrub_gain_in_deg: bool = False,

             roll_center_in_roll=True,
             
             motion_ratio: bool = False,
             motion_ratio_in_deg: bool = False
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
            print("Visualizing the Suspension...")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=18., azim=26)
            for pt in [self.wheel_center, *self.upper_wishbone, *self.lower_wishbone, *self.tie_rod, *self.p_rod, self.rocker, *self.shock]:
                ax.scatter(pt.origin[0], pt.origin[1], pt.origin[2], color = "k", s = 5)
            for pt in self.moving_points:
                xs, ys, zs = zip(*pt.hist)
                ax.plot(xs, ys, zs, color = "grey")
            xs, ys, zs = zip(*self.shock[1].hist)
            ax.plot(xs, ys, zs, color = "grey")
            xs, ys, zs = zip(*self.p_rod[0].hist)
            ax.plot(xs, ys, zs, color = "grey")
            ax.plot((self.lower_wishbone[0].origin[0], self.lower_wishbone[2].origin[0]),
                    (self.lower_wishbone[0].origin[1], self.lower_wishbone[2].origin[1]),
                    (self.lower_wishbone[0].origin[2], self.lower_wishbone[2].origin[2]),
                    color = "blue")
            ax.plot((self.lower_wishbone[1].origin[0], self.lower_wishbone[2].origin[0]),
                    (self.lower_wishbone[1].origin[1], self.lower_wishbone[2].origin[1]),
                    (self.lower_wishbone[1].origin[2], self.lower_wishbone[2].origin[2]),
                    color = "blue")
            ax.plot((self.upper_wishbone[1].origin[0], self.upper_wishbone[2].origin[0]),
                    (self.upper_wishbone[1].origin[1], self.upper_wishbone[2].origin[1]),
                    (self.upper_wishbone[1].origin[2], self.upper_wishbone[2].origin[2]),
                    color = "r")
            ax.plot((self.upper_wishbone[0].origin[0], self.upper_wishbone[2].origin[0]),
                    (self.upper_wishbone[0].origin[1], self.upper_wishbone[2].origin[1]),
                    (self.upper_wishbone[0].origin[2], self.upper_wishbone[2].origin[2]),
                    color = "r")
            ax.plot((self.tie_rod[0].origin[0], self.tie_rod[1].origin[0]),
                    (self.tie_rod[0].origin[1], self.tie_rod[1].origin[1]),
                    (self.tie_rod[0].origin[2], self.tie_rod[1].origin[2]),
                    color = "g")
            ax.plot((self.lower_wishbone[2].origin[0], self.upper_wishbone[2].origin[0]),
                    (self.lower_wishbone[2].origin[1], self.upper_wishbone[2].origin[1]),
                    (self.lower_wishbone[2].origin[2], self.upper_wishbone[2].origin[2]))
            ax.plot((self.p_rod[0].origin[0], self.p_rod[1].origin[0]),
                    (self.p_rod[0].origin[1], self.p_rod[1].origin[1]),
                    (self.p_rod[0].origin[2], self.p_rod[1].origin[2]))
            ax.set_title('Corner Visualization', pad = 15)
            ax.set_xlabel('x ['+self.unit+']')
            ax.set_ylabel('y ['+self.unit+']')
            ax.set_zlabel('z ['+self.unit+']')
            # Code Below Shows Steering Arm change at full jounce and rebound
            # Used for debugging, you can ignore
            # x1 = [xyz[0] for xyz in self.tie_rod[1].hist]
            # y1 = [xyz[1] for xyz in self.tie_rod[1].hist]
            # z1 = [xyz[2] for xyz in self.tie_rod[1].hist]
            # sax = [xyz[0] for xyz in self.sa]
            # say = [xyz[1] for xyz in self.sa]
            # saz = [xyz[2] for xyz in self.sa]
            # x2 = [a+b for a,b in zip(x1,sax)]
            # y2 = [a+b for a,b in zip(y1,say)]
            # z2 = [a+b for a,b in zip(z1,saz)]
            # steps = len(x2)//2
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # for i in [0, steps, 2*steps-1]:
            #     ax.plot((x1[i],x2[i]),(y1[i],y2[i]),(z1[i],z2[i]))
            
            # Code Below Shows King Pin change at full jounce and rebound
            # Used for debugging, you can ignore
            # x1 = [xyz[0] for xyz in self.upper_wishbone[2].hist]
            # y1 = [xyz[1] for xyz in self.upper_wishbone[2].hist]
            # z1 = [xyz[2] for xyz in self.upper_wishbone[2].hist]
            # x2 =  [xyz[0] for xyz in self.lower_wishbone[2].hist]
            # y2 = [xyz[1] for xyz in self.lower_wishbone[2].hist]
            # z2 = [xyz[2] for xyz in self.lower_wishbone[2].hist]
            # steps = len(x2)//2
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # for i in [0, steps, 2*steps-1]:
            #     ax.plot((x1[i],x2[i]),(y1[i],y2[i]),(z1[i],z2[i]))
    
        if camber_gain:
            fig, ax = plt.subplots()
            ax.axhline(y= 0,color='k', linestyle ='dashed', alpha = 0.25)
            if camber_gain_in_deg:
                print("Plotting Camber Gain vs Vehicle Roll...")
                # annotation
                (x,y) = (self.roll_angle[-1], self.camber_gain[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 75 if y < 0 else -75
                x_align = 'right' if x > 0 else 'left'
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (0,ann_y), textcoords = 'offset points',
                            horizontalalignment=x_align,
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
                ann_y = 75 if y < 0 else -75
                x_align = 'right' if x > 0 else 'left'
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (0,ann_y), textcoords = 'offset points',
                            horizontalalignment=x_align,
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            ax.set_ylabel('Camber Change [deg]')
            ax.set_title('Camber', pad = 15)
            # Set min limits on y axis
            if abs(max(self.camber_gain)-min(self.camber_gain)) < 0.5:
                ax.set_ylim([min(self.camber_gain)-0.25,max(self.camber_gain)+0.25])
    
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
                ann_y = 75 if y < 0 else -75
                x_align = 'right' if x > 0 else 'left'
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (0,ann_y), textcoords = 'offset points',
                            horizontalalignment=x_align,
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            else:
                print("Plotting Bump Steer vs Vertical Travel...")
                ax.plot(self.bump_zs, self.bump_steer, color = 'k')
                ax.set_xlabel('Vertical Wheel Center Travel [' + self.unit + ']')
                # annotation
                (x,y) = (self.bump_zs[-1], self.bump_steer[-1])
                s = '('+str(round(x,2))+', '+str(round(y,2))+')'
                ann_y = 75 if y < 0 else -75
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
                ann_y = 75 if y < 0 else -75
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
                ann_y = 75 if y < 0 else -75
                ax.annotate(s, xy = (x,y), xycoords = 'data',
                            xytext = (10,ann_y), textcoords = 'offset points',
                            horizontalalignment='right',
                            arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))
            ax.set_ylabel('Caster Change [deg]')
            ax.set_title('Caster', pad = 15)
            
        if scrub_gain:
            fig, ax = plt.subplots()
            ax.axhline(y= 0,color='k', linestyle ='dashed', alpha = 0.25)
            if caster_gain_in_deg:
                print("Plotting Scrub Radius vs Vehicle Roll...")
                ax.plot( self.roll_angle, self.scrub_radius, color = 'k')
                ax.set_xlabel('Vehicle Roll [deg]')
            else:
                print("Plotting Scrub Radius vs Vertical Travel...")
                ax.plot(self.bump_zs, self.scrub_radius, color = 'k')
                ax.set_xlabel('Vertical Wheel Center Travel [' + self.unit + ']')
            ax.set_ylabel('Scrub Radius ['+self.unit+']')
            ax.set_title('Scrub Radius', pad = 15)
    
        if roll_center_in_roll:
            # cmap = plt.cm.get_cmap('cividis')
            cmap = plt.cm.get_cmap('turbo')
            print("Plotting Path of Roll Center as Car Rolls...")
            fig, ax = plt.subplots()
            ys = [y for [y,z] in self.roll_center]
            zs = [z for [y,z] in self.roll_center]
            norm = plt.Normalize(min(self.roll_angle), max(self.roll_angle))
            ax.scatter(ys, zs, c = self.roll_angle, cmap = cmap, norm = norm)
            points = np.array([ys, zs]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = mcoll.LineCollection(segments, array = self.roll_angle, cmap = cmap, norm = norm)
            ax.add_collection(lc)
            if all((max(zs)-min(zs) < 3,self.unit == "mm")):
                z_mid = (max(zs)+min(zs))/2
                plt.ylim(z_mid-1.5,z_mid+1.5)
            # debugging code below, ignore
            # x1,y1,z1 = self.upper_wishbone[2].hist[1]
            # y2, z2 = self.instant_center[1]
            # y3,z3 = self.contactpatch_yz[1]
            # ax.plot((y3,y2),(z3,z2))
            # x1,y1,z1 = self.upper_wishbone[2].hist[-1]
            # y2, z2 = self.instant_center[-1]
            # y3,z3 = self.contactpatch_yz[-1]
            # ax.plot((y3,y2),(z3,z2))
            ax.set_ylabel(  'Vertical Roll Center Travel [' + self.unit + ']')
            ax.set_xlabel('Horizontal Roll Center Travel [' + self.unit + ']')
            ax.set_title('Dynamic Roll Center', pad = 15)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label = 'Roll [deg]')
    
        if motion_ratio:
            fig, ax = plt.subplots()
            if motion_ratio_in_deg:
                print("Plotting Motion Ratio vs Vehicle Roll")
                ax.plot(self.roll_angle, self.mr, color = 'k')
                ax.set_xlabel('Vehicle Roll [deg]')
            else:
                ax.plot(self.bump_zs, self.mr, color = 'k')
                ax.set_xlabel('Vertical Wheel Center Travel')
                x_max = self.bump_zs[-1]
            ax.set_ylabel('Motion Ratio')
            ax.set_title('Dynamic Motion Ratio')
            # Annotation
            if self.mr[-1] > self.mr[self.steps]:
                # upwards slope
                ax.annotate("Max Motion Ratio: "+str(self.mr[-1])[:4]+"\n"+
                            "Avg Motion Ratio: "+str(np.mean(self.mr))[:4]+"\n"+
                            "Min Motion Ratio: "+str(self.mr[0])[:4],
                            (x_max,self.mr[0]), ha="right")
            else:
                ax.annotate("Max Motion Ratio: "+str(self.mr[0])[:4]+"\n"+
                            "Avg Motion Ratio: "+str(np.mean(self.mr))[:4]+"\n"+
                            "Min Motion Ratio: "+str(self.mr[-1])[:4],
                            (x_max,self.mr[0]), ha="right")
                
        plt.show()
