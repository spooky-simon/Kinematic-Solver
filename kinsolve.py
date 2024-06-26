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
import scipy
from scipy.sparse import coo_array

class Point:
    def __init__(self, coords):
        """
        A "Point" is used to keep track of the location history for all moving points
        """
        self.hist = []  # total travel history
        self.origin = np.array(coords)  # to keep track of the static position

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

def intersection_of_spheres(center_1,center_2,intersection_pt):
    """
    Intersection of two spheres if an intersection point is known.
    """
    
    # Distance between sphere centers
    # Radii of spheres
    # Fraction along line between centers that the circle sits on
    d  = norm(center_1 - center_2)
    r1 = norm(center_1 - intersection_pt)
    r2 = norm(center_2 - intersection_pt)
    h = 0.5 + (r1*r1 - r2*r2) / (2 * d*d)
    
    # Radius of circle of intersection
    r_i = np.sqrt(r1*r1 - h*h * d*d)
    
    # Normal vector
    n_i = (center_2 - center_1)/d
    
    # Center of circle of intersection
    c_i = center_1 + h * (center_2 - center_1)
    
    return n_i, c_i, r_i

def intersection_of_spheres_radii(center_1,center_2, r1, r2):
    """
    Intersection of two spheres if an intersection point is not known,
    but the radii to the circle of intersection is known.
    """
    
    # Distance between sphere centers
    # Fraction along line between centers that the circle sits on
    d  = norm(center_1 - center_2)
    h = 0.5 + (r1*r1 - r2*r2) / (2 * d*d)
    
    # Radius of circle of intersection
    r_i = np.sqrt(r1*r1 - h*h * d*d)
    
    # Normal vector
    n_i = (center_2 - center_1)/d
    
    # Center of circle of intersection
    c_i = center_1 + h * (center_2 - center_1)
    
    return n_i, c_i, r_i

def intersection_sphere_circle(c_s,r_s,n_i,c_i,r_i):
    dp = np.dot(n_i, c_i - c_s) # distance of plane to sphere center
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
    return p_0, p_1


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
    bump_steer: List[float] = None
    camber_gain: List[float] = None
    caster_gain: List[float] = None
    roll_angle = None
    bump_zs = None
    roll_center = None
    instant_center = None

    def solve(self,
              steps: int = 5,
              offset_toe: float = 0,
              offset_camber: float = 0,
              offset_caster: float = 0,
              ):
        """
        Solves stuff

        :param steps: Number of steps in each direction. (e.g. 10 -> 20 datapoints)
        :param offset_toe: Static offset
        :param offset_camber: Static offset
        :param offset_caster: Static offset
        """
        
        t0 = time.time_ns()
        
        # Step 0: Initialize step size, linkage lengths, and lower a-arm array
        v_move = self.full_jounce/steps # mm
        self.lower_wishbone[2].hist = [self.lower_wishbone[2].origin]

        uprt_ht = norm(self.upper_wishbone[2].origin- self.lower_wishbone[2].origin)
        u_t_d = norm(self.upper_wishbone[2].origin - self.tie_rod[1].origin)
        l_t_d = norm(self.lower_wishbone[2].origin - self.tie_rod[1].origin)
        tr_d = norm(self.tie_rod[0].origin - self.tie_rod[1].origin)
        u_wc = norm(self.upper_wishbone[2].origin - self.wheel_center.origin)
        l_wc = norm(self.lower_wishbone[2].origin - self.wheel_center.origin)
        str_arm = norm(self.tie_rod[1].origin - self.wheel_center.origin)



        # Step 1: Find the arcs traced out by the upper and lower outboard pickup points.
        # All orientations of the suspension linkages MUST have the upper and lower 
        # outboard pickup points along their respective arcs.
        # These arcs are defined by a normal axis, a center point, and a radius.
        n_u, c_u, r_u = intersection_of_spheres(self.upper_wishbone[0].origin, self.upper_wishbone[1].origin, self.upper_wishbone[2].origin)
        n_l, c_l, r_l = intersection_of_spheres(self.lower_wishbone[0].origin, self.lower_wishbone[1].origin, self.lower_wishbone[2].origin)

        # Step 2: We eventually want to look at wheel center movement. Unfortunately the 
        # wheel center location is found by first finding all of the other points first. 
        # So in order to increment the wheel up and down to sample an orientation, we
        # can only move a point that is defined by suspension points that do not 
        # articulate. The only possible points are the lower or upper outboard pickup points.
        # I have chosen to move the lower point and all other points will follow. This is
        # a "good enough" moment where the suspension is likely linear enough for this 
        # to request an articulation through full rebound and jounce.
        # To do this, we want to find points that are a certain distance (v_move) away
        # from the static lower outer point. This means finding the intersection of a
        # sphere of radius v_move centered around the lower outer point and the arc
        # traced out by the lower outer point. This yields 2 points, one will be in jounce,
        # the other will be in rebound.

        for i in range(1, steps+1):
            
            # Generate lower arm movement
            p0,p1 = intersection_sphere_circle (self.lower_wishbone[2].origin, v_move * i, n_l, c_l, r_l)
            
            # if p1 is the jounce point (higher in z)
            # put it on the end and p0 on the beginning
            if p1[2] > p0[2]:
                self.lower_wishbone[2].hist = [p0] + self.lower_wishbone[2].hist + [p1]
            else:
                self.lower_wishbone[2].hist = [p1] + self.lower_wishbone[2].hist + [p0]

        # Step 3: To find the upper point at each sample point, we need to find the 
        # intersection of a sphere centered at the lower outboard point with radius 
        # equal to the upright height and the arc traced out by the upper outboard point

        for pt in self.lower_wishbone[2].hist:
            
            # find both points of intersection
            p2, p3 = intersection_sphere_circle(pt, uprt_ht, n_u, c_u, r_u)
            
            # since we know the upper point is always above the lower point,
            # we can discard the lower point
            if p3[2] > p2[2]:
                self.upper_wishbone[2].hist.append(p3)
            else:
                self.upper_wishbone[2].hist.append(p2)

        # Step 4: Find the steering arm point / tie rod outer point using our spheres. 
        # The tie rod outer point must lie along a circle that is equidistant from both 
        # the upper and lower pick-up points. To find the point on this circle, we use a sphere
        # with radius equal to the tie rod length to intersect the circle

        # Step 5: (In the same for loop) Find the wheel center
        # Now we have three points at each articulation step. Using methods similar to
        # the previous steps, we find the wheel center.

        for lp, up in zip (self.lower_wishbone[2].hist, self.upper_wishbone[2].hist):
            
            # tie rod circle
            n_t, c_t, r_t = intersection_of_spheres_radii(lp, up, l_t_d, u_t_d)
            
            # intersection with tie rod length    
            p4, p5 = intersection_sphere_circle(self.tie_rod[0].origin, tr_d, n_t, c_t, r_t)
            
            # one point will result in non-possible wheel position
            # the point that is further inboard is almost (hopefully) guaranteed to be wrong
            if p4[1] > p5[1]:
                self.tie_rod[1].hist.append(p4)
            else:
                self.tie_rod[1].hist.append(p5)
            
            # wheel_center circle
            n_w, c_w, r_w = intersection_of_spheres_radii(lp, up, l_wc, u_wc)
            # intersect with steer pt
            p6, p7 = intersection_sphere_circle(self.tie_rod[1].hist[-1], str_arm, n_w, c_w, r_w)
            
            # one of these will be non-possible -wheel flipped around
            if p7[1] > p6[1]:
                self.wheel_center.hist.append(p7)
            else:
                self.wheel_center.hist.append(p6)
                
            
        t1 = time.time_ns()
        print("Solved everything in", (t1 - t0) / 10 ** 6, "ms")
        
        # Now we have all of our suspension kinematics
        # We can use these to do math and figure out kinematic behavior
        print("Calculating kinematic changes over wheel travel:")
        
        # Bump steer is first up
        print("* Bump Steer")
        
        # Find vertical XY projection of axle/hub
        # Find angular change of that projection from static
        # Axle in static should be y axis + static toe
        # The axle will rotate the same amount as the line from the wc to the king pin
        # so I actually just have to find that
        axl_static = pt_to_ln(self.wheel_center.origin, self.lower_wishbone[2].origin, self.upper_wishbone[2].origin)[:2]
        axl_hist = [pt_to_ln(pt,a,b)[:2] for pt,a,b in zip(self.wheel_center.hist,
                                                       self.lower_wishbone[2].hist,
                                                       self.upper_wishbone[2].hist)]
        # My angle function is only positive so i have to measure angle from an axis
        # and subtract the static angle
        static_ang = angle([0,1],axl_static)
        bmp_str = [angle([0,1],v)-static_ang for v in axl_hist] # angle magnitude
        
        
        
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
        ui_mid = (self.upper_wishbone[0].origin+self.upper_wishbone[1].origin)/2
        upr = [ui_mid[1:] for i in self.upper_wishbone[2].hist]
        # print(upr[-1])
        # upr2 = ui_mid[1:]
        lwr = [self.lower_wishbone[0].origin[1:] for i in self.lower_wishbone[2].hist]
        # print(lwr[-1])
        uo_yz = [i[1:] for i in self.upper_wishbone[2].hist]
        lo_yz = [i[1:] for i in self.lower_wishbone[2].hist]
        ic_pts = zip(upr, uo_yz, lwr, lo_yz)
        ic = [seg_intersect(a1, a2, b1, b2) for a1, a2, b1, b2 in ic_pts]
        # print(ic[-1])
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
        # Find kp intersect by going along kingpin the same distance as static 
        # distance
        kp_v = [a - b for a,b in zip(self.upper_wishbone[2].hist, self.lower_wishbone[2].hist)] # vector
        kp_m = norm(self.upper_wishbone[2].origin - self.lower_wishbone[2].origin) # magnitude
        kp_n = [v/kp_m for v in kp_v] # normal vector
        
        # get static length fron line-plane intersection from wikipedia
        # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        p0 = np.array([0,0,0]) # point on plane
        l0 = self.lower_wishbone[2].origin # point on line
        n = np.array([0,0,1]) # ground plane normal line
        d = dot((p0 - l0),n)/dot(kp_n[steps],n)
        kp_intersect_static = l0 + kp_n[steps] * d
        kpinter_static = norm(self.lower_wishbone[2].origin - kp_intersect_static)
        self.kpinter = [pt - n * kpinter_static for pt,n in zip(self.lower_wishbone[2].hist, kp_n)]
        # the scrub radius is just the delta between the y component
        # of the wheel center and y component of the contact patch
        bump_zs = [z - self.wheel_center.origin[2] for x,y,z in self.wheel_center.hist]
        self.cp_new = [[wc[0],wc[1],z] for wc, z in zip(self.wheel_center.hist, bump_zs)]
        sr = [cp[1] - kp[1] for cp,kp in zip(self.cp_new, self.kpinter)]
        
        # bump_zs is a list of the z height for each iterable in the code compared to static
        # roll_ang is a list of the body roll of the vehicle for each iterable in the code compared to static
        # bump_zs = [z - self.wheel_center.origin[2] for x,y,z in self.wheel_center.hist]
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

        (n_i,c_i,r_i) = intersection_of_spheres(c1, c2, self.p_rod[1].origin)
            
        def higher_pt(pt_0,pt_1):
            if pt_1[2] > pt_0[2]:
                return pt_1
            return pt_0

        for c_s in self.lower_wishbone[2].hist:
            r_s = norm(self.lower_wishbone[2].origin-self.p_rod[1].origin)
            (p1,p2) = intersection_sphere_circle(c_s, r_s, n_i, c_i, r_i)
            p = higher_pt(p1,p2)
            self.p_rod[1].hist.append(p)

        # Circle of Rocker
        c_i = self.rocker.origin
        r_i = norm(self.rocker.origin - self.p_rod[0].origin)
        n_i = np.cross((self.rocker.origin - self.p_rod[0].origin),(self.rocker.origin - self.shock[0].origin))/norm(
            np.cross((self.rocker.origin - self.p_rod[0].origin),(self.rocker.origin - self.shock[0].origin)))
        
        for c_s in self.p_rod[1].hist:
            r_s = norm(self.p_rod[0].origin-self.p_rod[1].origin)
            (p1,p2) = intersection_sphere_circle(c_s, r_s, n_i, c_i, r_i)
            p = higher_pt(p1,p2)
            self.p_rod[0].hist.append(p)

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
        
        # def rotation_about_axis(pt,axis_a,axis_b,theta):
        def rotation_about_axis(pt,ctr,u_axis,theta):
            cs = np.cos(theta)
            sn = np.sin(theta)
            
            a = (pt-ctr)/norm(pt-ctr)
            b = np.cross(u_axis,a)
            a_r = cs*a + sn*b

            rot_pt = ctr+a_r*norm(pt-ctr)
            
            return rot_pt

        self.shock[1].hist.append(self.shock[1].origin)

        # Rebound angles are stored counting from full rebound in rkr _ang_r for some reason
        # not going to try to understand what I did or why a few years ago but it's easy enough
        # to just chug through the array backwards
        for theta in rkr_ang_r[::-1]:
            pt = rotation_about_axis(self.shock[1].hist[0], c_i, -n_i, np.radians(theta))
            self.shock[1].hist.insert(0, pt)

        for theta in rkr_ang_j:
            pt = rotation_about_axis(self.shock[1].hist[-1], c_i, n_i, np.radians(theta))
            self.shock[1].hist.append(pt)

        # find deltas in wheel travel and shock travel
        wc_z = [z for x,y,z in self.wheel_center.hist]
        self.wheel_travel = np.diff(wc_z)
        # wheel_travel is 1 value shorter than wc_z
        self.shock_len = [norm(a-self.shock[0].origin) for a in self.shock[1].hist]
        self.shock_travel = [l for l in np.diff(self.shock_len)]
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
        # and none of them would be the true value at any sampled point.
        # So I will do another round of "good enough" kinematics, where the 2
        # end points will just be the extremes of the "200" motion ratio values
        # and the other middle ones will be an average of neighboring pairs of 
        # the 200 pts which should give us 2 + 199 = 201
        # and the ones at the end will just be a little off
        avg_mr = [a/b for a,b in zip(self.shock_travel,self.wheel_travel) if b > 0]
        self.mr = [avg_mr[0]] + [(mr1+mr2)/2 for mr1,mr2 in zip(avg_mr[:-1],avg_mr[1:])] + [avg_mr[-1]]
                
        # Save calculated values
        self.steps = steps
        self.camber_gain = cbr_gn
        self.caster_gain = cstr_gn
        self.roll_angle = roll_ang
        self.bump_zs = bump_zs
        self.bump_steer = bmp_str
        self.roll_center = rcr
        self.instant_center = ic
        self.contactpatch_yz = cp_yz
        self.scrub_radius = sr
        # self.moving_points = moving_points
        return  (self.wheel_center, self.lower_wishbone, self.upper_wishbone,
                 self.p_rod, self.tie_rod,
                 self.camber_gain, self.caster_gain, self.roll_angle, self.bump_zs,
                 self.bump_steer, self.roll_center, self.instant_center, self.scrub_radius)
                # self.moving_points

    def linkforce(self, Fx = 0, Fy = 0, Fz = 0, pneumatic_trail = 0):
        
        # https://fswiki.us/Suspension_Forces
        # I'm gonna do some funky moves here so hold on
        
        # Step 0: find vectors, norms, and normal vectors for each of the 6 links
        # norms
        lenLF = norm(self.lower_wishbone[0].origin - self.lower_wishbone[2].origin)
        lenLA = norm(self.lower_wishbone[1].origin - self.lower_wishbone[2].origin)
        lenUF = norm(self.upper_wishbone[0].origin - self.upper_wishbone[2].origin)
        lenUA = norm(self.upper_wishbone[1].origin - self.upper_wishbone[2].origin)
        lenTR = norm(self.tie_rod[0].origin - self.tie_rod[1].origin)
        lenPR = norm(self.p_rod[0].origin - self.p_rod[1].origin)
        
        # normal vectors
        nLF = np.array([(self.lower_wishbone[0].origin - pt)/lenLF for pt in self.lower_wishbone[2].hist])
        nLA = np.array([(self.lower_wishbone[1].origin - pt)/lenLA for pt in self.lower_wishbone[2].hist])
        nUF = np.array([(self.upper_wishbone[0].origin - pt)/lenUF for pt in self.upper_wishbone[2].hist])
        nUA = np.array([(self.upper_wishbone[1].origin - pt)/lenUA for pt in self.upper_wishbone[2].hist])
        nTR = np.array([(self.tie_rod[0].origin - pt)/lenTR for pt in self.tie_rod[1].hist])
        nPR = np.array([(self.p_rod[0].origin - pt)/lenPR for pt in self.p_rod[1].hist])
        
        # Step 1: find all r vectors for our r x F equations. Since all the 
        # forced are centered at the wheel center, this is the vector from 
        # the wheel center to the 4 upright pickup points
        r_u = np.array([pt - self.wheel_center.origin for pt in self.upper_wishbone[2].hist])
        r_l = np.array([pt - self.wheel_center.origin for pt in self.lower_wishbone[2].hist])
        r_tr = np.array([pt - self.wheel_center.origin for pt in self.tie_rod[1].hist])
        r_pr = np.array([pt - self.wheel_center.origin for pt in self.p_rod[1].hist])
        
        # Step 2: we need a "unit moment" which instead of r x F we have r x n (unit vector)
        # This is weird and i dont understand it enough to explain it, but the 
        # matrix math we will do will end up scaling these by the correct magnitude
        mLF = np.array([np.cross(a,b) for a,b in zip(r_l, nLF)])
        mLA = np.array([np.cross(a,b) for a,b in zip(r_l, nLA)])
        mUF = np.array([np.cross(a,b) for a,b in zip(r_u, nUF)])
        mUA = np.array([np.cross(a,b) for a,b in zip(r_u, nUA)])
        mTR = np.array([np.cross(a,b) for a,b in zip(r_tr, nTR)])
        mPR = np.array([np.cross(a,b) for a,b in zip(r_pr, nPR)])
        
        A1 = np.hstack((np.stack((nLF, nLA, nUF, nUA, nTR, nPR), axis = 2),
                        np.stack((mLF, mLA, mUF, mUA, mTR, mPR), axis = 2)))
        # print(nLF)
        q = np.stack((nLF, nLA, nUF, nUA, nTR, nPR), axis = 2)
        r = np.stack((mLF, mLA, mUF, mUA, mTR, mPR), axis = 2)
        s = np.vstack((q,r))
        # print(s.shape)
        # print(A1.shape)
        # print(mLF)
        # print(r[:,1,:])
        # second copy if we need it
        A2 = np.hstack((np.stack((nLF, nLA, nUF, nUA, nTR, nPR), axis = 2),
                       np.stack((mLF, mLA, mUF, mUA, mTR, mPR), axis = 2)))
        
        # Step 3: find moments about each axis
        # This will be self aligning torque, overturning moment, and any drive/briking
        # torque
        # Step 3a: find moment arms for each force
        # mech_trail = where king pin intersects groud plane to cp in x
        mech_trail = [cp[0] - kp[0] for cp,kp in zip(self.cp_new,self.kpinter)]
        trail = [mt + pneumatic_trail for mt in mech_trail] # add pnuematic trail    
        
        # Step 3b: get moments
        # We have to get the Moments about each axis
        # We will analyze moments about wheel center
        # Moment about x is Fz * distance from contact patch to wheel center in y
        # which should be 0 (in theory) + Fy * distance from contact patch to 
        # wheel center in z
        # Moment about y should be Fx * distance from contact patch to wc in z
        # which is 0 unless there is an acceleration or braking load + Fz * 
        # distance from contact patch to wc in x (which is 0)
        # Moment about z is Fx * distance from cp to wc in y (0) (self aligning torque "ish")
        # print(A1.shape)
        Mx = [Fy * self.wheel_center.origin[2]] * (2 * self.steps + 1)
        My = [-Fx * self.wheel_center.origin[2]] * (2 * self.steps + 1)
        Mz = [-Fx * sr + Fy * mt for sr, mt in zip(self.scrub_radius, mech_trail)]
        
        B1 = np.array([[Fx, Fy, Fz, mx, my, mz] for mx,my,mz in zip(Mx,My,Mz)]).flatten()
        # B1 = np.array([[Fx, Fy, Fz, mx, my, mz] for mx,my,mz in zip(Mx,My,Mz)])
        # print(B1)
        
        # Step 4: Change A1 matrix to giant sparse 2d matrix so we can just compute them all at one time
        # Instead of computing one by one
        m,n,p = A2.shape
        # Step 4a: get index matrix that will map rows of each 6x6 slice of A1 to 
        # a row in a sparse matrix, so we need a list that goes from 0 to 2*steps+1
        # repeating each digit so the list is 6* longer
        I = np.repeat([range(m*n)],p)
        # Step 4b: we need column indicies of the sparse matrix so we need 
        # a matrix that repeated countes up to 6 higher 6 times then jumps by 6
        J = np.tile(np.reshape(range(m*n),(m,n)),p).flatten()
        # Now turn it into a huge sparse diagonal block matrix where the 
        # diagonal is made up of 6x6 matrices. This lets us solve every iteration
        # in one solve step
        M = coo_array((A1.flatten(), (I,J)))

        x = scipy.sparse.linalg.spsolve(M,B1)
        # print(x.shape)
        # print([x[i] for i in range(1206) if i%6 ==0 ])
        # print(x[:6])
        print(("LF: {}\n"+
               "LU: {}\n"+
               "UF: {}\n"+
               "UA: {}\n"+
               "TR: {}\n"+
               "PR: {}\n").format(*[str(f) for f in x[:6]]))
        
        print("Max link force is:", max(x))
        print("Min link force is:", min(x))
        
        # Debugging - x should sum to Fx, y should sum to Fy, z should sum to Fz
        # print(sum([z for x,y,z in [(nLF, nLA, nUF, nUA, nTR, nPR)[i%6][0] * x[i] for i in range(12,18)]]))
        
        # Plot of link forces for debugging
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.plot([x[i] for i in range(1206) if i%6 ==0])
        # ax.plot([x[i] for i in range(1206) if i%6 ==1])
        # ax.plot([x[i] for i in range(1206) if i%6 ==2])
        # ax.plot([x[i] for i in range(1206) if i%6 ==3])
        # ax.plot([x[i] for i in range(1206) if i%6 ==4])
        # ax.plot([x[i] for i in range(1206) if i%6 ==5])
        
        
        
        
        
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
            for pt in [self.upper_wishbone[2].hist, self.lower_wishbone[2].hist, self.tie_rod[1].hist, self.wheel_center.hist, self.p_rod[1].hist]:
                xs, ys, zs = zip(*pt)
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
                            (x_max,self.mr[0]),
                            ha="right", va="bottom")
            else:
                ax.annotate("Max Motion Ratio: "+str(self.mr[0])[:4]+"\n"+
                            "Avg Motion Ratio: "+str(np.mean(self.mr))[:4]+"\n"+
                            "Min Motion Ratio: "+str(self.mr[-1])[:4],
                            (x_max,self.mr[0]),
                            ha="right", va="top")
                
        plt.show()
