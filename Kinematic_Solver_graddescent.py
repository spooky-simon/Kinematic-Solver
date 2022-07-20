# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:21:41 2022

@author: simon
"""
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

#%% Class Definition
class Point():
    def __init__(self,coords):
        self.coords = np.array(coords) # to track coordinates of the point
        self.links = {} # a dictionary used to link points to eachother
        self.jhist = [] # jounce history
        self.rhist = [] # rebound history
        self.hist = []  # total travel history
        self.friends = [] # list of points that this point is linked to
        self.origin = np.array(coords) # to keep track of the static position

    def jr_combine(self):
        # function used to combine jounce and rebound history
        self.rhist.reverse()
        self.hist = self.rhist+self.jhist
        return self.hist

#%% User Input

# Input Suspension Points Below
# In form of Point([x,y,z])

# Wheel_Center
wc = Point([0,	622.5,	203])
# Lower Wishbone
lfi = Point([175.1,	    175,	111]) # Lower_Fore_Inner
lai = Point([-175.1,	175,	111]) # Lower_Aft_Inner
lo  = Point([-3.1,	    608,	114]) # Lower_Upright_Point
# Upper Wishbone
ufi = Point([120.1,	    240,	223]) # Upper_Fore_Inner
uai = Point([-120.1,	240,	216]) # Upper_Aft_Inner
uo  = Point([-7.1,	    595,	299]) # Upper_Upright_Point
# Tie Rod or Steering Rod
tri = Point([55.1, 140, 163]) # Tie_Rod_Inner
tro = Point([55.1, 600, 163]) # Tie_Rod_Outer

unit = "mm" # used in graph axis labels, not used in code (yet...)

# Suspension Setup
# Full jounce and rebound mark the bounds for the solver
# if they are too large, and cannot be achieved with your linkage system
# the code will not throw an error but will not finish solving
full_jounce = 25.4
full_rebound = -25.4
# toe, camber and caster are used for static offsets on the graphs
# these will not affect the solver
toe = 0
camber = 0
caster = 0

# List the points of the suspension that will move
moving_pts = [uo, lo, tro, wc]

# Input the list of points that each moving point is linked to below
# This includes linking the upper and lower upright points
# as that length cannot be allowed to change

wc.friends = [uo, lo, tro]
uo.friends = [ufi, uai, wc, tro, lo]
lo.friends = [lfi, lai, wc, tro, uo]
tro.friends = [uo, lo, wc, tri]

# Plotting
# On or Off
# put %matplotlib into the kernel to get pop-out graphs that are interactive
# if you fix my bad graphs, send me your fix pls
sus_plt = 1        # Visualize the corner
bmp_str_plt = 1     # Bump Steer vs vertical travel
bmp_str_roll = 0    # Sets y axis of bump steer plot to roll angle in deg
cbr_gn_plt = 1     # Camber Gain vs vertical travel
cbr_gn_roll = 0     # Sets y axis of camber gain plot to roll angle in deg
cstr_gn_plt = 1    # caster gain plot
cstr_gn_roll = 0
roll_center_in_roll = 1 # Path of roll center as the car rolls

# Solver Parameters
# number of steps in each direction, so a value of 10 will yield 20 datapoints
# num_steps = 5 is lightning fast, gives blocky curve
# weird stuff happens when you set it to anything between 25 and 50
# num steps = 100-1000 gives nice smooth lines
num_steps = 5
# happy is the error margin for the gradient descent to be considered complete
# For some reason you get really ugly data with learning rate < 10^-4 not sure why
happy = 10**-3
learning_rate = 10**-3
# I did not implement a dynamic learning rate because im lazy and this works


# Code Below: No Need to Touch
#%% Gradient Descent

# Derive link lengths for use in Grad Descent
for pt in moving_pts:
    for friend in pt.friends:
        link = norm(np.array(pt.coords - friend.coords))
        pt.links.update({friend: link})

# Associative Function
def Ass_Fcn(pt):
    # Gx describes how much longer each link is than it should be
    Gx = []
    for friend in pt.friends:
        v = pt.coords-friend.coords # link vector
        v_norm = norm(v)            # link length
        l = pt.links[friend]        # link target
        Gx.append(v_norm - l)
    return Gx

# Objective Function (cost fcn)
def Obj_Fcn(pt):
    Fx = []
    for friend in pt.friends:
        v = pt.coords-friend.coords # link vector
        v_norm = norm(v)            # link length
        l = pt.links[friend]        # link target
        Fx.append(v_norm - l)
    Fx = [(i**2)*0.5 for i in Fx]
    return Fx

# Jacobian of Objective Function
def Jacobian(pt):
    Jcb = []
    for friend in pt.friends:
        df = 2 * (pt.coords - friend.coords)
        Jcb.append(df)
    return np.array(Jcb)

# Solving for Jounce Kinematics
t0 = time.time_ns()
err = []
v_move = [0,0,full_jounce/num_steps]
# This is the good stuff
# I'm straight up not explaining gradient decsent in code comments
# but it loops through each step and solves the whole shibang muy bueno
for i in range(0,num_steps):
    # print(wc.coords)
    for pt in moving_pts:
        pt.jhist.append(pt.coords)
        pt.coords = pt.coords + v_move
    window = 1
    while window > happy:
        for pt in moving_pts:
            J = Jacobian(pt)
            G = Ass_Fcn(pt)
            E = Obj_Fcn(pt)
            JT = J.T
            step = learning_rate * JT @ G
            pt.coords = pt.coords - step
        err.append(sum(E))
        window = sum(E)
t1 = time.time_ns()
print("Solved Jounce in",(t1-t0)/10**6, "ms")

# Reset to do rebound
for pt in moving_pts:
    pt.coords = pt.origin
mid_pt_index = 1
err = []
v_move = [0,0,full_rebound/num_steps]

# Rebound Kinematics
t0 = time.time_ns()
for i in range(0,num_steps):
    window = 1
    for pt in moving_pts:
        pt.rhist.append(pt.coords)
        pt.coords = pt.coords + v_move
    while window > happy:
        for pt in moving_pts:
            J = Jacobian(pt)
            G = Ass_Fcn(pt)
            E = Obj_Fcn(pt)
            JT = J.T
            step = learning_rate * JT @ G
            pt.coords = pt.coords - step
        err.append(sum(E))
        window = sum(E)
t1 = time.time_ns()
print("Solved rebound in",(t1-t0)/10**6, "ms")

# Combine Jounce and Rebound into a single list
for pt in moving_pts:
    pt.jr_combine()

#%% Calculate kinematic changes over wheel travel

def angle(v1,v2):
    # angle between two vectors
    # arccos of the dot product of two unit vectors
    uv1 = v1 / norm(v1)
    uv2 = v2 / norm(v2)
    dot12 = dot(uv1,uv2)
    ang = np.degrees(np.arccos(dot12))
    return(ang)

def pt_to_ln(pt,a,b):
    # This function calculates the shortest vector from point pt to a line
    # that goes from a to b
    # The vector that is returned is point to line or ptl
    ln_ab = [a-b for a,b in zip(a,b)]
    ln_ap = [a-b for a,b in zip(a,pt)]
    ab_u = ln_ab / norm(ln_ab)
    ptl = ln_ap - dot(ln_ap,ab_u) * ab_u
    return ptl

# Bump Steer
# Projecting the steeing arm into the XY plane and measuring the angle
sa = [pt_to_ln(tro,uo,lo) for tro,uo,lo in zip(tro.hist,uo.hist,lo.hist)]
sa_xy = [[x,y] for x,y,z in sa]  # project into xy plane (top view)
bmp_str = [-angle(v,[1,0]) for v in sa_xy]  # angle bw v1 and x axis
bmp_str = [i - bmp_str[num_steps]+toe for i in bmp_str]

# Camber Gain
# Projects the kingpin into the YZ plane to meaure camber
# by measuring the angle between the kingpin and the Z axis
kp = [a-b for a,b in zip(uo.hist,lo.hist)]
kp_yz = [[y,z] for x,y,z in kp] # project into YZ plane (front view)
cbr_gn = [-angle([0,1],v) for v in kp_yz] # compare to z axis
cbr_gn = [i - cbr_gn[num_steps] +camber for i in cbr_gn] # compares to static

# Caster change
# Projects the kingpin into the YZ plane to meaure caster
# by measuring the angle between the kingpin and the Z axis
kp_xz = [[x,z] for x,y,z in kp] # project into XZ plane (side view)
cstr_gn = [-angle([0,1],v) for v in kp_xz] # compare to z axis
cstr_gn = [i - cstr_gn[num_steps] +caster for i in cstr_gn] # compares to static

# bump_zs is a list of the z height for each iterable in the code compared to static
# roll_ang is a list of the body roll of the vehicle for each iterable in the code compared to static

bump_zs = [xyz[2] - wc.origin[2] for xyz in wc.hist]
roll_ang = [np.degrees(np.sin(x/wc.origin[1])) for x in bump_zs]


# Roll center
# line intersecting functions taken from
# https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/%7Erod/2500/notes/numpy-arrays/numpy-arrays.html
# I don't really get the whole thing but it works so I don't need to think about it

def perp( a ) : # Creates a perpendicular line segement
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom)*db + b1

# Get using fore link for upper and lower wishbones
# Shouldn't change behavior significantly unless anti-dive characteristics are huge
# project to yz plane
upr = [ufi.coords[1:] for i in uo.hist]
lwr = [lfi.coords[1:] for i in lo.hist]
uo_xy = [i[1:] for i in uo.hist]
lo_xy = [i[1:] for i in lo.hist]
ic_pts = zip(upr,uo_xy,lwr, lo_xy)
ic = [seg_intersect(a1, a2, b1, b2) for a1,a2,b1,b2 in ic_pts]
fa_gd = [np.array([wc.coords[1] - wc.origin[1],0]) for i in ic] # front axle at the ground
# Roll Center in Heave
# opp variables are for opposite side
opp_ic = [np.array([-y,z]) for y,z in ic]
opp_fa_gd = [np.array([-y,z]) for y,z in fa_gd]
# rch is roll center in heave
rch_pts = zip(fa_gd,ic,opp_fa_gd,opp_ic)
rch = [seg_intersect(a1, a2, b1, b2) for a1,a2,b1,b2 in rch_pts]
# Roll Center in Roll
opp_ic_r = opp_ic
opp_ic_r.reverse()
# rcr is roll center in roll
rcr_pts = zip(fa_gd,ic,opp_fa_gd,opp_ic)
rcr = [seg_intersect(a1, a2, b1, b2) for a1,a2,b1,b2 in rcr_pts]


#%% Plotting

if sus_plt:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=18., azim=-51)
    for pt in [wc,uo,lo,lfi,lai,ufi,uai,tro,tri]:
        ax.scatter(pt.origin[0],pt.origin[1],pt.origin[2])
    for pt in moving_pts:
        xs,ys,zs = zip(*pt.hist)
        ax.plot(xs,ys,zs)
    ax.plot((lfi.origin[0],lo.origin[0]),
            (lfi.origin[1],lo.origin[1]),
            (lfi.origin[2],lo.origin[2]))
    ax.plot((lai.origin[0],lo.origin[0]),
            (lai.origin[1],lo.origin[1]),
            (lai.origin[2],lo.origin[2]))
    ax.plot((uai.origin[0],uo.origin[0]),
            (uai.origin[1],uo.origin[1]),
            (uai.origin[2],uo.origin[2]))
    ax.plot((ufi.origin[0],uo.origin[0]),
            (ufi.origin[1],uo.origin[1]),
            (ufi.origin[2],uo.origin[2]))
    ax.plot((tri.origin[0],tro.origin[0]),
            (tri.origin[1],tro.origin[1]),
            (tri.origin[2],tro.origin[2]))
    ax.plot((lo.origin[0],uo.origin[0]),
            (lo.origin[1],uo.origin[1]),
            (lo.origin[2],uo.origin[2]))
    x1 = [xyz[0] for xyz in tro.hist]
    y1 = [xyz[1] for xyz in tro.hist]
    z1 = [xyz[2] for xyz in tro.hist]
    sax = [xyz[0] for xyz in sa]
    say = [xyz[1] for xyz in sa]
    saz = [xyz[2] for xyz in sa]
    # Code Below Shows Steering Arm change at full jounce and rebound
    # x2 = [a+b for a,b in zip(x1,sax)]
    # y2 = [a+b for a,b in zip(y1,say)]
    # z2 = [a+b for a,b in zip(z1,saz)]
    # for i in [0, num_steps, 2*num_steps-1]:
    #     ax.plot((x1[i],x2[i]),(y1[i],y2[i]),(z1[i],z2[i]))


if cbr_gn_plt:
    fig, ax = plt.subplots()
    if cbr_gn_roll:
        ax.plot(cbr_gn,roll_ang)
        ax.set_ylabel('Vehicle Roll [deg]')
    else:
        ax.plot(cbr_gn,bump_zs)
    ax.set_ylabel('Vertical Wheel Center Travel ['+unit+']')
    ax.set_xlabel('Camber Change [deg]')
    ax.set_title('Camber Gain')

if bmp_str_plt:
    fig, ax = plt.subplots()
    if bmp_str_roll:
        ax.plot(bmp_str,roll_ang)
        ax.set_xlabel('Vehicle Roll [deg]')
    else:
        ax.plot(bmp_str,bump_zs)
    ax.set_ylabel('Vertical Wheel Center Travel ['+unit+']')
    ax.set_xlabel('Toe Change [deg]')
    ax.set_title('Bump Steer')

if cstr_gn_plt:
    fig, ax = plt.subplots()
    if cstr_gn_roll:
        ax.plot(cstr_gn,roll_ang)
        ax.set_xlabel('Vehicle Roll [deg]')
    else:
        ax.plot(cstr_gn,bump_zs)
    ax.set_ylabel('Vertical Wheel Center Travel ['+unit+']')
    ax.set_xlabel('Caster Change [deg]')
    ax.set_title('Caster Gain')

if roll_center_in_roll:
    fig, ax = plt.subplots()
    xs = [xyz[0] for xyz in rcr]
    ys = [xyz[1] for xyz in rcr]
    ax.plot(xs,ys)
    ax.set_ylabel('Vertical Roll Center Travel ['+unit+']')
    ax.set_xlabel('Horizontal Roll Center Travel ['+unit+']')
    ax.set_title('Dynamic Roll Center')

plt.show()
