#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:21:41 2022

@author: simon
"""
from rocker_kinsolve_alpha import *


def main():

    """ Suspension Points """
    # In form of Point([x,y,z])
    # Wheel_Center
    # Y point of wc should be track width / 2
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
    
    unit = "mm"  # used in graph axis labels, not used in code (yet...)

    
    # Pushrod or Pullrod Points
    # The P-rod inner point is the outboard (usually) point of the rocker/bellcrank
    pri = Point([ -20.3792, 350  ,  487.934 ])
    pro = Point([ -13.9192, 556.8442,  124.9426])
    
    # Rocker Center of Rotation
    rkr = Point([ -23.38  , 280     ,  450])
    
    # Shock Pickup Points (upper, lower)
    # The shock upper point is the inner (usually) point of the rocker/bellcrank
    sku = Point([ -25     , 150     ,  490])
    skl = Point([ -30     , 150     ,  300])

    """ Suspension Setup """
    # Full jounce and rebound mark the bounds for the solver
    # if they are too large, and cannot be achieved with your linkage system
    # the code will not throw an error but will either not finish solving or give erroneous results
    full_jounce = 25.4
    full_rebound = -25.4
    
    # toe, camber and caster are used for static offsets on the graphs
    # these will not affect the solver
    toe = 0
    camber = 0
    caster = 0

    """ Input the list of points that each moving point is linked to below. """
    # Each point listed in moving_points needs a list of friends.
    # The default in the code should apply to all standard SLA/double wishbone setups.
    # This includes linking the upper and lower upright points
    # as that length cannot be allowed to change
    wc.friends = [uo, lo, tro]
    uo.friends = [ufi, uai, wc, tro, lo]
    lo.friends = [lfi, lai, wc, tro, uo]
    tro.friends = [uo, lo, wc, tri]
    
    kin = KinSolve(
        wheel_center=wc,
        lower_wishbone=(lfi, lai, lo),
        upper_wishbone=(ufi, uai, uo),
        tie_rod=(tri, tro),
        
        p_rod=(pri, pro),
        rocker=rkr,
        shock=(skl, sku),

        full_jounce=full_jounce,
        full_rebound=full_rebound,

        unit=unit,
    )

    """ Solver Parameters """
    # number of steps in each direction, so a value of 10 will yield 20 datapoints
    # algorithm runs fast enough that its fine to use 1000+, but 100 is just as accurate
    # and it will result in a comprehensible amount of data
    num_steps = 100
    
    kin.solve(
        steps=num_steps,
        offset_toe=toe,
        offset_camber=camber,
        offset_caster=caster
    )

    """ Plot """
    kin.plot(
        suspension=True,  # Visualize the corner
        
        bump_steer=False,  # Bump Steer vs vertical travel
        bump_steer_in_deg=False,  # Sets y-axis of bump steer plot to roll angle in deg

        camber_gain=False,  # Camber Gain vs vertical travel
        camber_gain_in_deg=False,  # Sets y-axis of camber gain plot to roll angle in deg

        caster_gain=False,  # Caster gain plot
        caster_gain_in_deg=False,  # Sets y-axis of caster gain plot to roll angle in deg

        scrub_gain = False, # Scrub change plot
        scrub_gain_in_deg = False,    # Sets y-axis of scrub gain plot to roll angle in deg

        roll_center_in_roll=True,  # Path of roll center as the car rolls
        
        motion_ratio=False, # Motion Ratio vs vertical travel
        motion_ratio_in_deg=False # Sets y-axis of motion ratio plot to roll angle in deg
    )


if __name__ == "__main__":
    main()
