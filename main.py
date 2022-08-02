#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:21:41 2022

@author: simon
"""
from kinsolve import *


def main():

    """ Suspension Points """
    # In form of Point([x,y,z])
    # Wheel_Center
    # Y point of wc should be track width / 2
    wc = Point([0, 1245/2, 203])
    # Lower Wishbone
    lfi = Point([175.1, 175, 111.45])  # Lower_Fore_Inner
    lai = Point([-175.1, 175, 111.45])  # Lower_Aft_Inner
    lo = Point([18.15, 610, 114.8])  # Lower_Upright_Point
    # Upper Wishbone
    ufi = Point([120.1, 240, 223])  # Upper_Fore_Inner
    uai = Point([-120.1, 240, 215.8])  # Upper_Aft_Inner
    uo = Point([5.25, 600.5, 298.3])  # Upper_Upright_Point
    # Tie Rod or Steering Rod
    tri = Point([55.1, 140, 163])  # Tie_Rod_Inner
    tro = Point([55.1, 600, 163])  # Tie_Rod_Outer
    
    unit = "mm"  # used in graph axis labels, not used in code (yet...)

    
    # Push Rod or Pull Rod
    # Not used for kinematics, only used for link-force solving
    # when I finish and publish it, you can ignore for now
    # pri = Point([55.1, 140, 163])  # P_Rod_Inner
    # pro = Point([55.1, 600, 163])  # P_Rod_Outer

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

    """ List the points of the suspension that will move """
    # default of [uo, lo, tro, wc] should apply to most double wishbone setups
    moving_pts = [uo, lo, tro, wc]

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

        full_jounce=full_jounce,
        full_rebound=full_rebound,

        moving_points=moving_pts,

        unit=unit,
    )

    """ Solver Parameters """
    # number of steps in each direction, so a value of 10 will yield 20 datapoints
    # num_steps = 5 is lightning fast, gives blocky curve
    # I have found 10 to be a good middle ground
    # Weird stuff happens when you set it to anything between 25 and 50
    # num steps = 100-1000 gives nice smooth lines, but takes like half a second longer
    num_steps = 10
    # happy is the error margin for the gradient descent to be considered complete
    # For some reason you get really ugly data with learning rate < 10^-4 not sure why
    happy = 10 ** -3
    learning_rate = 10 ** -3
    # I did not implement a dynamic learning rate because im lazy and this works
    
    kin.solve(
        steps=num_steps,
        happy=happy,
        learning_rate=learning_rate,
        offset_toe=toe,
        offset_camber=camber,
        offset_caster=caster
    )

    """ Plot """
    kin.plot(
        suspension=False,  # Visualize the corner
        bump_steer=True,  # Bump Steer vs vertical travel
        camber_gain=False,  # Camber Gain vs vertical travel
        caster_gain=False,  # Caster gain plot
        scrub_gain = True, # Scrub change plot
        roll_center_in_roll=False,  # Path of roll center as the car rolls
        bump_steer_in_deg=False,  # Sets y-axis of bump steer plot to roll angle in deg
        camber_gain_in_deg=True,  # Sets y-axis of camber gain plot to roll angle in deg
        caster_gain_in_deg=False,  # Sets y-axis of caster gain plot to roll angle in deg
        scrub_gain_in_deg = True    # Sets y-axis of scrub gain plot to roll angle in deg
    )


if __name__ == "__main__":
    main()
