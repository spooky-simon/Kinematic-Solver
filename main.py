#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:21:41 2022

@author: simon
"""
from kinsolve import *


def main():
    unit = "mm"  # used in graph axis labels, not used in code (yet...)

    """ Suspension Points """
    # In form of Point([x,y,z])
    # Wheel_Center
    wc = Point([0, 622.5, 203])
    # Lower Wishbone
    lfi = Point([175.1, 175, 111])  # Lower_Fore_Inner
    lai = Point([-175.1, 175, 111])  # Lower_Aft_Inner
    lo = Point([3.1, 608, 114])  # Lower_Upright_Point
    # Upper Wishbone
    ufi = Point([120.1, 240, 223])  # Upper_Fore_Inner
    uai = Point([-120.1, 240, 216])  # Upper_Aft_Inner
    uo = Point([-1.1, 595, 299])  # Upper_Upright_Point
    # Tie Rod or Steering Rod
    tri = Point([55.1, 140, 163])  # Tie_Rod_Inner
    tro = Point([55.1, 600, 163])  # Tie_Rod_Outer

    """ Suspension Setup """
    # Full jounce and rebound mark the bounds for the solver
    # if they are too large, and cannot be achieved with your linkage system
    # the code will not throw an error but will not finish solving
    full_jounce = 25.4
    full_rebound = -25.4

    """ List the points of the suspension that will move """
    moving_pts = [uo, lo, tro, wc]

    """ Input the list of points that each moving point is linked to below """
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

    # toe, camber and caster are used for static offsets on the graphs
    # these will not affect the solver
    toe = 0
    camber = 0
    caster = 0

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
        bump_steer=False,  # Bump Steer vs vertical travel
        camber_gain=False,  # Camber Gain vs vertical travel
        caster_gain=False,  # Caster gain plot
        roll_center_in_roll=True,  # Path of roll center as the car rolls
        bump_steer_in_deg=False,  # Sets y-axis of bump steer plot to roll angle in deg
        camber_gain_in_deg=False,  # Sets y-axis of camber gain plot to roll angle in deg
        caster_gain_in_deg=False  # Sets y-axis of caster gain plot to roll angle in deg
    )


if __name__ == "__main__":
    main()
