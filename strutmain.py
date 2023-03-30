#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:21:41 2022

@author: simon
"""
from strutsolve import *


def main():
    unit = "in"  # used in graph axis labels, not used in code (yet...)

    """ Suspension Points """
    # In form of Point([x,y,z])
    # Wheel_Center
    # Points
    wc  = Point([0,51,10])
    spi = Point([0,32,40]) # strut pickup inboard
    lai = Point([-10,20,3]) # control arm pickup inboard
    lfi = Point([10,20,3]) # control arm pickup outboard
    lo  = Point([1,50,3]) # Lower outer
    tri = Point([-5,19,4]) # tie rod inboard
    tro = Point([-5,50,4]) # tie rod outboard
    wco = Point([0,0,0]) # offset point that will get filled in automatically, ignore

    """ Suspension Setup """
    # Full jounce and rebound mark the bounds for the solver
    # if they are too large, and cannot be achieved with your linkage system
    # the code will not throw an error but will not finish solving
    full_jounce = 3
    full_rebound = -3

    """ List the points of the suspension that will move """
    # default of [lo, tro, wc] should apply to most macpherson setups
    # Moving Points
    moving_pts = [lo, tro, wc]

    """ Input the list of points that each moving point is linked to below. """
    # Each point listed in moving_points needs a list of friends.
    # The default in the code should apply to all standard macpherson strut setups.
    # This includes linking the upper and lower upright points
    # as that length cannot be allowed to change
    wc.friends = [lo, tro, wco]
    lo.friends = [lfi, lai, wc, tro, wco]
    tro.friends = [lo, wc, tri, wco]

    kin = StrutSolve(
        wheel_center=wc,
        lower_wishbone=(lfi, lai, lo),
        strut_mount=spi,
        tie_rod=(tri, tro),
        wheel_center_offset = wco,

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
    num_steps = 1000
    # happy is the error margin for the gradient descent to be considered complete
    # For some reason you get really ugly data with learning rate < 10^-4 not sure why
    happy = 10 ** -4
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
        bump_steer_in_deg=True,  # Sets y-axis of bump steer plot to roll angle in deg
        camber_gain_in_deg=False,  # Sets y-axis of camber gain plot to roll angle in deg
        caster_gain_in_deg=False  # Sets y-axis of caster gain plot to roll angle in deg
    )


if __name__ == "__main__":
    main()
