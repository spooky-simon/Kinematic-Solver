# Kinematic-Solver

* free to use, just throw me a mention somwhere
* Limited to pure heave and pure roll (for now)
* if you figure out a bug pls lmk
* pair of scripts each for double wishbone and macpherson strut
  * macpherson stuff is prefixed w/ strut
* See Naming_Convention for double wishbone points naming convention used in this script
* code should be somewhat commented... grad descent algorithm partially stolen from wikipedia
* approach taken from Spinning Thumbs on yt

## How to use
#### Double Wishbone
* just download kinsolve.py and main.py into the same folder
* open main.py and input your parameters
* run main.py and enjoy:)

#### Double Wishbone (with rocker)
* Download rocker_alpha and kinsolve_rocker_alpha into same folder
* Use as above
	* Be wary
	* Currently unvalidated

#### MacPherson Strut
* just download strutsolve.py and strutmain.py into the same folder
* open strutmain.py and input your parameters
* run strutmain.py and then send me what errors you get
* Slower and less accurate due to eariler version of the grad descent algo.
	* Probably gonna stay that way because it's good enough

## Assumptions Made by the Code
* Corner sees pure heave or roll
* In roll-based analysis, the vertical travel of the side input into the vehicle is equaled in the opposite direction on the opposite side of the car
* Suspension components are infinitely stiff and there is no compliance

## Future additions
* working on a link force solver, check back... at some point in the future
* ~~rocker geometry and dynamic motion ratios should be coming~~ Currently in an early unvalidated state
* Report Generator to help organize all the information
* Analytical solver instead of grad descent (preliminary trials show ~x20 speed increase)

## Known bugs
* Roll center is off by a few mm. This is due to tolerance stackup in the calculations. I don't think its worth chasing this down, as the behavior is correct, and the magintude is only off by a few mm
* ~~weird stuff can be seen in the caster/camber/toe graphs when you set num_steps to anything between 25 and 100.~~
  * ~~I dont know why this happens, and i probably won't fix it. just set it to a bigger or smaller value and it goes away~~
  * fixed with better grad descent algo
* ~~Macpherson Strut Roll Center may not be as accurate as I want, no verification for the moment so who knows~~
	* strut math all works out p well except the roll center is all wrong and the algo spits out junk sometimes
* Docs and doc website are out of date, Brian made those at some point and they haven't kept up with code changes and comment additions
