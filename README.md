# Kinematic-Solver

* free to use, just throw me a mention somwhere
* if you figure out a bug pls lmk
* pair of scripts each for double wishbone and macpherson strut
  * macpherson stuff is prefixed w/ strut
* See Naming_Convention for double wishbone points naming convention used in this script
* code should be somewhat commented... algorithm partially stolen from wikipedia
* approach taken from Spinning Thumbs on yt
* working on a link force solver, check back soon

## How to use
#### Double Wishbone
* just download kinsolve.py and main.py into the same folder
* open main.py and input your parameters
* run main.py and enjoy:)
#### MacPherson Strut
* just download strutsolve.py and strutmain.py into the same folder
* open strutmain.py and input your parameters
* run strutmain.py and then send me what errors you get

## Known bugs
* weird stuff can be seen in the caster/camber/toe graphs when you set num_steps to anything between 25 and 100.
  * I dont know why this happens, and i probably won't fix it. just set it to a bigger or smaller value and it goes away
* Macpherson Strut Roll Center may not be as accurate as I want, no verification for the moment so who knows
* Docs and doc website are out of date, Brian made those at some point and they haven't kept up with code changes and comment additions
