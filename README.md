# Calculating Pi Through Elastic Collisions
This is a final project for my computational physics class in the Fall Semester of 2024. My partner and I took inspiration from 3Blue1Brown's video "The Most Unexpected Answer to a counting puzzle," which explores how two objects colliding with each other in an elastic setting can calculate pi. Using the conservation of energy and momentum formula, we were able to derive the velocity and dynamically plot the point using Matplotlib to showcase a circle after our collisions. We did everything from scratch but the graphics. Although this method is highly inefficient, it is a fun project. 

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python PHY300_Final.py
```

## Files

- `PHY300_Final.py` - The pygame visualization. Set `n_digits` near the top of the file.
- `pi_collisions.py` - The physics engine. It is event-driven: rather than stepping time forward in tiny increments and checking for overlaps (which let the fast small ball tunnel through the big one and lose collisions from n = 4 up), it solves for the exact moment of each next collision and jumps straight to it. Compiled with numba, it counts 31,415,926 collisions (n = 8) in under a tenth of a second. Run `python pi_collisions.py` for a timing table.
- `tests/` - Checks that the collision counts match the digits of pi, that energy is conserved, and that the balls never pass through each other. Run with `python -m pytest`.
- `Phase_Space.ipynb` - Jupyter notebook that graphs all the points by running a simulation itself. Multiple cells that do the same thing.
- `assets/boop.mp3` - the sound made when objects collide.
- `assets/pieImage.jpg` - the window icon.
- `assets/phase_images/` - pre-rendered phase-space graphs shown by `PHY300_Final.py`.

