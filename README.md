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

- `PHY300_Final.py` - Contains the simulation of the collision that unfortunately only calculates up to 5 digits of pi; anything more will crash the program.
- `Phase_Space.ipynb` - Jupyter notebook that graphs all the points by running a simulation itself. Multiple cells that do the same thing.
- `assets/boop.mp3` - the sound made when objects collide.
- `assets/pieImage.jpg` - the window icon.
- `assets/phase_images/` - pre-rendered phase-space graphs shown by `PHY300_Final.py`.

