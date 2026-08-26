# mga-compass

Core algorithms for real-time exploration of near-optimal solutions (Modeling to Generate Alternatives) in PyPSA energy planning models. This is the standalone library used by [MGA Compass](https://github.com/EINS-TUDa/MGA-Compass); it has no GUI/server dependencies.

Finding the near-optimal points themselves (i.e. running MGA against a PyPSA network) is the
job of the sibling [mga-surveyor](../mga-surveyor) package, not this one — `mga-compass` only
navigates and interpolates over points it's given.
