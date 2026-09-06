"""Ray-traced micro-Doppler with NVIDIA Sionna RT.

sim/drone.py models a blade as a line of isotropic scatterers and sums phase
analytically. That gets the kinematics right -- tip Doppler, HERM spacing,
the cos(elevation) null -- but knows nothing about blade twist, chord, the
body shadowing a blade on the far side, the metal motor cans, or the floor
bounce. This package puts real geometry into a physically based ray tracer
and lets those effects fall out.

The trick that makes it affordable: a rotor's scattering is a function of its
angle alone (radar fixed, body fixed). So trace ONE rotor revolution finely,
once, and then drive that lookup with any RPM profile -- four independent
rotors, jitter, throttle changes -- at no further tracing cost. Exact for the
geometry; the only approximation is interpolation between traced angles.

    mesh.py    numpy geometry -> PLY. DJI Mini 3 parts, parameterised.
    scene.py   Sionna scene: floor, drone parts, K-LC6 / TinyRad antennas.
    trace.py   angle-lookup tracer with on-disk cache.
    synth.py   lookup tables + RPM profile -> complex IF time series.

Output flows through sim.capture.render() and is written with source="sionna",
the tag dataset/README.md has reserved for exactly this since day one.
"""
