After implementing the angular spetrum simulator, I tried to produce the
Fresnel diffraction of circular aperture. It turns out that the resolution 
of the objective will affect the diffraction pattern.

Resolution here means the spatial grid of our algorithm, not the pixel size
of the DMD.

Some tuning of the parameters revealed that, when $d = 2mm$, $a = 10\mu$m, and
$\lambda=500$nm, (Fresnel number $F = 0.1$),
a grid size of $\sim 0.1\mu$m would be sufficient. In actual 
experiments, I may be able to use a larger grid by ensuring the patterns look
roughly the same.

My GTX2080Ti has ~10GB memory, which is compatible with a ~1e4 grid number.
For the DMD of 1920 * 1080 resolution, and ~8um pixel size, we will need
1e3 * 1e2 will is basically the memory limit. 
Unfortunately, we may also need some zero padding... 
I can imagine a lot of tuning.
