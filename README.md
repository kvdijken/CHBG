# CHBG
Convex Hull BG for ImageJ

This is a plugin for ImageJ (or Fiji if you wish) that evens out background illumniation in brightfield microscopy images. 

Compared to the wellknown Rolling Ball algorithm which is implemented in ImageJ in Process -> Subtract Background…

- it does not decrease contrast as much as RB 
- it is much, much faster

but also

- it cannot handle a decrease in intensity followed by in increase when scanning in any direction over the image; it only handle increases, followed by decreases, or only decreases. In other words, a local decrease in intensity cannot be handled. These cases are handled by RB very well. In brightfield microscopy most cases of uneven illumination are vignetting and lightfall to one side of the image (or a combination thereof).

This algorithm depends on

QuickHull3d by Richard van Nieuwenhoven, see https://github.com/Quickhull3d/quickhull3d

JSI by aled, see https://github.com/aled/jsi

Sources for QuickHull3D and JSI are included in this distribution.

Through dependencies of JSI it also needs the following jars

trove_3.0.3.jar
slf4j-api-1.5.11.jar
slf4j-nop-1.5.6.jar 

These jars can be found on the web.

