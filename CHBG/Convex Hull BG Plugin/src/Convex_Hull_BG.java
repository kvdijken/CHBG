import java.awt.AWTEvent;
import java.awt.Color;
import java.awt.geom.GeneralPath;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import gnu.trove.procedure.TIntProcedure;
import ij.IJ;
import ij.ImagePlus;
import ij.Macro;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import net.sf.jsi.Point;
import net.sf.jsi.Rectangle;
import net.sf.jsi.SpatialIndex;
import net.sf.jsi.rtree.RTree;
import quickhull3d.Point3d;
import quickhull3d.QuickHull3D;

/**
 * 
 * @author koen
 *
 */
public class Convex_Hull_BG implements ExtendedPlugInFilter, DialogListener {
	ImagePlus imp;

	public class Edge {

		Point3d start;
		Point3d end;

		public Edge(Point3d start, Point3d end) {
			this.start = start;
			this.end = end;
		}

		public double dx() {
			return end.x - start.x;
		}

		public double dy() {
			return end.y - start.y;
		}

	}

	/**
	 * This is the z-component of the vertices for a square bottom plane in the
	 * intensity diagram. It is unclear if this is really needed.
	 */
	private static final int BOTTOM_DEPTH = -100;

	/**
	 * Switch (only during development) for using an artificial bottom in the
	 * intensity diagram or not.
	 */
	private static final boolean ARTIFICIAL_BOTTOM = false;

	/**
	 * 
	 * @author koen
	 *
	 */
	private class Face {

		private static final double DELTA = 0.1;

		// Reference to all the vertices of the hull
		Point3d[] hullVertices;

		// Reference to a list of indices of the vertices of this face
		int[] faceVertices;

		// cross product
		// this crossproduct points out of the hull
		double nx;
		double ny;
		double nz;

		// First point of this face
		private double px;
		private double py;
		private double pz;

		public Face(Point3d[] hullVertices, int[] faceVertices) {
			super();
			this.hullVertices = hullVertices;
			this.faceVertices = faceVertices;

			// Compute a point on this face, we need it later in several
			// occasions
			int p = faceVertices[0];
			px = hullVertices[p].x;
			py = hullVertices[p].y;
			pz = hullVertices[p].z;

			// We already have one vertex of the plane (p), collect two more to
			// calculate a normal for this plane..
			int q = faceVertices[1];
			int r = faceVertices[2];

			// get the coordinates of two more vertices of this face
			double qx = hullVertices[q].x;
			double qy = hullVertices[q].y;
			double qz = hullVertices[q].z;
			double rx = hullVertices[r].x;
			double ry = hullVertices[r].y;
			double rz = hullVertices[r].z;

			// compute two vectors in the plane
			// p, q and r are points on the plane
			// moving from p to q to r is in a ccw fashion, looking on
			// the hull from the outside
			//
			// qp = q - p
			// rq = r - q
			//
			// qp and rq are two vectors on the edge of the face
			// these vectors are directed in a righthand way
			double qpx = qx - px;
			double qpy = qy - py;
			double qpz = qz - pz;
			double rqx = rx - qx;
			double rqy = ry - qy;
			double rqz = rz - qz;

			// compute the cross product of these two vectors qp and rq
			// this crossproduct is a vector which points out of the hull
			nx = qpy * rqz - qpz * rqy;
			ny = qpz * rqx - qpx * rqz;
			nz = qpx * rqy - qpy * rqx;
		}

		/**
		 * Returns a bounding box for the projection of his face on the x-y
		 * plane
		 * 
		 * @return
		 */
		public Rectangle getBoundingBox() {
			int minX = Integer.MAX_VALUE;
			int maxX = Integer.MIN_VALUE;
			int minY = Integer.MAX_VALUE;
			int maxY = Integer.MIN_VALUE;
			for (int i = 0; i < faceVertices.length; i++) {
				int j = faceVertices[i];
				Point3d vertex = hullVertices[j];
				int x = (int) vertex.x;
				if (x > maxX)
					maxX = x;
				if (x < minX)
					minX = x;
				int y = (int) vertex.y;
				if (y > maxY)
					maxY = y;
				if (y < minY)
					minY = y;
			}
			return new Rectangle(minX, minY, maxX, maxY);
		}

		/**
		 * Returns true if the footstamp of this face contains the point (x,y).
		 * 
		 * If (x,y) is on one of the edges of this face, or is equal to any of
		 * the vertices of this face, true is returned.
		 * 
		 * From:
		 * http://stackoverflow.com/questions/8721406/how-to-determine-if-a-
		 * point-is-inside-a-2d-convex-polygon
		 * 
		 * @param x
		 * @param y
		 * @return
		 */
		public boolean contains(int x, int y) {
			int i;
			int j;
			boolean result = false;
			for (i = 0, j = faceVertices.length - 1; i < faceVertices.length; j = i++) {
				Point3d vertexI = hullVertices[faceVertices[i]];
				// The default algorithm has been extended to check for equality
				// with
				// any of the vertices. They could be missed by the default
				// algorithm.
				if ((int) vertexI.x == x && (int) vertexI.y == y)
					return true;
				Point3d vertexJ = hullVertices[faceVertices[j]];
				if ((vertexI.y > y) != (vertexJ.y > y)
						&& (x < (vertexJ.x - vertexI.x) * (y - vertexI.y) / (vertexJ.y - vertexI.y) + vertexI.x))
					result = !result;
			}
			return result;
		}

		public List<Edge> getEdges() {
			List<Edge> edges = new ArrayList<Edge>(faceVertices.length);
			int i;
			int j;
			for (i = 0, j = faceVertices.length - 1; i < faceVertices.length; j = i++) {
				Point3d vertexI = hullVertices[faceVertices[i]];
				Point3d vertexJ = hullVertices[faceVertices[j]];
				Edge edge = new Edge(vertexI, vertexJ);
				edges.add(edge);
			}
			return edges;
		}

		/**
		 * Returns h for the point (x,y,h) on this face.
		 * 
		 * From: http://tutorial.math.lamar.edu/Classes/CalcII/EqnsOfPlanes.aspx
		 * 
		 * @param x
		 * @param y
		 * @return
		 */
		public float height(int x, int y) {

			// The equation of the plane through this face is
			// nx * (x - px) + ny * (y - py) + nz * (z - pz) = 0
			//
			// So: nx * (x - px) + ny * (y - py) = nz * (pz - z)
			// So: (nx * (x - px) + ny * (y - py)) / nz = pz - z
			// So: z = pz - (nx * (x - px) + ny * (y - py)) / nz

			assert (contains(x, y) || closeTo(x, y));

			float h = (float) (pz - (nx * ((double) x - px) + ny * ((double) y - py)) / nz);
			return h;
		}

		@Override
		public String toString() {
			String result = "";
			for (int i = 0; i < faceVertices.length; i++) {
				Point3d vertex = hullVertices[faceVertices[i]];
				result += " (" + vertex.x + ", " + vertex.y + ", " + vertex.z + ")";
			}
			return result;
		}

		private double dotProduct(double p0x, double p0y, double p1x, double p1y) {
			return p0x * p1x + p0y * p1y;
		}

		private double magnitude(double x, double y) {
			return Math.sqrt(x * x + y * y);
		}

		/**
		 * http://stackoverflow.com/questions/10983872/distance-from-a-point-to-
		 * a-polygon
		 * 
		 * and
		 * 
		 * com.vividsolutions.jts.algorithm.CGAlgorithms.distancePointLine(
		 * Coordinate, Coordinate, Coordinate)
		 * 
		 * @param x
		 * @param y
		 * @param p1x
		 * @param p1y
		 * @param p2x
		 * @param p2y
		 * @return
		 */
		private double distance(double x, double y, double p1x, double p1y, double p2x, double p2y) {
			double p2m1x = p2x - p1x;
			double p2m1y = p2y - p1y;
			double xmp1x = x - p1x;
			double xmp1y = y - p1y;
			double r = ((x - p1x) * (p2x - p1x) + (y - p1y) * (p2y - p1y))
					/ ((p2x - p1x) * (p2x - p1x) + (p2y - p1y) * (p2y - p1y));

			if (r < 0) {
				return magnitude(xmp1x, xmp1y);
			} else if (r > 1) {
				double p2mx = p2x - x;
				double p2my = p2y - y;
				return magnitude(p2mx, p2my);
			} else {
				double p2m1x2 = (p2m1x) * (p2m1x);
				double p2m1y2 = (p2m1y) * (p2m1y);
				double s = ((p1y - y) * (p2x - p1x) - (p1x - x) * (p2m1y)) / (p2m1x2 + p2m1y2);
				return Math.abs(s) * Math.sqrt((p2m1x2 + p2m1y2));
			}
		}

		/**
		 * Checks if one of the edges of this face is close to (x,y)
		 * 
		 * @param x
		 * @param y
		 * @return
		 */
		public boolean closeTo(int x, int y) {
			int n = faceVertices.length;
			for (int i = 0; i < n; i++) {
				Point3d p1 = hullVertices[faceVertices[i]];
				Point3d p2 = hullVertices[faceVertices[(i + 1) % n]];
				double d = distance(x, y, p1.x, p1.y, p2.x, p2.y);
				if (d < DELTA)
					return true;
			}
			return false;
		}

	} // Face2

	/* parameters from the dialog: */
	private static boolean staticSeparateColors; // whether to create a separate
													// background for each color
													// channel
	private boolean separateColors = staticSeparateColors;

	private boolean isRGB; // whether we have an RGB image
	private static final int DIRECTION_PASSES = 9;

	private static final int CREATE_BG = 0;

	private static final int ADJUST_BG = 1;

	private static final int OVERLAY_CH = 2;

	// These two intensity adjustment methods do not apply to 32-bit float
	// images.
	private static final int SIMPLE_BOOST = 0;
	private static final int CONTRAST_ENHANCING_BOOST = 1;

	// different directions
	private int nPasses = DIRECTION_PASSES;
	private int pass;
	private int flags = DOES_ALL | FINAL_PROCESSING | KEEP_PREVIEW | PARALLELIZE_STACKS;

	private boolean calledAsPlugin = false; // http://stackoverflow.com/questions/6226290/default-value-of-boolean-in-java

	private String[] operations = { "Create background", "Adjust background", "Overlay CH" };

	private int operation = 1;

	private String[] boosts = { "Simple", "Contrast enhancing" };

	private int boost = CONTRAST_ENHANCING_BOOST;

	private String[] speedUps = { "1", "2", "3", "4", "5", "10", "25", "100", "500" };

	private int speedUp = 0;

	/**
	 * 
	 */
	public int setup(String arg, ImagePlus imp) {
		this.imp = imp;
		if (arg.equals("final")) {
			// TODO Kan wrs weg
			imp.getProcessor().resetMinAndMax();
			return DONE;
		} else
			return flags;
	}

	/**
	 * 
	 */
	public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
		isRGB = imp.getProcessor() instanceof ColorProcessor;
		calledAsPlugin = true;
		String options = Macro.getOptions();
		if (options != null) { // macro
			Macro.setOptions(options.replaceAll("white", "light"));
			separateColors = false;
		}
		GenericDialog gd = new GenericDialog(command);
		if (isRGB)
			gd.addCheckbox("Separate colors", separateColors);
		gd.addChoice("Operation", operations, operations[operation]);
		gd.addChoice("Brightness lifting", boosts, boosts[boost]);
		// We want to set the default speedup factor to such a value that an
		// accidental hit on enter will not cause the program to start a very
		// long proces. The goal is to find a speedup factor so about one
		// million samplepoints will result. This should be handled by most
		// computers quite easily.
		//
		// For that we need the image this dialog is showing for.
		ImageProcessor ip = imp.getProcessor();
		float mp = ip.getPixelCount() / 1024 / 1024;
		// Increase the default speedUp so the samplesize is less than one
		// million.
		for (speedUp = 0; speedUp < speedUps.length
				&& mp / Integer.parseInt(speedUps[speedUp]) / Integer.parseInt(speedUps[speedUp]) > 1; speedUp++)
			;

		gd.addChoice("Speedup", speedUps, speedUps[speedUp]);
		gd.addPreviewCheckbox(pfr);
		gd.addDialogListener(this);
		gd.showDialog();
		if (gd.wasCanceled())
			return DONE;
		IJ.register(this.getClass()); // protect static class variables (filter
										// parameters) from garbage collection
		if (options == null) // not a macro
			staticSeparateColors = separateColors;
		return IJ.setupDialog(imp, flags); // ask whether to process all slices
											// of stack (if a stack)
	}

	/**
	 * 
	 */
	public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
		if (isRGB)
			separateColors = gd.getNextBoolean();
		operation = gd.getNextChoiceIndex();
		boost = gd.getNextChoiceIndex();
		speedUp = gd.getNextChoiceIndex();
		return true;
	}

	/**
	 * 
	 */
	public void run(ImageProcessor ip) {
		switch (operation) {
		case CREATE_BG:
			command_createBG(ip);
			break;
		case ADJUST_BG:
			command_adjustBG(ip);
//			ip.resetMinAndMax();
//			imp.resetDisplayRange();
//			imp.updateChannelAndDraw();
			break;
		case OVERLAY_CH:
			command_overlayCH(ip);
			break;
		}
	} // run

	/**
	 * Creates an overlay of the faces of the convex hull over the image. This
	 * overlay is an overlay in the ImageJ sense in that it can be turned on and
	 * off.
	 * 
	 * It only works on brightness of the image (HSB) because it is not clear
	 * how to display 3 overlays and still make sense of them.
	 * 
	 * @param ip
	 */
	private void command_overlayCH(ImageProcessor ip) {
		// For now only on brightness, not on separate channels
		if (separateColors) {
			IJ.showMessage("overlayCH not supported on >1 channel");
			return;
		}
		ColorProcessor cp = (ColorProcessor) ip;
		int width = ip.getWidth();
		int height = ip.getHeight();
		byte[] H = new byte[width * height];
		byte[] S = new byte[width * height];
		byte[] B = new byte[width * height];
		cp.getHSB(H, S, B);
		ByteProcessor bp = new ByteProcessor(width, height, B, null);
		FloatProcessor fp = bp.toFloat(0, null);
		QuickHull3D hull = createHull(fp);
		final List<Face> faces = extractFaces(hull);
		drawFaces(faces);
	}

	/**
	 * Adjusts the image in ip to give it an even background.
	 * 
	 * @param ip
	 */
	private void command_adjustBG(ImageProcessor ip) {
		if (isRGB && !separateColors)
			adjustBackgroundOnBrightness((ColorProcessor) ip, false);
		else
			createOrAdjustBackground(ip, false);
	}

	/**
	 * Creates the background of the image in ip and puts it in ip.
	 * 
	 * @param ip
	 */
	private void command_createBG(ImageProcessor ip) {
		if (isRGB && !separateColors)
			adjustBackgroundOnBrightness((ColorProcessor) ip, true);
		else
			createOrAdjustBackground(ip, true);
	}

	/**
	 * Create or subtract a background, based on the brightness of an RGB image
	 * (keeping the hue of each pixel unchanged)
	 * 
	 * @param ip
	 *            The RGB image. On output, it will become the
	 *            background-subtracted image or the background (depending on
	 *            <code>createBackground</code>).
	 * @param createBackground
	 *            Whether to create a background, not to subtract it.
	 */
	public void adjustBackgroundOnBrightness(ColorProcessor ip, boolean createBackground) {
		int width = ip.getWidth();
		int height = ip.getHeight();
		byte[] H = new byte[width * height];
		byte[] S = new byte[width * height];
		byte[] B = new byte[width * height];
		ip.getHSB(H, S, B);
		ByteProcessor bp = new ByteProcessor(width, height, B, null);
		createOrAdjustBackground(bp, createBackground);
		ip.setHSB(H, S, (byte[]) bp.getPixels());
	}

	/**
	 * Create or subtract a background, works for all image types. For RGB
	 * images, the background is subtracted from each channel separately
	 * 
	 * @param ip
	 *            The image. On output, it will become the background-subtracted
	 *            image or the background (depending on
	 *            <code>createBackground</code>).
	 * @param createBackground
	 *            Whether to create a background, not to subtract it.
	 */
	public void createOrAdjustBackground(ImageProcessor ip, boolean createBackground) {
		// TODO Do not know how to test this
		boolean invertedLut = ip.isInvertedLut();
		boolean invert = !invertedLut;
		FloatProcessor bg = null;
		for (int channelNumber = 0; channelNumber < ip.getNChannels(); channelNumber++) {
			bg = ip.toFloat(channelNumber, bg);
			// bg know holds one channel of the original image in floats
			if ((ip instanceof FloatProcessor) && calledAsPlugin && !createBackground)
				bg.snapshot(); // float images need a snapshot to subtract

			convexHullFloatBackground(bg);
			// now bg holds the background of the image (in floats)

			if (createBackground)
				ip.setPixels(channelNumber, bg);
			else
				adjustBG_channel(ip, invert, bg, channelNumber);
		} // channel loop
	}

	/**
	 * 
	 * @param ip
	 * @param invert
	 * @param bg
	 * @param channel
	 */
	private void adjustBG_channel(ImageProcessor ip, boolean invert, FloatProcessor bg, int channel) {
		if (ip instanceof FloatProcessor)
			adjustNG_channelFloat(bg);
		else if (ip instanceof ShortProcessor)
			// for example 16-bit tiff
			adjustBG_channelShort(ip, bg, invert);
		else if (ip instanceof ByteProcessor)
			adjustBG_channelByte(ip, bg, invert);
		else if (ip instanceof ColorProcessor)
			adjustBG_channelColor(ip, bg, invert, channel);
	} // adjustBG_channel

	/**
	 * 
	 * @param ip
	 *            Holds the original image. This image will be adjusted to have
	 *            a flat, max background.
	 * @param bg
	 *            Holds the background of the image in ip (floats)
	 * @param invert
	 * @param channel
	 *            Which channel
	 */
	private void adjustBG_channelColor(ImageProcessor ip, FloatProcessor bg, boolean invert, int channel) {
		float[] bgPixels = (float[]) bg.getPixels();
		float ceiling = invert ? 255.5f : 0.5f;
		int[] pixels = (int[]) ip.getPixels();
		int shift = 16 - 8 * channel;

		int byteMask = 255 << shift;
		int resetMask = 0xffffffff ^ (255 << shift);

		for (int p = 0; p < bgPixels.length; p++) {
			float BG = bgPixels[p];
			float I = (pixels[p] & byteMask) >> shift;
			float I2 = calculatePixelIntensity(ceiling, BG, I);
			pixels[p] = (pixels[p] & resetMask) | ((int) I2 << shift);
		}
	}

	/**
	 * 
	 * @param ip
	 * @param bg
	 * @param invert
	 */
	private void adjustBG_channelByte(ImageProcessor ip, FloatProcessor bg, boolean invert) {
		float[] bgPixels = (float[]) bg.getPixels();
		float L = invert ? 255.5f : 0.5f;
		byte[] pixels = (byte[]) ip.getPixels();
		for (int p = 0; p < bgPixels.length; p++) {
			float BG = bgPixels[p];
			float I = pixels[p] & 0xff;
			float I2 = calculatePixelIntensity(L, BG, I);
			pixels[p] = (byte) I2;
		}
	}

	/**
	 * This method handles short images, for example 16-bit tiff's.
	 * 
	 * @param ip
	 * @param bg
	 * @param invert
	 */
	private void adjustBG_channelShort(ImageProcessor ip, FloatProcessor bg, boolean invert) {
		float[] bgPixels = (float[]) bg.getPixels();
		float L = invert ? 65535.5f : 0.5f;
		short[] pixels = (short[]) ip.getPixels();
		for (int p = 0; p < bgPixels.length; p++) {
			float BG = bgPixels[p];
			float I = pixels[p] & 0xffff;
			float I2 = calculatePixelIntensity(L, BG, I);
			pixels[p] = (short) I2;
		}
	}

	/**
	 * 
	 * @param bg
	 */
	private void adjustNG_channelFloat(FloatProcessor bg) {
		float[] bgPixels = (float[]) bg.getPixels();
		float[] snapshotPixels = (float[]) bg.getSnapshotPixels(); // original
																	// data
																	// in
																	// the
																	// snapshot
		for (int p = 0; p < bgPixels.length; p++)
			bgPixels[p] = snapshotPixels[p] - bgPixels[p];
//		reset(imp,bg);
//		bg.resetMinAndMax();
//		imp.resetDisplayRange();
//		double defaultMin = imp.getDisplayRangeMin();
//		double defaultMax = imp.getDisplayRangeMax();
//		double min = defaultMin;
//		double max = defaultMax;
//		imp.setDisplayRange(min, max);
//		imp.updateChannelAndDraw();
	}

	void reset(ImagePlus imp, ImageProcessor ip) {
		int bitDepth = imp.getBitDepth();
			imp.resetDisplayRange();
			double defaultMin = imp.getDisplayRangeMin();
			double defaultMax = imp.getDisplayRangeMax();
		double min = defaultMin;
		double max = defaultMax;
		setMinAndMax(imp, min, max);
	}
	
	void setMinAndMax(ImagePlus imp, double min, double max) {
		boolean rgb = imp.getType()==ImagePlus.COLOR_RGB;
				imp.setDisplayRange(min, max);
	}


	/**
	 * This calculates the new intensity for this pixel. It depends on the
	 * ceiling, the estimated local background intensity and the observed
	 * intensity. The new pixel is estimated such that were it the background,
	 * it would get the ceiling value.
	 * 
	 * This method currently knows two methods of calculating the new pixel
	 * value: SIMPLE_BOOST and CONTRAST_ENHANCING_BOOST.
	 * 
	 * @param ceiling
	 *            The maximum value for the current image type. For a 16-bit
	 *            tiff this is 65535, for an RGB this is 255 (for each channel).
	 * @param BG
	 *            Estimated intensity of the local background.
	 * @param I
	 *            Current intensity of the pixel.
	 * @return
	 */
	private float calculatePixelIntensity(float ceiling, float BG, float I) {
		float I2;
		// To avoid division by zero
		if (BG > 0) {
			float d = BG - I;
			float D = ceiling - BG;

			// Calculate the brightness boost (dI)
			float dI = 0;
			if (boost == SIMPLE_BOOST)
				// Simple scenario
				dI = D;
			else if (boost == CONTRAST_ENHANCING_BOOST)
				// Contrast enhancing scenario
				dI = D * (1f - d / BG);

			I2 = I + dI;
			if (I2 < 0f)
				I2 = 0f;
			else if (I2 > ceiling)
				I2 = (float) Math.floor(ceiling);
		} else
			I2 = 0f;
		return I2;
	}

	/**
	 * Replaces the image in fp with its background
	 * 
	 * @param fp
	 *            on entrance holds the image, on exit holds its estimated
	 *            background
	 */
	void convexHullFloatBackground(FloatProcessor fp) {
		// float[] pixels = (float[]) fp.getPixels(); // this will become the
		// background
		showProgress(0.0);
		// if (invert)
		// for (int i = 0; i < pixels.length; i++)
		// pixels[i] = -pixels[i];
		if (Thread.currentThread().isInterrupted())
			return;
		createBackground(fp);
		if (Thread.currentThread().isInterrupted())
			return;
		showProgress(0.9);
		if (Thread.currentThread().isInterrupted())
			return;

		// if (invert)
		// for (int i = 0; i < pixels.length; i++)
		// pixels[i] = -pixels[i];
		pass++;
	}

	/**
	 * This class checks a point against a face. Is this point part of the face?
	 * 
	 * @author koen
	 *
	 */
	private class PointAgainstFaceChecker implements TIntProcedure {

		public static final int NOT_FOUND = -1;
		private int faceIndex = NOT_FOUND;
		private List<Face> faces;
		private int x;
		private int y;
		private boolean strict;

		public PointAgainstFaceChecker(List<Face> faces) {
			super();
			this.faces = faces;
		}

		/**
		 * 
		 */
		@Override
		public boolean execute(int i) {
			Face face = faces.get(i);
			// TODO Probably it is just as good to combine strict and not strict
			// into one test. The resulting is probably very, very small. If
			// a closeTo match is found even if later a strict match could
			// be found. the face which is found by closeTo has at nearly
			// the same brightness value at this position. Faces connect
			// brightness values at their common edges.
			if (strict && face.contains(x, y)) {
				faceIndex = i;
				return false;
			} else if (!strict && face.closeTo(x, y)) {
				faceIndex = i;
				return false;
			} else
				return true;
		}

		public int getFaceIndex() {
			return faceIndex;
		}

		public void findStrict(int x, int y) {
			this.x = x;
			this.y = y;
			faceIndex = NOT_FOUND;
			strict = true;
		}

		public void findNotStrict(int x, int y) {
			this.x = x;
			this.y = y;
			faceIndex = NOT_FOUND;
			strict = false;
		}
	} // PointAgainstFaceChecker

	/**
	 * On exit fp holds the background of the image
	 * 
	 * @param fp
	 */
	private void createBackground(FloatProcessor fp) {
		QuickHull3D hull = createHull(fp);
		final List<Face> faces = extractFaces(hull);
		SpatialIndex index = indexFaces(faces);

		// Now cycle over each (x,y) in the image to find the corresponding
		// face in the hull.
		List<Point> problemPixels = new Vector<Point>();
		calculateBackgroundIntensity(fp, faces, index, problemPixels);
		if (problemPixels.size() > 0)
			solveProblemPixels(problemPixels);
	}

	/**
	 * 
	 * @param ip
	 * @param faces
	 */
	private void drawFaces(List<Face> faces) {
		GeneralPath path = new GeneralPath();
		for (Face face : faces)
			drawFace(path, face);
		imp.setOverlay(path, Color.BLACK, null);
	}

	/**
	 * 
	 * @param path
	 * @param face
	 */
	private void drawFace(GeneralPath path, Face face) {
		List<Edge> edges = face.getEdges();
		for (Edge edge : edges)
			if (edge.dx() >= 0 || edge.dy() == 0)
				drawEdge(path, edge);
	}

	/***
	 * 
	 * @param path
	 * @param edge
	 */
	private void drawEdge(GeneralPath path, Edge edge) {
		path.moveTo((int) edge.start.x, (int) edge.start.y);
		path.lineTo((int) edge.end.x, (int) edge.end.y);
	}

	/**
	 * 
	 * @param problemPixels
	 */
	private void solveProblemPixels(List<Point> problemPixels) {
		// TODO Auto-generated method stub

	}

	/**
	 * Calculates the intensity of the current background. The background is
	 * stored as 3d-faces of a 3d-Convex Hull over an image.
	 * 
	 * @param fp
	 *            In fp the background of the image will be stored.
	 * @param faces
	 *            Holds the faces of the hull
	 * @param index
	 *            Holds the spatial index with the projection on the x-y plane
	 *            of the faces
	 * @param problemPixels
	 *            Receives the pixels which could not be tied to a face. Should
	 *            remain empty.
	 */
	private void calculateBackgroundIntensity(FloatProcessor fp, final List<Face> faces, SpatialIndex index,
			List<Point> problemPixels) {
		int width = fp.getWidth();
		int height = fp.getHeight();
		float[] pixels = (float[]) fp.getPixels();
		PointAgainstFaceChecker checker = new PointAgainstFaceChecker(faces);
		Face cache = null;
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				Face face = null;

				// First check the face cache
				if (cache != null && cache.contains(x, y))
					face = cache;

				if (face == null) {
					// Nothing in the cache or the cache item is not the right
					// one.
					Point point = new Point(x, y);
					checker.findStrict(x, y);
					index.nearestN(point, checker, 1, 1);
					int faceIndex = checker.getFaceIndex();
					if (faceIndex == PointAgainstFaceChecker.NOT_FOUND) {
						// This point may be on the edge of one of the faces.
						// In that case the face.contains() method might fail.
						// Because adjacent faces have the same height at the
						// same point it is irrelevant which face we find.
						checker.findNotStrict(x, y);
						index.nearestN(point, checker, 1, 1);
						faceIndex = checker.getFaceIndex();
					}
					if (faceIndex != PointAgainstFaceChecker.NOT_FOUND)
						face = faces.get(faceIndex);
				}

				if (face != null) {
					// and calculate the value for the background intensity
					float h = face.height(x, y);
					int p = x + y * width;
					pixels[p] = h;
					cache = face;
				} else {
					// A face for this pixel could not be found.
					// Although we should never reach here, robustness problems
					// may cause this to happen.
					// Store this pixel and try something else later.
					if (problemPixels != null)
						problemPixels.add(new Point(x, y));
				}
			}
		}
	} // calculateBackgroundIntensity

	/**
	 * Creates a point array with for every samplepoint its x and y value and
	 * its intensity. These take 3 consecutive entries in the array. A speedup
	 * factor is taken into account so not every pixel in fp has to be sampled
	 * (but can, if speedupfactor == 1).
	 * 
	 * The resulting array can be used as input for the QuickHull algorithm.
	 * 
	 * @param fp
	 * @return
	 */
	private double[] createPointArray(FloatProcessor fp) {
		int speedUpFactor = Integer.parseInt(speedUps[speedUp]);

		int nPoints;
		int samplesX;
		int samplesY;
		int width = fp.getWidth();
		int height = fp.getHeight();
		float[] pixels = (float[]) fp.getPixels();

		// nPoints = nPixels / speedUpFactor;
		samplesX = width / speedUpFactor;
		if (width % speedUpFactor != 0)
			samplesX += 1;
		samplesY = height / speedUpFactor;
		if (height % speedUpFactor != 0)
			samplesY += 1;

		// Make sure we also can store the edge of the image
		boolean extraX = width % speedUpFactor != 1;
		if (extraX)
			samplesX += 1;
		boolean extraY = height % speedUpFactor != 1;
		if (extraY)
			samplesY += 1;

		nPoints = samplesX * samplesY;
		if (ARTIFICIAL_BOTTOM)
			nPoints += 4;

		double d[] = new double[nPoints * 3];

		int i = 0;

		// First copy the pixel data
		for (int x = 0; x < width; x += speedUpFactor)
			for (int y = 0; y < height; y += speedUpFactor) {
				d[i++] = x;
				d[i++] = y;
				d[i++] = sample(width, height, pixels, x, y, speedUpFactor);
			}
		if (extraX) {
			int x = width - 1;
			for (int y = 0; y < height; y += speedUpFactor) {
				d[i++] = x;
				d[i++] = y;
				d[i++] = sample(width, height, pixels, x, y, speedUpFactor);
			}
		}
		if (extraY) {
			int y = height - 1;
			for (int x = 0; x < width; x += speedUpFactor) {
				d[i++] = x;
				d[i++] = y;
				d[i++] = sample(width, height, pixels, x, y, speedUpFactor);
			}
		}
		if (extraX && extraY) {
			int x = width - 1;
			int y = height - 1;
			d[i++] = x;
			d[i++] = y;
			d[i++] = sample(width, height, pixels, x, y, speedUpFactor);
		}

		if (ARTIFICIAL_BOTTOM) {
			// (0,0,0)
			d[i++] = 0;
			d[i++] = 0;
			d[i++] = BOTTOM_DEPTH;

			// (0,height-1,0)
			d[i++] = 0;
			d[i++] = height - 1;
			d[i++] = BOTTOM_DEPTH;

			// (width-1,height-1,0)
			d[i++] = width - 1;
			d[i++] = height - 1;
			d[i++] = BOTTOM_DEPTH;

			// (width-1,0,0)
			d[i++] = width - 1;
			d[i++] = 0;
			d[i++] = BOTTOM_DEPTH;
		}

		assert (i == nPoints * 3);

		return d;
	} // cretaePointArray

	/**
	 * Samples dxy * dxy pixels with (x,y) as the left bottom corner and returns
	 * the max value.of the pixels in that range.
	 * 
	 * @param width
	 * @param height
	 * @param pixels
	 * @param x
	 * @param y
	 * @param dxy
	 * @return
	 */
	private float sample(int width, int height, float[] pixels, int x, int y, int dxy) {
		float max = Float.MIN_VALUE;
		for (int i = x; i < x + dxy && i < width; i++)
			for (int j = y; j < y + dxy && j < height; j++) {
				int p = i + j * width;
				float f = pixels[p];
				if (f > max)
					max = f;
			}
		return max;
	} // sample

	/**
	 * 
	 * @param fp
	 * @return
	 */
	private QuickHull3D createHull(FloatProcessor fp) {
		double d[] = createPointArray(fp);
		QuickHull3D hull = new QuickHull3D();
		hull.build(d);
		return hull;
	} // createHull

	/**
	 * Returns a list of the 3d-faces of the 3d Convex Hull. The resulting list
	 * of faces does not include the downward looking faces.
	 */
	private List<Face> extractFaces(QuickHull3D hull) {
		Point3d[] hullVertices = hull.getVertices();
		int[][] faces = hull.getFaces();

		List<Face> result = new ArrayList<Face>(faces.length);

		for (int faceIndex = 0; faceIndex < faces.length; faceIndex++) {
			// Get indices to the vertices of this face.
			int[] faceVertices = faces[faceIndex]; // these are indices to the
													// vertices of this face

			// Create a highlevel face object of it
			Face face = new Face(hullVertices, faceVertices);
			// check if this face is one of the bottom or side faces. If so, do
			// not create a face object
			if (face.nz < 0)
				continue;
			if (face.nx == 0 && face.nz == 0)
				continue;
			if (face.ny == 0 && face.nz == 0)
				continue;

			// And add it to the list of faces
			result.add(face);
		}

		assert (result.size() > 0);

		return result;
	} // extractFaces

	/**
	 * The faces will be inserted in the returned spatial index with their
	 * corresponding index in the faces list.
	 * 
	 * @param faces
	 * @return
	 */
	private SpatialIndex indexFaces(List<Face> faces) {
		SpatialIndex index = new RTree();
		index.init(null);
		for (int i = 0; i < faces.size(); i++) {
			Face face = faces.get(i);
			Rectangle r = face.getBoundingBox();
			index.add(r, i);
		}
		return index;
	} // indexFaces

	public void setNPasses(int nPasses) {
		if (isRGB && separateColors)
			nPasses *= 3;
		this.nPasses = nPasses;
		pass = 0;
	}

	private void showProgress(double percent) {
		if (nPasses <= 0)
			return;
		percent = (double) pass / nPasses + percent / nPasses;
		IJ.showProgress(percent);
	}

}
