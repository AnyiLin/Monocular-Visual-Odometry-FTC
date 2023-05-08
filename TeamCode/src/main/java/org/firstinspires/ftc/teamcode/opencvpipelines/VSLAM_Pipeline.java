package org.firstinspires.ftc.teamcode.opencvpipelines;

import static org.opencv.core.Core.NORM_HAMMING;

import org.firstinspires.ftc.robotcore.external.Telemetry;
//import org.firstinspires.ftc.teamcode.util.PreviousFrameStorage;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class VSLAM_Pipeline extends OpenCvPipeline {

    Telemetry telemetry;

    boolean firstFrame = false;
    boolean empty;

    final int MIN_MATCHES = 5;
    final int MAX_ITERATIONS_RANSAC = 1000;
    final int RANSAC_REPROJECTION_THRESHOLD = 3;

    int width;
    int height;
    int minDescriptorCols;

    double fx;
    double fy;
    double cx;
    double cy;

    final double INLIER_THRESHOLD_RANSAC = 2.0;

    final Size imgSize;

    Mat img;
    Mat previousImg;
    Mat outputImg;
    Mat descriptors;
    Mat descriptorsSubset;
    Mat previousDescriptors;
    Mat previousDescriptorsSubset;
    Mat cameraMatrix;
    Mat optimalCameraMatrix;
    Mat undistorted;
    Mat mask;
    Mat homography;

    MatOfByte maskByte;

    MatOfDouble distCoeffs;

    MatOfKeyPoint keyPoints;
    MatOfKeyPoint previousKeyPoints;

    MatOfPoint transformedCornersPoint;

    MatOfPoint2f corners;
    MatOfPoint2f transformedCorners;
    MatOfPoint2f currentFrameNormalizedKeyPoints;
    MatOfPoint2f previousFrameNormalizedKeyPoints;

    MatOfDMatch matches;
    MatOfDMatch matchesOutliersRemoved;

    List<DMatch> matchesList;

    final Scalar RED = new Scalar(255, 0, 0);
    final Scalar GREEN = new Scalar(0, 255, 0);
    final Scalar BLUE = new Scalar(0, 0, 255);

    ORB orb;

    BFMatcher bfMatcher;

    public VSLAM_Pipeline() {
        width = 640;
        height = 480;
        imgSize = new Size(width, height);
        fx = 822.317;
        fy = 822.317;
        cx = 319.495;
        cy = 242.502;
        distCoeffs = new MatOfDouble(new double[]{-0.0449369, 1.17277, 0, 0, -3.63244, 0, 0, 0});
        orb = ORB.create();
        bfMatcher = BFMatcher.create(NORM_HAMMING, true);
        constructCameraMatrices();
        previousImg = new Mat();
        previousKeyPoints = new MatOfKeyPoint();
        previousDescriptors = new Mat();
    }

    public VSLAM_Pipeline(int width, int height, double fx, double fy, double cx, double cy, double[] distCoeffsArray, Telemetry telemetry) {
        this.width = width;
        this.height = height;
        imgSize = new Size(width, height);
        this.fx = fx;
        this.fy = fy;
        this.cx = cx;
        this.cy = cy;
        distCoeffs = new MatOfDouble(distCoeffsArray);
        this.telemetry = telemetry;
        // Initiate ORB detector
        orb = ORB.create();
        // create BFMatcher object
        bfMatcher = BFMatcher.create(NORM_HAMMING, true);
        constructCameraMatrices();
        previousImg = new Mat();
        previousKeyPoints = new MatOfKeyPoint();
        previousDescriptors = new Mat();
    }

    @Override
    public Mat processFrame(Mat input) {
        empty = false;
        if (outputImg != null) outputImg.release();
        img = new Mat();
        Imgproc.cvtColor(input, img, Imgproc.COLOR_RGB2GRAY);
        outputImg = img.clone();

        undistortImage();

        if (firstFrame) {
            firstFrame = false;
            findKeyPointsAndDescriptorsORB();
        } else {
            //getPrevious();
            findKeyPointsAndDescriptorsORB();
            featureMatch();
            RANSAC();
            normalizeMatchedKeyPoints();
            // TODO: Do all the odometry stuff
            fundamentalAndEssentialMatrices();
        }




        releaseAllMats();

        resizeOutputImg();

        return outputImg;
    }

    public void findKeyPointsAndDescriptorsORB() {
        // find keypoints and descriptors with ORB
        keyPoints = new MatOfKeyPoint();
        descriptors = new Mat();
        mask = new Mat();
        orb.detectAndCompute(img, mask, keyPoints, descriptors);
        if (keyPoints.empty()||descriptors.empty()) {
            empty = true;
            return;
        }

        // draw keypoints
        Features2d.drawKeypoints(img, keyPoints, outputImg, GREEN);
    }

    public void featureMatch() {
        if (!empty) {
            if (previousDescriptors == null || previousDescriptors.empty()) {
                empty = true;
                return;
            }
            // match descriptors and change types as needed
            matches = new MatOfDMatch();
            //if (descriptors.type() != CvType.CV_32F) descriptors.convertTo(descriptors, CvType.CV_32F);
            //if (previousDescriptors.type() != CvType.CV_32F) previousDescriptors.convertTo(previousDescriptors, CvType.CV_32F);
            minDescriptorCols = Math.min(descriptors.cols(), previousDescriptors.cols());
            descriptorsSubset = descriptors.colRange(0, minDescriptorCols);
            previousDescriptorsSubset = previousDescriptors.colRange(0, minDescriptorCols);
            bfMatcher.match(descriptorsSubset, previousDescriptorsSubset, matches);

            // Convert matches to a list
            matchesList = matches.toList();

            // Sort matches by distance
            Collections.sort(matchesList, new Comparator<DMatch>() {
                @Override
                public int compare(DMatch match1, DMatch match2) {
                    return Float.compare(match1.distance, match2.distance);
                }
            });

            // draw matches
            Features2d.drawMatches(img, keyPoints, previousImg, previousKeyPoints, matches, outputImg, GREEN, BLUE);
        }
    }

    public void RANSAC() {
        if (!empty) {
            if (matches.toArray().length<MIN_MATCHES) {
                empty = true;
                return;
            }

            ArrayList<Point> currentPts = new ArrayList<Point>();
            for (DMatch m : matches.toArray()) {
                Point pt = keyPoints.toList().get(m.queryIdx).pt;
                currentPts.add(pt);
            }

            MatOfPoint2f currentPoints = new MatOfPoint2f();
            currentPoints.fromList(currentPts);

            ArrayList<Point> previousPts = new ArrayList<Point>();
            for (DMatch m : matches.toArray()) {
                Point pt = previousKeyPoints.toList().get(m.trainIdx).pt;
                previousPts.add(pt);
            }

            MatOfPoint2f previousPoints = new MatOfPoint2f();
            previousPoints.fromList(previousPts);

            Mat mask = new Mat();
            matchesOutliersRemoved = new MatOfDMatch(Calib3d.findHomography(currentPoints, previousPoints, Calib3d.RANSAC, RANSAC_REPROJECTION_THRESHOLD, mask, MAX_ITERATIONS_RANSAC, INLIER_THRESHOLD_RANSAC));

            currentPoints.release();
            previousPoints.release();
            mask.release();
        }
    }

    public void normalizeMatchedKeyPoints() {
        if (!empty) {
            currentFrameNormalizedKeyPoints = new MatOfPoint2f();
            previousFrameNormalizedKeyPoints = new MatOfPoint2f();

            List<KeyPoint> currentMatchedKeypoints = new ArrayList<>();
            List<KeyPoint> previousMatchedKeypoints = new ArrayList<>();

            for (DMatch match : matches.toArray()) {
                // Get the index of the matched keypoints in the original MatOfKeyPoint
                int currentIdx = match.queryIdx;
                int previousIdx = match.trainIdx;

                // Get the matched keypoints from the original MatOfKeyPoint
                KeyPoint keyPoint1 = keyPoints.toList().get(currentIdx);
                KeyPoint keyPoint2 = previousKeyPoints.toList().get(previousIdx);

                // Add the matched keypoints to their respective lists
                currentMatchedKeypoints.add(keyPoint1);
                previousMatchedKeypoints.add(keyPoint2);
            }

            double sumX = 0, sumY = 0;

            for (KeyPoint kp : currentMatchedKeypoints) {
                sumX += kp.pt.x;
                sumY += kp.pt.y;
            }
            Point currentCentroid = new Point(sumX / currentMatchedKeypoints.size(), sumY / currentMatchedKeypoints.size());

            sumX = 0;
            sumY = 0;

            for (KeyPoint kp : previousMatchedKeypoints) {
                sumX += kp.pt.x;
                sumY += kp.pt.y;
            }
            Point previousCentroid = new Point(sumX / currentMatchedKeypoints.size(), sumY / currentMatchedKeypoints.size());

            double currentAvgDist = 0;
            for (KeyPoint kp : currentMatchedKeypoints) {
                double dx = kp.pt.x - currentCentroid.x;
                double dy = kp.pt.y - currentCentroid.y;
                currentAvgDist += Math.sqrt(dx * dx + dy * dy);
            }
            currentAvgDist /= currentMatchedKeypoints.size();

            double previousAvgDist = 0;
            for (KeyPoint kp : previousMatchedKeypoints) {
                double dx = kp.pt.x - previousCentroid.x;
                double dy = kp.pt.y - previousCentroid.y;
                previousAvgDist += Math.sqrt(dx * dx + dy * dy);
            }
            previousAvgDist /= previousMatchedKeypoints.size();

            List<Point> pointsList = new ArrayList<>();
            for (KeyPoint kp : currentMatchedKeypoints) {
                pointsList.add(new Point((kp.pt.x - currentCentroid.x) / currentAvgDist, (kp.pt.y - currentCentroid.y) / currentAvgDist));
            }
            currentFrameNormalizedKeyPoints.fromList(pointsList);

            pointsList.clear();
            for (KeyPoint kp : previousMatchedKeypoints) {
                pointsList.add(new Point((kp.pt.x - previousCentroid.x) / previousAvgDist, (kp.pt.y - previousCentroid.y) / previousAvgDist));
            }
            previousFrameNormalizedKeyPoints.fromList(pointsList);
        }
    }
    public void fundamentalAndEssentialMatrices() {

    }

    public void resizeOutputImg() {
        Imgproc.resize(outputImg, outputImg, imgSize);
    }

    public void getPrevious() {
        //previousImg = PreviousFrameStorage.getPreviousImg();
        //previousKeyPoints = PreviousFrameStorage.getPreviousKeyPoints();
        //PreviousFrameStorage.getPreviousDescriptors(previousDescriptors);
    }

    public void releaseAllMats() {
        img.copyTo(previousImg);
        keyPoints.copyTo(previousKeyPoints);
        descriptors.copyTo(previousDescriptors);
        //PreviousFrameStorage.setPreviousImg(img);
        //PreviousFrameStorage.setPreviousKeyPoints(keyPoints);
        //PreviousFrameStorage.setPreviousDescriptors(descriptors);
        if (img != null) img.release();
        if (keyPoints != null) keyPoints.release();
        if (descriptors != null) descriptors.release();
        //if (previousImg != null) previousImg.release();
        //if (previousKeyPoints != null) previousKeyPoints.release();
        //if (previousDescriptors != null) previousDescriptors.release();
        if (mask != null) mask.release();
        if (maskByte != null) maskByte.release();
        if (matches != null) matches.release();
        if (undistorted != null) undistorted.release();
        if (homography != null) homography.release();
        if (corners != null) corners.release();
        if (transformedCorners != null) transformedCorners.release();
        if (transformedCornersPoint != null) transformedCornersPoint.release();
        if (currentFrameNormalizedKeyPoints != null) currentFrameNormalizedKeyPoints.release();
        if (previousFrameNormalizedKeyPoints != null) previousFrameNormalizedKeyPoints.release();
    }

    public void undistortImage() {
        undistorted = new Mat();
        Calib3d.undistort(img, undistorted, cameraMatrix, distCoeffs, optimalCameraMatrix);
    }

    public void constructCameraMatrices() {
        //     Construct the camera matrix.
        //
        //      --         --
        //     | fx   0   cx |
        //     | 0    fy  cy |
        //     | 0    0   1  |
        //      --         --
        //

        cameraMatrix = new Mat(3,3, CvType.CV_32FC1);

        cameraMatrix.put(0,0, fx);
        cameraMatrix.put(0,1,0);
        cameraMatrix.put(0,2, cx);

        cameraMatrix.put(1,0,0);
        cameraMatrix.put(1,1,fy);
        cameraMatrix.put(1,2,cy);

        cameraMatrix.put(2, 0, 0);
        cameraMatrix.put(2,1,0);
        cameraMatrix.put(2,2,1);
        optimalCameraMatrix = Calib3d.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1);
    }
}