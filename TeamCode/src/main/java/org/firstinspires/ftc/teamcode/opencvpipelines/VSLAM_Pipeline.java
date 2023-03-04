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

    int width;
    int height;
    int minDescriptorCols;

    double fx;
    double fy;
    double cx;
    double cy;

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

    MatOfPoint2f srcPoints;
    MatOfPoint2f dstPoints;
    MatOfPoint2f corners;
    MatOfPoint2f transformedCorners;

    MatOfDMatch matches;

    List<MatOfPoint> transformedCornersList;

    List<DMatch> matchesList;

    List<Point> srcPts;
    List<Point> dstPts;
    List<Point> cornerPts;

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
            homography();
        }

        releaseAllMats();
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

    public void homography() {
        if (!empty) {
            if (matches.toArray().length<MIN_MATCHES) {
                empty = true;
                return;
            }

            ArrayList<Point> srcPts = new ArrayList<Point>();
            for (DMatch m : matches.toArray()) {
                Point pt = keyPoints.toList().get(m.queryIdx).pt;
                srcPts.add(pt);
            }

            MatOfPoint2f srcPoints = new MatOfPoint2f();
            srcPoints.fromList(srcPts);

            ArrayList<Point> dstPts = new ArrayList<Point>();
            for (DMatch m : matches.toArray()) {
                Point pt = previousKeyPoints.toList().get(m.trainIdx).pt;
                dstPts.add(pt);
            }

            MatOfPoint2f dstPoints = new MatOfPoint2f();
            dstPoints.fromList(dstPts);

            maskByte = new MatOfByte();
            homography = Calib3d.findHomography(srcPoints, dstPoints, Calib3d.RANSAC, 5.0, maskByte);

            if (homography != null) {
                cornerPts = new ArrayList<>(4);
                cornerPts.add(new Point(0, 0));
                cornerPts.add(new Point(0, img.rows() - 1));
                cornerPts.add(new Point(img.cols() - 1, img.rows() - 1));
                cornerPts.add(new Point(img.cols() - 1, 0));
                corners = new MatOfPoint2f(Converters.vector_Point2f_to_Mat(cornerPts));
                transformedCorners = new MatOfPoint2f();
                Core.perspectiveTransform(corners, transformedCorners, homography);
                transformedCornersPoint = new MatOfPoint(transformedCorners.toArray());
                transformedCornersList = new ArrayList<>();
                transformedCornersList.add(transformedCornersPoint);
                Imgproc.polylines(previousImg, transformedCornersList, true, RED, 3, Imgproc.LINE_AA);
            } else {
                maskByte = new MatOfByte(MatOfByte.zeros(matches.toList().size(), 1, (byte) 0));
            }

            Features2d.drawMatches(img, keyPoints, previousImg, previousKeyPoints, matches, outputImg, GREEN, BLUE, maskByte);
        }
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
        if (srcPoints != null) srcPoints.release();
        if (dstPoints != null) dstPoints.release();
        if (homography != null) homography.release();
        if (corners != null) corners.release();
        if (transformedCorners != null) transformedCorners.release();
        if (transformedCornersPoint != null) transformedCornersPoint.release();
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