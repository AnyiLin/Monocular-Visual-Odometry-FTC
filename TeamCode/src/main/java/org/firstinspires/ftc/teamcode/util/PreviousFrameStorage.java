package org.firstinspires.ftc.teamcode.util;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

public class PreviousFrameStorage {

    private static Mat previousImg = new Mat();
    private static MatOfKeyPoint previousKeyPoints = new MatOfKeyPoint();
    private static Mat previousDescriptors = new Mat();

    public static void clear() {
        previousDescriptors = new Mat();
        previousImg = new Mat();
        previousKeyPoints = new MatOfKeyPoint();
    }

    public static void setPreviousImg(Mat set) {
        if (set != null) {
            set.copyTo(previousImg);
        } else {
            previousImg = new Mat();
        }
    }

    public static void setPreviousKeyPoints(MatOfKeyPoint set) {
        if (set != null) {
            set.copyTo(previousKeyPoints);
        } else {
            previousKeyPoints = new MatOfKeyPoint();
        }
    }

    public static void setPreviousDescriptors(Mat set) {
        if (set != null) {
            set.copyTo(previousDescriptors);
        } else {
            previousDescriptors = new Mat();
        }
    }

    public static Mat getPreviousImg() {
        Mat output = new Mat();
        previousImg.copyTo(output);
        return output;
    }

    public static MatOfKeyPoint getPreviousKeyPoints() {
        MatOfKeyPoint output = new MatOfKeyPoint();
        previousKeyPoints.copyTo(output);
        return output;
    }

    public static Mat getPreviousDescriptors() {/*
        Mat output = new Mat();
        previousDescriptors.copyTo(output);
        return output;*/
        return previousDescriptors;
    }

    public static void getPreviousDescriptors(Mat get) {
        previousDescriptors.copyTo(get);
    }
}
