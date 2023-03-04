package org.firstinspires.ftc.teamcode.opmodes;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.firstinspires.ftc.teamcode.opencvpipelines.VSLAM_Pipeline;
import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraFactory;
import org.openftc.easyopencv.OpenCvCameraRotation;

@TeleOp (name = "VSLAM Test")
public class TestVSLAM extends LinearOpMode {

    OpenCvCamera camera;
    VSLAM_Pipeline pipeline;

    int width = 640;
    int height = 480;

    // Lens intrinsics
    // UNITS ARE PIXELS
    double fx = 822.317;
    double fy = 822.317;
    double cx = 319.495;
    double cy = 242.502;

    double[] distCoeffsArray = {-0.0449369, 1.17277, 0, 0, -3.63244, 0, 0, 0};

    @Override
    public void runOpMode() throws InterruptedException {
        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier("cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());
        camera = OpenCvCameraFactory.getInstance().createWebcam(hardwareMap.get(WebcamName.class, "camera"), cameraMonitorViewId);
        pipeline = new VSLAM_Pipeline(width, height, fx, fy, cx, cy, distCoeffsArray, telemetry);

        camera.setPipeline(pipeline);
        camera.openCameraDeviceAsync(new OpenCvCamera.AsyncCameraOpenListener()        {
            @Override
            public void onOpened()
            {
                camera.startStreaming(640,480, OpenCvCameraRotation.UPRIGHT);
            }

            @Override
            public void onError(int errorCode) {

            }
        });

        waitForStart();

        while (!isStopRequested()) {
            telemetry.addData("FPS:", camera.getFps());
            telemetry.addData("Overhead ms", camera.getOverheadTimeMs());
            telemetry.addData("Pipeline ms", camera.getPipelineTimeMs());
        }
    }
}
