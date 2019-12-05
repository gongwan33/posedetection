from matplotlib import pyplot as plt
import cv2
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
import mxnet as mx
import time
import os
import sys
from output import JSONOutput

filename = sys.argv[1]
#filename = '/local/gong/testdata/Event20190916130248003-rotated.avi'
#filename = '/local/gong/PD601/short.mp4'
resdir = 'resvideos'

filenameNoPath = os.path.basename(filename)
filenameNoExt = os.path.splitext(filenameNoPath)[0]
#filename = '/local/gong/PD601/short.mp4'
shortsize = 320
detect = True
showimage = False
frame_rate_skip = 20

#detector = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
#detector = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
#detector = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

detector.reset_class(["person"], reuse_weights=['person'])

#vidcap = cv2.VideoCapture('short.mp4')
vidcap = cv2.VideoCapture(filename)

if vidcap.isOpened() == False:
    print("Cannot open file")
    exit(1)

vw = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
vh = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

filenoext = os.path.splitext(os.path.basename(filename))[0]
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
print("Video Resolution: %dx%d"%(vw, vh))

fps = vidcap.get(cv2.CAP_PROP_FPS)
print("Video is at %d fps"%fps)

shortside = vh if vw > vh else vw
scale = shortside/shortsize

print("Video out: %s"%filenoext)
vidout = cv2.VideoWriter(resdir + '/' + filenoext + '-res.mp4', fourcc, int(fps), (int(vw/scale), int(vh/scale)))

frame_count  = 0
frame_time_sum = 0
detect_time_sum = 0
pose_time_sum = 0
active_pose_frame_count = 0
total_time = 0

print("Scale down ratio: %d"%scale)

jsonOutput = JSONOutput(resdir + '/' + filenameNoExt + '.json')

while vidcap.isOpened():
    start_frame_time = time.time()
    success, frame = vidcap.read()

    if not success:
        break

    frame = cv2.resize(frame, (int(frame.shape[1]/scale), int(frame.shape[0]/scale)))

    if detect:
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        starttime = time.time()
        x, frame = data.transforms.presets.ssd.transform_test(frame, short = shortsize)

        class_IDs, scores, bounding_boxs = detector(x)

        pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs)
        detect_time_sum += (time.time() - starttime)

        if pose_input is not None:
            starttime = time.time()
            predicted_heatmap = pose_net(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

            jsonOutput.write(confidence, pred_coords, scores, frame_count + 1)

            img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                            box_thresh=0.5, keypoint_thresh=0.2)
            pose_time_sum += (time.time() - starttime)
            active_pose_frame_count += 1

            if showimage:
                cv_plot_image(img)

            vidout.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            if showimage:
                cv_plot_image(frame)

            vidout.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    else:
        if showimage:
            cv2.imshow('Frame', frame)

        vidout.write(frame)

    cv2.waitKey(1)
    frame_count += 1
    frame_time = time.time() - start_frame_time
    frame_time_sum += frame_time
    total_time += frame_time

    if frame_count%frame_rate_skip == 0:
        print("Frame %d"%frame_count)
        print("Frame Rate: %f"%(frame_rate_skip/frame_time_sum))
        print("Detection Time Consumption (per frame): %f s"%(detect_time_sum/frame_rate_skip))

        if active_pose_frame_count > 0:
            print("Pose Time Consumption (per frame): %f s"%(pose_time_sum/active_pose_frame_count))

        frame_time_sum = 0
        detect_time_sum = 0
        pose_time_sum = 0
        active_pose_frame_count = 0


print("Total Time: %d"%total_time)
print("Avg Frame Rate: %f"%round(total_time/(frame_count + 1), 2))

vidcap.release()
vidout.release()
cv2.destroyAllWindows()
jsonOutput.release()
