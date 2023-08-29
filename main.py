import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from array import array
import time

##########################################################################################################################################################
                              
                                                    ###############################################
                                                    ##                SOCKET CONFIG              ##
                                                    ###############################################

# import socket

# ### Config socket ###
# s = socket.socket(socket.AF_INET,
# socket.SOCK_STREAM)         
# s.bind(('192.168.131.175',9999))   
# # s.bind(('localhost',9999))   
# s.listen(1)                  
# c, addr = s.accept()         

# print("CONNECTION FROM:", str(addr)) 

##########################################################################################################################################################

##########################################################################################################################################################
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)


model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=[480,640])

def base_control(x_error, y_error):
    global vx, vz
    if(x_error<=40 and x_error>=-40):
        x_error = 0

    if (y_error <=0):
        y_error = 0

    # #simulation
    # kx = 0.008
    # kz = 0.1

    kx = 0.008
    kz = 0.002

    vx = kx*y_error
    vz = kz*x_error

    # max = 0.3
    vx_max = 0.5
    vz_max = 0.5
    if( vx > vx_max):
        vx = vx_max
    if( vx < -vx_max):
        vx = -vx_max
    if (vz > vz_max):
        vz = vz_max
    if (vz < -vz_max):
        vz = -vz_max

    vx = round(vx, 4)
    vz = round(vz, 4)

    data = str(vx) + '#' + str(vz) + '#'
    print(data)
    # c.send(data.encode())
##########################################################################################################################################################
     
##########################################################################################################################################################
                              
                                                    ###############################################
                                                    ## RUNNING WITH INTEL REALSENSE D435i CAMERA ##
                                                    ###############################################

# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))

# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# if device_product_line == 'L500':
#     config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# try:
#     while True:

#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#         # depth_colormap_dim = depth_colormap.shape
#         # color_colormap_dim = color_image.shape

#         # # If depth and color resolutions are different, resize color image to match depth image for display
#         # if depth_colormap_dim != color_colormap_dim:
#         #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
#         #     images = np.hstack((resized_color_image, depth_colormap))
#         # else:
#         #     images = np.hstack((color_image, depth_colormap))

#         # Show images
#         # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         # cv2.imshow('RealSense', color_image)
#         # cv2.waitKey(1)
#         for result in model.track(source=color_image, stream=True, agnostic_nms=True):    #show=True
#         # for result in model.track(source='http://169.254.199.250/mjpg/video.mjpg?timestamp=1692257423856', show=True, stream=True, agnostic_nms=True):    #show=True
#             frame = result.orig_img
#             detections = sv.Detections.from_yolov8(result)

#             # xxyy_check = detections.xyxy
#             # print(xxyy_check)

#             if result.boxes.id is not None:
#                 detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

#             detections = detections[detections.class_id == 0]

#             labels = [
#                 f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
#                 for _, confidence, class_id, tracker_id
#                 in detections
#             ]

#             #center of frame - dh392
#             x0c = 320
#             y0c = 240
#             cv2.circle(frame, (x0c,y0c), radius=5, color=(255, 0, 0), thickness=-1)

#             # print("type: " + str(type(detections.tracker_id)))

#             if (detections.tracker_id == 1):
#                 xyxy_check = (detections.xyxy)
#                 # print(type(xyxy_check))
#                 # print(xyxy_check)

#                 x_left_top = xyxy_check.item(0)
#                 y_left_top = xyxy_check.item(1)
#                 x_right_bottom = xyxy_check.item(2)
#                 y_right_bottom = xyxy_check.item(3)
#                 # print(type(x_left_top))
#                 # print(x_left_top)
#                 # print(str(x_left_top) + "---" + str(y_left_top) + "---" + str(x_right_bottom) + "---" + str(y_right_bottom))
#                 xc = round((x_left_top+x_right_bottom)/2, 2)
#                 yc = round((y_left_top+y_right_bottom)/2, 2)
#                 cv2.circle(frame, (int(xc),int(yc)), radius=5, color=(0, 0, 255), thickness=-1)

#                 center_human = "(" + str(xc) + " - " + str(yc) + ")"
#                 cv2.putText(frame, center_human, (int(xc)-10, int(yc)-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
#                 print("Center: " + str(xc) + " - " + str(yc))

#                 # error
#                 x_error = round(x0c- xc, 2)
#                 y_error = round(475 - y_right_bottom, 2)
#                 print("x_error: " + str(x_error) + " - y_error = " + str(y_error))


#                 # start_time = time.time()

#                 base_control(x_error, y_error)


#             frame = box_annotator.annotate(
#                 scene=frame, 
#                 detections=detections,
#                 labels=labels
#             )

#             cv2.imshow("yolov8_human_tracking", frame)

#             if(cv2.waitKey(1) == ord('q')):
#                 # break
#                 exit()
# finally:

#     # Stop streaming
#     pipeline.stop()

##########################################################################################################################################################

##########################################################################################################################################################
                              
                                                    ###############################################
                                                    ##        RUNNING WITH NORMAL CAMERA         ##
                                                    ###############################################
def main():

    with open('dh392_t.txt', 'a') as f:
        f.write("------------------------------------")
        f.write("\n")

    with open('dh392_x_error.txt', 'a') as fx:
        fx.write("------------------------------------")
        fx.write("\n")

    with open('dh392_y_error.txt', 'a') as fy:
        fy.write("------------------------------------")
        fy.write("\n")

    with open('dh392_tv.txt', 'a') as fv:
        fv.write("------------------------------------")
        fv.write("\n")

    with open('dh392_vx.txt', 'a') as fvx:
        fvx.write("------------------------------------")
        fvx.write("\n")

    with open('dh392_vz.txt', 'a') as fvz:
        fvz.write("------------------------------------")
        fvz.write("\n")

    global vx, vz

    start_time = time.time()

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )


    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=[480,640])


    for result in model.track(source=0, stream=True, agnostic_nms=True):    #show=True  --- source=2
    # for result in model.track(source='http://169.254.199.250/mjpg/video.mjpg?timestamp=1693295012221', show=True, stream=True, agnostic_nms=True):    #show=True
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        # xxyy_check = detections.xyxy
        # print(xxyy_check)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = detections[detections.class_id == 0]

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        #center of frame - dh392
        x0c = 320
        y0c = 240
        cv2.circle(frame, (x0c,y0c), radius=5, color=(255, 0, 0), thickness=-1)

        # print("type: " + str(type(detections.tracker_id)))

        if (detections.tracker_id == 1):
            xyxy_check = (detections.xyxy)
            # print(type(xyxy_check))
            # print(xyxy_check)

            x_left_top = xyxy_check.item(0)
            y_left_top = xyxy_check.item(1)
            x_right_bottom = xyxy_check.item(2)
            y_right_bottom = xyxy_check.item(3)
            # print(type(x_left_top))
            # print(x_left_top)
            # print(str(x_left_top) + "---" + str(y_left_top) + "---" + str(x_right_bottom) + "---" + str(y_right_bottom))
            xc = round((x_left_top+x_right_bottom)/2, 2)
            yc = round((y_left_top+y_right_bottom)/2, 2)
            cv2.circle(frame, (int(xc),int(yc)), radius=5, color=(0, 0, 255), thickness=-1)

            center_human = "(" + str(xc) + " - " + str(yc) + ")"
            cv2.putText(frame, center_human, (int(xc)-10, int(yc)-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
            print("Center: " + str(xc) + " - " + str(yc))

            # error
            x_error = round(x0c- xc, 2)
            y_error = round(475 - y_right_bottom, 2)
            print("x_error: " + str(x_error) + " - y_error = " + str(y_error))


            end_error_time = time.time()

            t = end_error_time - start_time

            with open('dh392_t.txt', 'a') as f:
                f.write(str(t))
                f.write("\n")

            with open('dh392_x_error.txt', 'a') as fx:
                fx.write(str(x_error))
                fx.write("\n")

            with open('dh392_y_error.txt', 'a') as fy:
                fy.write(str(y_error))
                fy.write("\n")

            base_control(x_error, y_error)

            end_vel_time = time.time()

            tv = end_vel_time - start_time

            with open('dh392_tv.txt', 'a') as fv:
                fv.write(str(tv))
                fv.write("\n")

            with open('dh392_vx.txt', 'a') as fvx:
                fvx.write(str(vx))
                fvx.write("\n")

            with open('dh392_vz.txt', 'a') as fvz:
                fvz.write(str(vz))
                fvz.write("\n")


        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        cv2.imshow("yolov8_human_tracking", frame)

        if(cv2.waitKey(1) == ord('q')):
            # break
            exit()

if __name__== "__main__":
    main()