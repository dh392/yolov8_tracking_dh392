import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from array import array

# import socket

# ##########################
# ### Config socket ###
# s = socket.socket(socket.AF_INET,
# socket.SOCK_STREAM)         
# s.bind(('192.168.131.175',9999))   
# # s.bind(('localhost',9999))   
# s.listen(1)                  
# c, addr = s.accept()         

# print("CONNECTION FROM:", str(addr)) 

# ##########################

def main():

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )


    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=[480,640])


    for result in model.track(source=0, stream=True, agnostic_nms=True):    #show=True
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

        if (detections.tracker_id.any() == 1):
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


            # start_time = time.time()

            base_control(x_error, y_error)


        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        cv2.imshow("yolov8_human_tracking", frame)

        if(cv2.waitKey(1) == ord('q')):
            # break
            exit()

def base_control(x_error, y_error):
    if(x_error<=40 and x_error>=-40):
        x_error = 0

    if (y_error <=0):
        y_error = 0

    kx = 0.002
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

    data = str(vx) + '#' + str(vz) + '#'
    print(data)
    # c.send(data.encode())


if __name__== "__main__":
    main()