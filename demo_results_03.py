#!/usr/bin/env python3
"""
    demo_results_03.py
    Demonstrate the counting of objects after processing of a video source object returned after inference
    2024-09-03 v0.1 armw Initial DRAFT adapted from Ultralytics documentation
    https://github.com/baqwas/yolov8
    ParkCircus Productions

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    References
        https://yolov8.org/
"""
import os

import cv2
from ultralytics import YOLO

"""
Load a model using YOLOv8n which is a smaller and more efficient variant for object detection on ARM platforms
It offers a balance between model size, inference speed, and accuracy, making it a valuable choice applications where computational resources (e.g. mobile, embedded, SBC, etc.) are constrained.
The key characteristics of the models are:
yolov8n         low-power (ARM, SBC, mobile); smaller size, faster inference, reduced resources, moderate accuracy
yolov8m         medium-size balancing speed & accuracy
yolov8l         largest and most accurate, ideal for high-precision tasks
yolov8x         extra large for the highest accuracy
yolov8s         smallest and fastest
yolov8-pose     for human pose estimation (not applicable for current test cases)
"""
import os

def prefix_filenames(filenames, subfolder_name):
  """Prefixes the names of all files in a list with a subfolder name.

  Args:
    filenames: A list of filenames.
    subfolder_name: The subfolder name to prefix.

  Returns:
    A list of prefixed filenames.
  """

  return list(map(lambda filename: os.path.join(subfolder_name, filename), filenames))

source = "/home/reza/Videos/yolo/yolov8/rpi5/images/lavon1.mp4"
pretrained = ["yolov8n", "yolov8m", "yolov8l", "yolov8x", "yolov8s"]  # available models
selection = 4                           # set the index to choose a model for the current run
model = YOLO(f"{pretrained[selection]}.pt")  # pretrained YOLOv8n model

cap = cv2.VideoCapture(source)          # access the video file

while cap.isOpened():                   # loop through the video file
    success, frame = cap.read()         # frame-by-frame
    if success:
        results = model(frame)          # return a generator of Results objects
        annotated_frame = results[0].plot()  # visualize the results on the frame

        obb = results[0].obb               # oriented bounding boxes
        if obb is not None:
            xyxy_boxes = obb.xyxy
            print(xyxy_boxes.shape)
            object_count = len(obb)  # count detected objects

            for box in obb:
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame,
                    (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame,
                    f"Object Count: {object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("demo_results_02", annotated_frame)  # display the results

        if cv2.waitKey(1) & 0xFF == ord('q'):  # iteration broken by pressing q on the keyboard
            break
    else:
        break                           # no more frames to process

cap.release()                           # good housekeeping
cv2.destroyAllWindows()
