#!/usr/bin/env python3
"""
    count_line_02.py
    Count objects that are in specified classes using YOLOv8
    2024-09-05 v0.1 armw Initial deployment using Ultralytics example

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Parametric evaluation of ObjectCounter properties
    Original video is This is essentially the unmodified (except for video_path) code from Ultralytics documentation
    Processing
        Import libraries
        Load a pretrained YOLOv8 model
        Define the classes (i.e. 2 for car, 7 for truck, 9 for traffic light, etc.)
        Setup video capture and initialize object counter
        Process each frame to track objects and count the within the defined regions

    Reference
        https://docs.ultralytics.com/guides/object-counting/#advantages-of-object-counting
"""
import cv2
from ultralytics import YOLO, solutions

pretrained = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]  # available models
selection = 0
model_name = pretrained[selection]
model = YOLO(model_name + ".pt")

video_path = "/home/reza/Videos/yolo/yolov8/rpi5/images/"
video_file = "lavon3"       # the lavonx.mp4 files have 3840x2160 resolution
video_stream = video_path + video_file + ".mp4"
object_count_output = video_path + video_file + "_" +  model_name + "_02.avi"

cap = cv2.VideoCapture(video_stream)
assert cap.isOpened(), f"Error reading video file {video_stream}"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

                            # Define region points as rectangle in a 3840x2160 frame
                            # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
line_points = [(540, 1050), (1700, 1050)]
classes_to_count = [2, 7, 9]  # car, truck & traffic light

                            # Video writer
video_writer = cv2.VideoWriter(object_count_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                            # Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,          # flag to control whether to display the video stream
    reg_pts=line_points,    # list of points defining the counting region
    names=model.names,      # dictionary of classes names.
    draw_tracks=True,       # flag to control whether to draw the object tracks
    line_thickness=2,       # line thickness for bounding boxes
                            # count_reg_color, RGB color of the counting region, tuple, (255,0,255)
                            # count_txt_color, RGB color of the count text, tuple, (0,0,0)
                            # count_bg_color, RGB color of the count text background, tuple, (255,255,255)
                            # track_thickness, thickness of the track lines, int, 2
                            # view_in_counts, display in counts on video stream, bool, True
                            # view_out_counts, display out counts on video stream, bool, True
                            # track_color, RGB color of the tracks, tuple, None
                            # region_thickness, thickness of object counting regions, int, 5
                            # line_dist_thresh, Euclidean distance threshold for line counter, int, 15
                            #  cls_txtdisplay_gap, display gap between each class count, int, 50
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("All frames from the video have been processed.")
        break
    tracks = model.track(
        im0,                # source, im0, None, source directory for images or videos
        persist=True,       # bool, False, persisting tracks between frames
        show=False,          #
        classes=classes_to_count
    )

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()