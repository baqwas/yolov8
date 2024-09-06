#!/usr/bin/env python3
"""
    demo_results_01.py
    Demonstrate the processing of results object returned after inference
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

sub_folder = "./images/"                # prefix for images sub-folder
#  images = ["CENTERSTAGE.jpg", "RubberDuck.jpg", "boats.jpg", "elephant_men.jpg", "puffins.jpg"]  # images for the object detection
images = ["boats.jpg", "temple.jpg"]  # images for the object detection
images = prefix_filenames(images, sub_folder)  # prepare fully qualified names for images

for image_file in images:               # check if the images do exist in the specified sub-folder
    if not os.path.exists(image_file):
        print(f"Unable to access image file {image_file}")
        exit()                          # let's bail out before YOLOv8 complains

pretrained = ["yolov8n", "yolov8m", "yolov8l", "yolov8x", "yolov8s"]  # available models
selection = 4                           # set the index to choose a model for the current run
model = YOLO(f"{pretrained[selection]}.pt")  # pretrained YOLOv8n model

                                        # Run batched inference on a list of images
results = model(images, stream=True)    # return a generator of Results objects
filecount = 0
for result in results:                  # Process results generator
    boxes = result.boxes                # Boxes object for bounding box outputs
    masks = result.masks                # Masks object for segmentation masks outputs
    keypoints = result.keypoints        # Keypoints object for pose outputs
    probs = result.probs                # Probs object for classification outputs
    obb = result.obb                    # Oriented boxes object for OBB outputs
    result.show()                       # display to screen
    savefile = f"{os.path.splitext(images[filecount])[0]}_{pretrained[selection]}_{filecount:02d}."
    result.save(filename=savefile + "jpg")      # save annotated results to disk
    """
    Required parameters:
        path: where the text file will be saved
        names: list of class names corresponding to the detected objects
    Optional parameters:
        format: format of the text file. Can be "xywh" (default), "xyxy", or "voc"
        img_id: image ID to include in the filename
        line_delimiter: The delimiter used to separate results for each object. Default is "\n"
        save_conf: Whether to save the confidence scores along with the detections. Default is True
        save_crop: Whether to save the cropped images of detected objects. Default is False
        crop_dir: The directory where cropped images will be saved if save_crop is True
    result.save_txt(savefile + "txt", model.names)  # not supported in yolov8
    
    # Create a YOLOv8 model    
model = YOLO(model='yolov5s', device='cpu') 

# Read the image and perform object detection on it
image_path = "/content/ABC.JPG"
predictions = model(image_path, save_txt=None)
https://github.com/ultralytics/ultralytics/issues/2143
# Save the predicted text file to disk
with open("/content/ABC.txt", 'w') as file:
    for prediction in predictions.xyxy[0]:
        file.write(f"{prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")

with open("predicted_labels.txt", '+w') as file:
      for idx, prediction in enumerate(predictions[0].boxes.xywhn): # change final attribute to desired box format
          cls = int(predictions[0].boxes.cls[idx].item())
          path = predictions[0].path
          class_name = model.names[cls]
          # Write line to file in YOLO label format : cls x y w h
          file.write(f"{path} {class_name} {cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
    """
    filecount += 1