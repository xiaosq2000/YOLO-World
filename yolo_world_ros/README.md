# YOLO‑World ROS 1 Wrapper

A ROS 1 package that wraps the YOLO‑World open‑vocabulary object detector and publishes detections as vision_msgs/Detection2DArray with an optional annotated Image stream.

This package is intended to live inside the YOLO‑World repository tree and uses dynamic_reconfigure to update the vocabulary and runtime inference settings without restarting the node.

- Subscribes: sensor_msgs/Image (bgr8 or rgb8)
- Publishes: vision_msgs/Detection2DArray
- Publishes (optional): sensor_msgs/Image with bounding boxes and labels
- Dynamic params: text prompts (manual or file), score threshold, top‑k, AMP, visualization

## Quick start

1. Install ROS 1 (Noetic recommended. and YOLO-World:

2. Build in a catkin workspace:

3. Launch the node (adjust config and checkpoint if needed.:
```bash
roslaunch yolo_world_ros yolo_world.launch
```

4. Publish a test image:
```bash
rosrun image_publisher image_publisher /path/to/image.jpg /image:=/camera/rgb/image_raw
```

5. View results:
```bash
rosrun rqt_image_view rqt_image_view
```
Select /yolo_world/annotated_image, or inspect detections:
```bash
rostopic echo /yolo_world/detections
```

## Launch and parameters

Default launch file: yolo_world_ros/launch/yolo_world.launch

Parameters (can be overridden on the command line):
- config_file: Path to a YOLO‑World config (.py). Defaults to a v2_x config under ../configs/…
- checkpoint_file: Path to model weights (.pth) under ../weights/…
- device: Inference device string, e.g., cuda:0 or cpu
- input_image_topic: sensor_msgs/Image topic to subscribe (default /camera/rgb/image_raw)
- output_detections_topic: vision_msgs/Detection2DArray (default /yolo_world/detections)
- annotated_image_topic: sensor_msgs/Image with overlays (default /yolo_world/annotated_image)

Example overrides:
```bash
roslaunch yolo_world_ros yolo_world.launch device:=cpu \
  config_file:=$(rospack find yolo_world_ros)/../configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
  checkpoint_file:=$(rospack find yolo_world_ros)/../weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth
```

## Dynamic reconfigure

Run:
```bash
rosrun rqt_reconfigure rqt_reconfigure
```
Select the node (yolo_world_ros_node or yolo_world_ros).

Parameters:
- use_manual_prompts (bool): If true, use the text_prompts string; if false, read from text_prompt_file
- text_prompts (str): Comma‑separated list, e.g., "person, car, dog"
- text_prompt_file (str): Path to .txt or .json prompt file
  - .txt: one prompt per line
  - .json: list of lists, e.g., [["person"], ["car"], …]
- score_threshold (double): Confidence threshold (0..1)
- top_k (int): Keep top‑K predictions after thresholding
- use_amp (bool): Enable mixed‑precision inference
- visualize (bool): Publish the annotated image

Changes to prompts trigger an internal model reparameterization.

## Topics

Subscribed:
- input_image_topic (sensor_msgs/Image)
  - Only rgb8 and bgr8 encodings are supported.

Published:
- output_detections_topic (vision_msgs/Detection2DArray)
  - Each Detection2D includes:
    - bbox (BoundingBox2D) in image coordinates (center.x/center.y, size_x/size_y)
    - results[0].id (int): class index corresponding to the prompt list order
    - results[0].score (float): confidence score
  - To get the label string, map the integer id to the corresponding prompt string you provided.
- annotated_image_topic (sensor_msgs/Image)
  - Overlay with boxes and "label score" text. Toggle via visualize.

## Prompt files

- Text file (.txt): one class name per line
- JSON file (.json): top‑level list where each item is a list and item[0] is the class string
  Example:
  ```json
  [["person"], ["car"], ["dog"]]
  ```

The default text_prompt_file is data/texts/coco_class_texts.json.

## Notes and tips

- Throughput: Reduce image size upstream, increase score_threshold, or lower top_k. Enabling use_amp can improve speed on modern GPUs.
- Image encoding: If your camera publishes compressed or other encodings, republish or convert to bgr8/rgb8 before feeding this node.
- Repo root: The node uses GitPython to set the working directory to the repository root. Ensure this package is inside a Git repository clone.

## License

Apache-2.0. See package.xml headers for details. YOLO‑World is developed by Tencent/Next‑VLM; please follow their licensing and model usage terms.

## Acknowledgements

Thanks to
- YOLO‑World authors and the OpenMMLab ecosystem (MMEngine, MMDetection)
- ROS community and the maintainers of vision_msgs
