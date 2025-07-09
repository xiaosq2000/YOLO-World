#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Tencent Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import cv2
import git
import numpy as np
import rospy
import supervision as sv
import torch
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose


class YOLOWorldROS:
    """
    A ROS node to perform object detection using the YOLO-World model and publish
    results as vision_msgs/Detection2DArray.
    """

    def __init__(self):
        """
        Initializes the ROS node, loads the model, and sets up publishers
        and subscribers.
        """
        rospy.loginfo("Initializing YOLO-World ROS node...")

        # Get parameters from ROS parameter server
        self.config_file = rospy.get_param("~config_file")
        self.checkpoint_file = rospy.get_param("~checkpoint_file")
        self.text_prompts = rospy.get_param("~text_prompts", "person,dog,cat")
        self.score_threshold = rospy.get_param("~score_threshold", 0.05)
        self.top_k = rospy.get_param("~top_k", 100)
        self.device = rospy.get_param("~device", "cuda:0")
        self.use_amp = rospy.get_param("~use_amp", False)
        input_image_topic = rospy.get_param("~input_image_topic", "/camera/rgb/image_raw")
        output_detections_topic = rospy.get_param(
            "~output_detections_topic", "/yolo_world/detections"
        )
        self.visualize = rospy.get_param("~visualize", False)

        # Load model configuration
        cfg = Config.fromfile(self.config_file)
        cfg.load_from = self.checkpoint_file
        self.model = init_detector(cfg, checkpoint=self.checkpoint_file, device=self.device)

        # Initialize test pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0]["type"] = "mmdet.LoadImageFromNDArray"
        self.test_pipeline = Compose(test_pipeline_cfg)

        # Prepare text prompts and reparameterize the model
        self.texts = [[t.strip()] for t in self.text_prompts.split(",")] + [[" "]]
        self.model.reparameterize(self.texts)

        # Initialize ROS components
        self.detection_pub = rospy.Publisher(
            output_detections_topic, Detection2DArray, queue_size=10
        )
        self.image_sub = rospy.Subscriber(
            input_image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )

        if self.visualize:
            annotated_image_topic = rospy.get_param(
                "~annotated_image_topic", "/yolo_world/annotated_image"
            )
            self.annotated_image_pub = rospy.Publisher(annotated_image_topic, Image, queue_size=10)
            # Rose-pine inspired colors
            rose_pine_colors = ["#ebbcba", "#c4a7e7", "#f6c177", "#9ccfd8", "#31748f", "#eb6f92"]
            self.color_palette = sv.ColorPalette.from_hex(rose_pine_colors)
            self.box_annotator = sv.BoxAnnotator(color=self.color_palette)
            self.label_annotator = sv.LabelAnnotator(color=self.color_palette)

        rospy.loginfo("YOLO-World ROS node initialized successfully.")
        rospy.loginfo(f"Detecting classes: {self.text_prompts}")

    def image_callback(self, msg):
        """
        Callback function for the image subscriber. Performs inference on the
        received image and publishes the detections.
        """
        try:
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == "rgb8":
                cv_image = cv_image[:, :, ::-1]  # Convert RGB to BGR
            elif msg.encoding != "bgr8":
                rospy.logerr(
                    f"Unsupported image encoding: {msg.encoding}. This node requires 'bgr8' or 'rgb8'."
                )
                return
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return

        # Prepare data for the model
        data_info = dict(img=cv_image, texts=self.texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0), data_samples=[data_info["data_samples"]]
        )

        # Perform inference
        start_time = rospy.get_time()
        with autocast(enabled=self.use_amp), torch.no_grad():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            pred_instances = pred_instances[pred_instances.scores.float() > self.score_threshold]
        latency_ms = (rospy.get_time() - start_time) * 1000

        if len(pred_instances.scores) > self.top_k:
            indices = pred_instances.scores.float().topk(self.top_k)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()

        if self.visualize:
            detections = sv.Detections(
                xyxy=pred_instances["bboxes"],
                confidence=pred_instances["scores"],
                class_id=pred_instances["labels"].astype(int),
            )
            labels = [
                f"{self.texts[class_id][0]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = self.box_annotator.annotate(
                scene=cv_image.copy(), detections=detections
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            latency_text = f"Latency: {latency_ms:.2f} ms"
            cv2.putText(
                annotated_frame,
                latency_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # Manually create Image message without cv_bridge
            annotated_image_msg = Image()
            annotated_image_msg.header = msg.header
            annotated_image_msg.height = annotated_frame.shape[0]
            annotated_image_msg.width = annotated_frame.shape[1]
            annotated_image_msg.encoding = "bgr8"
            annotated_image_msg.is_bigendian = 0
            annotated_image_msg.step = annotated_frame.shape[1] * 3
            annotated_image_msg.data = annotated_frame.tobytes()
            self.annotated_image_pub.publish(annotated_image_msg)

        # Create and populate the Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        for bbox, label, score in zip(
            pred_instances["bboxes"], pred_instances["labels"], pred_instances["scores"]
        ):
            detection = Detection2D()
            detection.header = msg.header

            # Bounding Box
            x1, y1, x2, y2 = bbox
            box = BoundingBox2D()
            box.size_x = x2 - x1
            box.size_y = y2 - y1
            box.center.x = x1 + box.size_x / 2.0
            box.center.y = y1 + box.size_y / 2.0
            detection.bbox = box

            # Object Hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(label)
            hypothesis.score = score
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = git.Repo(path=script_dir, search_parent_directories=True).working_tree_dir
    os.chdir(root_dir)
    try:
        rospy.init_node("yolo_world_ros_node")
        node = YOLOWorldROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
