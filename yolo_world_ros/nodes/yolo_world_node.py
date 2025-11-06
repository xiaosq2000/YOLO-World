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

import json
import os
import threading
import time

import cv2
import git
import numpy as np
import requests
import rospy
import supervision as sv
import torch
from dynamic_reconfigure.server import Server
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose
from yolo_world_ros.cfg import YOLOWorldConfig
from yolo_world_ros.msg import PromptList


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

        # Get static parameters from ROS parameter server
        self.config_file = rospy.get_param("~config_file")
        self.checkpoint_file = rospy.get_param("~checkpoint_file")
        self.device = rospy.get_param("~device", "cuda:0")
        input_image_topic = rospy.get_param("~input_image_topic", "/camera/rgb/image_raw")
        output_detections_topic = rospy.get_param(
            "~output_detections_topic", "/yolo_world/detections"
        )
        annotated_image_topic = rospy.get_param(
            "~annotated_image_topic", "/yolo_world/annotated_image"
        )

        # Initialize dynamic parameters
        self.text_prompts = None
        self.text_prompt_file = None
        self.prompt_source = None
        self.texts = [[" "]]
        self.base_texts = None
        self.auto_texts = None
        self.pending_texts = None

        # Tagger thread state and latest image buffer
        self.last_image = None
        self.last_image_lock = threading.Lock()
        self.tagger_thread = None
        self.tagger_stop_event = threading.Event()

        # Load model configuration
        cfg = Config.fromfile(self.config_file)
        cfg.load_from = self.checkpoint_file
        self.model = init_detector(cfg, checkpoint=self.checkpoint_file, device=self.device)

        # Initialize test pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0]["type"] = "mmdet.LoadImageFromNDArray"
        self.test_pipeline = Compose(test_pipeline_cfg)

        # Set up dynamic reconfigure
        self.reconfigure_server = Server(YOLOWorldConfig, self.reconfigure_callback)
        rospy.on_shutdown(self._stop_tagger_thread)

        # Initialize ROS components
        self.detection_pub = rospy.Publisher(
            output_detections_topic, Detection2DArray, queue_size=10
        )
        self.image_sub = rospy.Subscriber(
            input_image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )

        self.annotated_image_pub = rospy.Publisher(annotated_image_topic, Image, queue_size=10)
        self.prompts_pub = rospy.Publisher("prompts", PromptList, queue_size=10)
        # Rose-pine inspired colors
        rose_pine_colors = ["#ebbcba", "#c4a7e7", "#f6c177", "#9ccfd8", "#31748f", "#eb6f92"]
        self.color_palette = sv.ColorPalette.from_hex(rose_pine_colors)
        self.box_annotator = sv.BoxAnnotator(color=self.color_palette)
        self.label_annotator = sv.LabelAnnotator(color=self.color_palette)

        rospy.loginfo("YOLO-World ROS node initialized successfully.")

    def reconfigure_callback(self, config, level):
        """
        Callback for dynamic reconfigure server.
        """
        texts_changed = False

        # Update tagger config
        self.tagger_url = config.tagger_url
        self.tagger_fps = config.tagger_fps if config.tagger_fps > 0 else 1.0
        self.tagger_timeout = config.tagger_timeout if config.tagger_timeout > 0 else 10.0

        source_changed = self.prompt_source != config.prompt_source

        # Handle prompt source changes
        if source_changed:
            rospy.loginfo(
                f"Switching prompt source to {config.prompt_source} (0=manual,1=file,2=auto)"
            )
            if config.prompt_source == 2:
                self._start_tagger_thread()
            else:
                self._stop_tagger_thread()

        # Manual prompts
        if config.prompt_source == 0:
            if source_changed or self.text_prompts != config.text_prompts:
                rospy.loginfo(f"Using manual prompts: '{config.text_prompts}'")
                new_texts = [[t.strip()] for t in config.text_prompts.split(",") if t.strip()] + [
                    [" "]
                ]
                if len(new_texts) == 0:
                    new_texts = [[" "]]
                if not self._texts_equal(new_texts, self.texts):
                    self.texts = new_texts
                    texts_changed = True
                self.text_prompts = config.text_prompts

        # File prompts
        elif config.prompt_source == 1:
            if source_changed or self.text_prompt_file != config.text_prompt_file:
                if config.text_prompt_file and os.path.isfile(config.text_prompt_file):
                    rospy.loginfo(f"Using prompts from file: '{config.text_prompt_file}'")
                    try:
                        file_path = config.text_prompt_file
                        new_texts = None
                        if file_path.endswith(".txt"):
                            with open(file_path, "r") as f:
                                lines = f.readlines()
                            new_texts = [[t.rstrip("\r\n")] for t in lines if t.rstrip("\r\n")] + [
                                [" "]
                            ]
                        elif file_path.endswith(".json"):
                            with open(file_path, "r") as f:
                                loaded_json = json.load(f)
                            new_texts = [[item[0]] for item in loaded_json if item] + [[" "]]
                        else:
                            rospy.logwarn(
                                f"Unsupported prompt file format: {file_path}. "
                                "Only .txt and .json are supported. Not updating prompts."
                            )

                        if new_texts is not None and not self._texts_equal(new_texts, self.texts):
                            self.texts = new_texts
                            texts_changed = True

                        self.text_prompt_file = config.text_prompt_file
                    except Exception as e:
                        rospy.logerr(f"Error reading prompt file '{config.text_prompt_file}': {e}")
                elif config.text_prompt_file:
                    rospy.logwarn(
                        f"Prompt file not found: {config.text_prompt_file}. Using previous prompts."
                    )
                    self.text_prompt_file = config.text_prompt_file
                else:  # empty path
                    rospy.logwarn("Prompt file path is empty. Using previous prompts.")
                    self.text_prompt_file = config.text_prompt_file

        # Auto prompts: do nothing immediately; apply when tagger produces new tags
        elif config.prompt_source == 2:
            pass

        # Save source
        self.prompt_source = config.prompt_source

        if texts_changed:
            try:
                self.model.reparameterize(self.texts)
                self._publish_prompts(self.texts)
            except Exception as e:
                rospy.logerr(f"Failed to apply prompts: {e}")

        # Other runtime params
        self.score_threshold = config.score_threshold
        self.top_k = config.top_k
        self.use_amp = config.use_amp
        self.visualize = config.visualize
        return config

    def _start_tagger_thread(self):
        if self.tagger_thread is not None and self.tagger_thread.is_alive():
            return
        self.tagger_stop_event.clear()
        self.tagger_thread = threading.Thread(target=self._tagger_worker, daemon=True)
        self.tagger_thread.start()
        rospy.loginfo("Started VLM tagger thread.")

    def _stop_tagger_thread(self):
        if self.tagger_thread is None:
            return
        self.tagger_stop_event.set()
        try:
            self.tagger_thread.join(timeout=2.0)
        except Exception:
            pass
        self.tagger_thread = None
        rospy.loginfo("Stopped VLM tagger thread.")

    def _tagger_worker(self):
        # latest-only loop at configured FPS
        while not self.tagger_stop_event.is_set():
            start = time.time()
            img = None
            with self.last_image_lock:
                if self.last_image is not None:
                    img = self.last_image.copy()
            if img is not None and getattr(self, "prompt_source", None) == 2:
                try:
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
                        resp = requests.post(
                            self.tagger_url, files=files, timeout=self.tagger_timeout
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            tags = data.get("tags", [])
                            if isinstance(tags, list) and len(tags) > 0:
                                new_texts = [
                                    [str(t)] for t in tags if isinstance(t, str) and t.strip() != ""
                                ] + [[" "]]
                                # Defer reparameterize to image_callback; only set if changed
                                if not self._texts_equal(new_texts, self.texts):
                                    self.pending_texts = new_texts
                        else:
                            rospy.logwarn_throttle(5.0, f"Tagger HTTP {resp.status_code}")
                    else:
                        rospy.logwarn_throttle(5.0, "Failed to JPEG-encode frame for tagger.")
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"Tagger request failed: {e}")
            # sleep to maintain tagger_fps
            period = 1.0 / max(1e-6, getattr(self, "tagger_fps", 1.0))
            elapsed = time.time() - start
            sleep_t = max(0.0, period - elapsed)
            self.tagger_stop_event.wait(timeout=sleep_t)

    def _publish_prompts(self, texts):
        try:
            msg = PromptList()
            # drop the sentinel " " from publication
            msg.prompts = [
                row[0]
                for row in texts
                if isinstance(row, list) and len(row) > 0 and row[0].strip() != ""
            ]
            self.prompts_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to publish prompts: {e}")

    @staticmethod
    def _texts_equal(a, b):
        try:
            return a == b
        except Exception:
            return False

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

        # Update the latest image for the tagger (BGR)
        with self.last_image_lock:
            self.last_image = cv_image.copy()

        # Apply pending auto prompts if available
        if getattr(self, "prompt_source", None) == 2 and self.pending_texts is not None:
            new_texts = self.pending_texts
            self.pending_texts = None
            if not self._texts_equal(new_texts, self.texts):
                try:
                    self.model.reparameterize(new_texts)
                    self.texts = new_texts
                    self.auto_texts = new_texts
                    self._publish_prompts(self.texts)
                except Exception as e:
                    rospy.logerr(f"Failed to apply auto prompts: {e}")

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
            H, W = cv_image.shape[:2]
            detections = sv.Detections(
                xyxy=pred_instances["bboxes"],
                confidence=pred_instances["scores"],
                class_id=pred_instances["labels"].astype(int),
            )

            # Draw boxes using the existing color palette
            annotated_frame = self.box_annotator.annotate(
                scene=cv_image.copy(), detections=detections
            )

            # Draw readable labels with a solid background and on-screen clamping
            for bbox, class_id, confidence in zip(
                pred_instances["bboxes"], pred_instances["labels"], pred_instances["scores"]
            ):
                x1, y1, x2, y2 = bbox
                label = f"{self.texts[int(class_id)][0]} {float(confidence):.2f}"

                # Prefer to place label slightly above the top-left corner of the box
                tx = int(x1)
                ty = int(y1) - 8

                # Measure text and clamp so the full label stays visible on-screen
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                x0 = max(0, min(tx, W - tw - 6))
                y0 = max(th + 6, min(ty, H - 2))

                # Draw filled background for readability
                cv2.rectangle(
                    annotated_frame,
                    (x0, y0 - th - 6),
                    (x0 + tw + 6, y0 + baseline),
                    (0, 0, 0),
                    -1,
                )
                # Draw text on top
                cv2.putText(
                    annotated_frame,
                    label,
                    (x0 + 3, y0 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
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
