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
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
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
        self.last_scene_text = None

        # Visualization defaults
        self.hud_font_scale = 1.0
        self.bbox_thickness = 2
        self.label_font_scale = 0.6
        self.label_text_thickness = 1

        # Tagger thread state and latest image buffer
        self.last_image = None
        self.last_image_lock = threading.Lock()
        self.tagger_thread = None
        self.tagger_stop_event = threading.Event()
        self.last_tagger_latency_ms = None

        # Load model configuration
        cfg = Config.fromfile(self.config_file)
        cfg.load_from = self.checkpoint_file
        self.model = init_detector(cfg, checkpoint=self.checkpoint_file, device=self.device)

        # Initialize test pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0]["type"] = "mmdet.LoadImageFromNDArray"
        self.test_pipeline = Compose(test_pipeline_cfg)

        # Initialize visualization palette and annotators before dynamic reconfigure
        self.palette_lightness = 0.62
        self.palette_chroma = 0.18
        self._rebuild_color_palette()

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
        # Palette params
        self.palette_lightness = config.palette_lightness
        self.palette_chroma = config.palette_chroma

        # Visualization params
        self.hud_font_scale = config.hud_font_scale
        self.bbox_thickness = config.bbox_thickness
        self.label_font_scale = config.label_font_scale
        self.label_text_thickness = config.label_text_thickness

        # Recreate annotators with updated settings
        self._rebuild_color_palette()

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
                    t0 = time.perf_counter()
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
                        resp = requests.post(
                            self.tagger_url, files=files, timeout=self.tagger_timeout
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            self.last_tagger_latency_ms = (time.perf_counter() - t0) * 1000.0
                            objects = data.get("objects", [])
                            if isinstance(objects, list) and len(objects) > 0:
                                new_texts = [
                                    [str(t)]
                                    for t in objects
                                    if isinstance(t, str) and t.strip() != ""
                                ] + [[" "]]
                                # Defer reparameterize to image_callback; only set if changed
                                if not self._texts_equal(new_texts, self.texts):
                                    self.pending_texts = new_texts
                            scene = data.get("scene", None)
                            if isinstance(scene, str) and scene.strip() != "":
                                self.last_scene_text = scene.strip()
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

    # OKLCh-based color palette generation
    def _oklch_to_linear_srgb(self, L, C, h_deg):
        import math

        h = math.radians(h_deg)
        a = C * math.cos(h)
        b = C * math.sin(h)
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l = l_**3
        m = m_**3
        s = s_**3
        r_lin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        return r_lin, g_lin, b_lin

    def _srgb_compand(self, u):
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        if u <= 0.0031308:
            return 12.92 * u
        else:
            return 1.055 * (u ** (1.0 / 2.4)) - 0.055

    def _oklch_to_srgb_hex(self, L, C, h_deg):
        C_local = max(0.0, float(C))
        r_lin = g_lin = b_lin = 0.0
        for _ in range(24):
            r_lin, g_lin, b_lin = self._oklch_to_linear_srgb(L, C_local, h_deg)
            if 0.0 <= r_lin <= 1.0 and 0.0 <= g_lin <= 1.0 and 0.0 <= b_lin <= 1.0:
                break
            C_local *= 0.9
        r = self._srgb_compand(r_lin)
        g = self._srgb_compand(g_lin)
        b = self._srgb_compand(b_lin)
        r8 = max(0, min(255, int(round(r * 255))))
        g8 = max(0, min(255, int(round(g * 255))))
        b8 = max(0, min(255, int(round(b * 255))))
        return f"#{r8:02x}{g8:02x}{b8:02x}"

    def _generate_oklch_palette_hex(self, n, seed_h=0.0):
        # Evenly spaced hues using the golden angle
        hex_list = []
        golden_angle = 137.50776405003785
        L = float(getattr(self, "palette_lightness", 0.62))
        C = float(getattr(self, "palette_chroma", 0.18))
        for i in range(max(1, int(n))):
            h = (seed_h + i * golden_angle) % 360.0
            hex_list.append(self._oklch_to_srgb_hex(L, C, h))
        return hex_list

    def _rebuild_color_palette(self):
        n = len(self.texts) if isinstance(self.texts, list) else 1
        hex_list = self._generate_oklch_palette_hex(n)
        self.color_palette = sv.ColorPalette.from_hex(hex_list)
        self.box_annotator = sv.BoxAnnotator(
            color=self.color_palette,
            thickness=self.bbox_thickness,
            text_scale=self.label_font_scale,
            text_thickness=self.label_text_thickness,
        )

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
                    self._rebuild_color_palette()
                except Exception as e:
                    rospy.logerr(f"Failed to apply auto prompts: {e}")

        # Prepare data for the model
        data_info = dict(img=cv_image, texts=self.texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0), data_samples=[data_info["data_samples"]]
        )

        # Perform inference
        detector_start_t = rospy.get_time()
        with autocast(enabled=self.use_amp), torch.no_grad():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            pred_instances = pred_instances[pred_instances.scores.float() > self.score_threshold]
        detector_latency_ms = (rospy.get_time() - detector_start_t) * 1000

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

            # Draw boxes and labels using Supervision
            names = [
                (
                    row[0].strip()
                    if isinstance(row, list) and len(row) > 0 and isinstance(row[0], str)
                    else ""
                )
                for row in self.texts
            ]
            labels = []
            for cls, conf in zip(detections.class_id, detections.confidence):
                idx = int(cls)
                if 0 <= idx < len(names) and names[idx] != "":
                    labels.append(f"{names[idx]} {float(conf):.2f}")
                else:
                    labels.append("")
            annotated_frame = self.box_annotator.annotate(
                scene=cv_image.copy(),
                detections=detections,
                labels=labels,
            )

            detector_text = f"Detector: {detector_latency_ms:.2f} ms"

            # Draw HUD text with Pillow for crisper rendering
            _pil_img = PILImage.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            _draw = ImageDraw.Draw(_pil_img)

            # Choose font size based on hud_font_scale
            _font_size = max(10, int(18 * self.hud_font_scale))
            _font = None
            if os.path.isfile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                _font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _font_size
                )
            else:
                try:
                    _font = ImageFont.truetype("DejaVuSans.ttf", _font_size)
                except Exception:
                    _font = ImageFont.load_default()

            # Draw detector and tagger text (RGB color)
            _draw.text((10, 30), detector_text, font=_font, fill=(255, 0, 0))
            if getattr(self, "prompt_source", None) == 2:
                tagger_text = (
                    f"Tagger: {self.last_tagger_latency_ms:.2f} ms"
                    if self.last_tagger_latency_ms is not None
                    else "Tagger: N/A"
                )
                _draw.text(
                    (10, int(30 + 28 * self.hud_font_scale)),
                    tagger_text,
                    font=_font,
                    fill=(255, 0, 0),
                )
            # Draw scene label at top-right if available
            if (
                isinstance(getattr(self, "last_scene_text", None), str)
                and self.last_scene_text.strip() != ""
            ):
                scene_text = f"scene: {self.last_scene_text}"
                try:
                    bbox = _draw.textbbox((0, 0), scene_text, font=_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w, text_h = _draw.textsize(scene_text, font=_font)
                x = _pil_img.width - text_w - 10
                y = 10
                _draw.text((x, y), scene_text, font=_font, fill=(255, 0, 0))

            # Convert back to BGR numpy array
            annotated_frame = cv2.cvtColor(np.array(_pil_img), cv2.COLOR_RGB2BGR)

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
