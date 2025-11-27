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

import copy
import json
import os
import threading
import time
from typing import Tuple

import cv2
import git
import numpy as np
import requests
import rospy
import supervision as sv
import torch
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
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
from yolo_world_ros.msg import LabelSet


class YOLOWorldROS:
    """
    A ROS node to perform object detection using the YOLO-World model and publish
    results as vision_msgs/Detection2DArray, along with the current label set
    and optional color palette for downstream consumers.
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
        rgb_image_topic = rospy.get_param("~rgb_image_topic", "rgb/image_raw")
        detections_topic = rospy.get_param("~detections_topic", "yolo_world/detections")
        annotated_image_topic = rospy.get_param(
            "~annotated_image_topic", "yolo_world/annotated_image"
        )
        label_set_topic = rospy.get_param("~label_set_topic", "yolo_world/annotated_image")

        # Initialize dynamic parameters
        self.text_prompts = None
        self.text_prompt_file = None
        self.prompt_source = None
        # Internal representation used by the model: list of [text] rows plus a sentinel [" "]
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
        # HUD visibility defaults
        self.show_detector_hud = True
        self.show_tagger_hud = True
        self.show_scene_hud = True
        self.show_zupt_hud = True

        # Tagger thread state and latest image buffer
        self.last_image = None
        self.last_image_lock = threading.Lock()
        self.tagger_thread = None
        self.tagger_stop_event = threading.Event()
        self.last_tagger_latency_ms = None
        self.last_tagger_stamp = None

        # Detector latency tracking for diagnostics
        self.last_detector_latency_ms = None
        self.last_detector_stamp = None

        # Monotonically increasing identifier for label sets / palettes
        self.label_set_id = 0

        # Zero-Update (ZUPT) state - skip inference when image is unchanged
        self.zupt_reference_image: np.ndarray | None = None
        self.zupt_reference_image_lock = threading.Lock()
        self.zupt_cached_detections: Detection2DArray | None = None
        self.zupt_cached_annotated_image: Image | None = None
        self.zupt_cached_label_set_id: int = 0
        self.zupt_last_inference_time: float = 0.0
        # ZUPT statistics
        self.zupt_frames_skipped: int = 0
        self.zupt_frames_total: int = 0
        self.zupt_last_similarity_score: float = 0.0
        # ZUPT tagger-specific state
        self.zupt_tagger_reference_image: np.ndarray | None = None
        self.zupt_tagger_requests_skipped: int = 0

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

        # Prepare diagnostics publisher/timer state BEFORE dynamic reconfigure (callback runs immediately)
        self.diagnostics_pub = rospy.Publisher("~diagnostics", DiagnosticArray, queue_size=10)
        self._diag_timer = None
        self._diag_enabled = False
        self._diag_rate_hz = 2.0

        # Set up dynamic reconfigure
        self.reconfigure_server = Server(YOLOWorldConfig, self.reconfigure_callback)
        rospy.on_shutdown(self._stop_tagger_thread)

        # Initialize ROS components
        self.detection_pub = rospy.Publisher(detections_topic, Detection2DArray, queue_size=10)
        self.image_sub = rospy.Subscriber(
            rgb_image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )

        self.annotated_image_pub = rospy.Publisher(annotated_image_topic, Image, queue_size=10)

        # Unified label set + palette publisher (latched)
        self.label_set_pub = rospy.Publisher(label_set_topic, LabelSet, queue_size=1, latch=True)

        rospy.loginfo("YOLO-World ROS node initialized successfully.")
        # Publish initial label set so the latched topic is populated
        try:
            self._publish_label_set(self.texts)
        except Exception as e:
            rospy.logwarn(f"Failed to publish initial label set: {e}")

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
                self._publish_label_set(self.texts)
            except Exception as e:
                rospy.logerr(f"Failed to apply prompts: {e}")

        # Other runtime params
        self.score_threshold = config.score_threshold
        self.top_k = config.top_k
        self.use_amp = config.use_amp
        self.visualize = config.visualize
        # Palette params
        palette_changed = config.palette_lightness != getattr(
            self, "palette_lightness", None
        ) or config.palette_chroma != getattr(self, "palette_chroma", None)
        self.palette_lightness = config.palette_lightness
        self.palette_chroma = config.palette_chroma

        # Visualization params
        self.hud_font_scale = config.hud_font_scale
        self.bbox_thickness = config.bbox_thickness
        self.label_font_scale = config.label_font_scale
        self.label_text_thickness = config.label_text_thickness
        # HUD visibility params
        self.show_detector_hud = config.show_detector_hud
        self.show_tagger_hud = config.show_tagger_hud
        self.show_scene_hud = config.show_scene_hud
        self.show_zupt_hud = config.show_zupt_hud

        # Diagnostics params
        diag_enable = bool(config.publish_diagnostics)
        diag_rate = float(config.diagnostics_rate_hz) if config.diagnostics_rate_hz > 0 else 2.0
        self.diagnostics_stale_sec = float(config.diagnostics_stale_sec)
        self.detector_warn_ms = float(config.detector_warn_ms)
        self.detector_error_ms = float(config.detector_error_ms)
        self.tagger_warn_ms = float(config.tagger_warn_ms)
        self.tagger_error_ms = float(config.tagger_error_ms)

        # ZUPT params
        self.zupt_enable = bool(config.zupt_enable)
        self.zupt_threshold = float(config.zupt_threshold)
        self.zupt_min_interval_sec = float(config.zupt_min_interval_sec)
        self.zupt_downscale_size = int(config.zupt_downscale_size)
        self.zupt_republish_cached = bool(config.zupt_republish_cached)

        # Start/stop or adjust diagnostics timer based on settings
        if diag_enable and (not self._diag_enabled or abs(diag_rate - self._diag_rate_hz) > 1e-6):
            if self._diag_timer is not None:
                try:
                    self._diag_timer.shutdown()
                except Exception:
                    pass
            period = max(0.01, 1.0 / max(1e-3, diag_rate))
            self._diag_timer = rospy.Timer(rospy.Duration(period), self._diagnostics_timer_cb)
            self._diag_enabled = True
            self._diag_rate_hz = diag_rate
            rospy.loginfo_throttle(
                5.0, f"Diagnostics enabled at {diag_rate:.2f} Hz on ~diagnostics"
            )
        elif not diag_enable and self._diag_enabled:
            if self._diag_timer is not None:
                try:
                    self._diag_timer.shutdown()
                except Exception:
                    pass
                self._diag_timer = None
            self._diag_enabled = False
            rospy.loginfo_throttle(5.0, "Diagnostics disabled")

        # Recreate annotators with updated settings
        self._rebuild_color_palette()

        # If only palette changed (and labels stayed the same), publish updated label set
        if palette_changed and not texts_changed:
            try:
                self._publish_label_set(self.texts)
            except Exception as e:
                rospy.logerr(f"Failed to publish updated label set: {e}")

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
                # ZUPT: Check if image is unchanged to skip expensive tagger request
                should_skip_tagger = False
                if getattr(self, "zupt_enable", False):
                    if self.zupt_tagger_reference_image is not None:
                        score = self._compute_image_similarity(
                            self.zupt_tagger_reference_image, img
                        )
                        threshold = getattr(self, "zupt_threshold", 3.0)
                        if score <= threshold:
                            should_skip_tagger = True
                            self.zupt_tagger_requests_skipped += 1
                            rospy.logdebug_throttle(10.0, f"ZUPT: skip tagger (score={score:.1f})")

                if should_skip_tagger:
                    # Skip tagger request, sleep and continue
                    period = 1.0 / max(1e-6, getattr(self, "tagger_fps", 1.0))
                    elapsed = time.time() - start
                    sleep_t = max(0.0, period - elapsed)
                    self.tagger_stop_event.wait(timeout=sleep_t)
                    continue

                # Update tagger reference image
                self.zupt_tagger_reference_image = img.copy()

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
                            self.last_tagger_stamp = rospy.Time.now()
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

    def _publish_label_set(self, texts):
        """
        Publish the current label set and optional color palette as a single
        LabelSet message on a latched topic.

        The internal 'texts' structure is a list of [text] rows plus a sentinel
        [" "]. The sentinel is omitted from the published labels and palette.
        """
        try:
            # Increment label set version id and use a shared timestamp
            self.label_set_id = getattr(self, "label_set_id", 0) + 1
            now = rospy.Time.now()

            # Build published label list (exclude sentinel " ")
            labels = [
                row[0]
                for row in texts
                if isinstance(row, list) and len(row) > 0 and row[0].strip() != ""
            ]

            # Build color palette corresponding to current labels using the same
            # generator as the annotator. Filter out the sentinel entry to keep
            # indices aligned for consumers.
            n_full = len(texts) if isinstance(texts, list) else 1
            full_hex = self._generate_oklch_palette_hex(n_full)
            colors_hex = []
            for i, row in enumerate(texts if isinstance(texts, list) else []):
                if isinstance(row, list) and len(row) > 0 and row[0].strip() != "":
                    hx = full_hex[i]
                    colors_hex.append(hx)

            label_set_msg = LabelSet()
            label_set_msg.stamp = now
            label_set_msg.id = self.label_set_id
            label_set_msg.labels = labels
            label_set_msg.colors_hex = colors_hex
            self.label_set_pub.publish(label_set_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to publish label set: {e}")

    def _diagnostics_timer_cb(self, _event):
        try:
            now = rospy.Time.now()
            arr = DiagnosticArray()
            arr.header.stamp = now

            # Detector status
            det_stat = DiagnosticStatus()
            det_stat.name = "yolo_world_ros/Detector"
            det_stat.hardware_id = str(self.device)
            # Determine staleness
            det_stamp = getattr(self, "last_detector_stamp", None)
            det_lat = getattr(self, "last_detector_latency_ms", None)
            stale = det_stamp is None or (now - det_stamp).to_sec() > float(
                getattr(self, "diagnostics_stale_sec", 2.0)
            )
            if stale or det_lat is None:
                det_stat.level = DiagnosticStatus.STALE
                det_stat.message = "No recent detector update"
            else:
                if det_lat < float(self.detector_warn_ms):
                    det_stat.level = DiagnosticStatus.OK
                    det_stat.message = "OK"
                elif det_lat < float(self.detector_error_ms):
                    det_stat.level = DiagnosticStatus.WARN
                    det_stat.message = "High latency"
                else:
                    det_stat.level = DiagnosticStatus.ERROR
                    det_stat.message = "Very high latency"
            kv = KeyValue()
            kv.key = "latency_ms"
            kv.value = f"{det_lat:.2f}" if det_lat is not None else "NaN"
            det_stat.values.append(kv)
            arr.status.append(det_stat)

            # Tagger status (only if enabled)
            if getattr(self, "prompt_source", None) == 2:
                tag_stat = DiagnosticStatus()
                tag_stat.name = "yolo_world_ros/Tagger"
                tag_stat.hardware_id = str(self.device)
                tag_stamp = getattr(self, "last_tagger_stamp", None)
                tag_lat = getattr(self, "last_tagger_latency_ms", None)
                stale_t = tag_stamp is None or (now - tag_stamp).to_sec() > float(
                    getattr(self, "diagnostics_stale_sec", 2.0)
                )
                if stale_t or tag_lat is None:
                    tag_stat.level = DiagnosticStatus.STALE
                    tag_stat.message = "No recent tagger update"
                else:
                    if tag_lat < float(self.tagger_warn_ms):
                        tag_stat.level = DiagnosticStatus.OK
                        tag_stat.message = "OK"
                    elif tag_lat < float(self.tagger_error_ms):
                        tag_stat.level = DiagnosticStatus.WARN
                        tag_stat.message = "High latency"
                    else:
                        tag_stat.level = DiagnosticStatus.ERROR
                        tag_stat.message = "Very high latency"
                kv2 = KeyValue()
                kv2.key = "latency_ms"
                kv2.value = f"{tag_lat:.2f}" if tag_lat is not None else "NaN"
                tag_stat.values.append(kv2)
                arr.status.append(tag_stat)

            # ZUPT status (only if enabled)
            if getattr(self, "zupt_enable", False):
                zupt_stat = DiagnosticStatus()
                zupt_stat.name = "yolo_world_ros/ZeroUpdate"
                zupt_stat.hardware_id = str(self.device)
                zupt_stat.level = DiagnosticStatus.OK
                zupt_stat.message = "Active"

                total = getattr(self, "zupt_frames_total", 0)
                skipped = getattr(self, "zupt_frames_skipped", 0)
                skip_rate = (skipped / total * 100.0) if total > 0 else 0.0

                kv_skip_rate = KeyValue()
                kv_skip_rate.key = "skip_rate_percent"
                kv_skip_rate.value = f"{skip_rate:.1f}"
                zupt_stat.values.append(kv_skip_rate)

                kv_skipped = KeyValue()
                kv_skipped.key = "frames_skipped"
                kv_skipped.value = str(skipped)
                zupt_stat.values.append(kv_skipped)

                kv_total = KeyValue()
                kv_total.key = "frames_total"
                kv_total.value = str(total)
                zupt_stat.values.append(kv_total)

                kv_score = KeyValue()
                kv_score.key = "last_similarity_score"
                kv_score.value = f"{getattr(self, 'zupt_last_similarity_score', 0.0):.2f}"
                zupt_stat.values.append(kv_score)

                # Tagger skip stats (if auto mode)
                if getattr(self, "prompt_source", None) == 2:
                    kv_tagger_skipped = KeyValue()
                    kv_tagger_skipped.key = "tagger_requests_skipped"
                    kv_tagger_skipped.value = str(getattr(self, "zupt_tagger_requests_skipped", 0))
                    zupt_stat.values.append(kv_tagger_skipped)

                arr.status.append(zupt_stat)

            # Publish if we have at least detector status
            if len(arr.status) > 0:
                self.diagnostics_pub.publish(arr)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Diagnostics publish failed: {e}")

    @staticmethod
    def _texts_equal(a, b):
        try:
            return a == b
        except Exception:
            return False

    # Zero-Update (ZUPT) helper methods
    def _compute_image_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Mean Absolute Difference (MAD) on downsampled grayscale images.

        Args:
            img1: First BGR image
            img2: Second BGR image

        Returns:
            MAD score (0 = identical, higher = more different)
        """
        size = getattr(self, "zupt_downscale_size", 64)
        gray1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (size, size))
        gray2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (size, size))
        return float(np.mean(np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))))

    def _should_skip_inference(self, cv_image: np.ndarray) -> Tuple[bool, str]:
        """
        Check if inference should be skipped due to image stationarity.

        Args:
            cv_image: Current BGR image

        Returns:
            Tuple of (should_skip, reason_string)
        """
        if not getattr(self, "zupt_enable", False):
            return False, "disabled"

        with self.zupt_reference_image_lock:
            ref = self.zupt_reference_image

        if ref is None:
            return False, "no_reference"

        # Invalidate cache on label set change
        if self.zupt_cached_label_set_id != self.label_set_id:
            return False, "label_set_changed"

        # Force periodic inference after min_interval
        elapsed = time.time() - self.zupt_last_inference_time
        if elapsed >= getattr(self, "zupt_min_interval_sec", 1.0):
            return False, "min_interval"

        # No cached results to republish
        if self.zupt_cached_detections is None:
            return False, "no_cache"

        # Compute similarity
        score = self._compute_image_similarity(ref, cv_image)
        self.zupt_last_similarity_score = score

        threshold = getattr(self, "zupt_threshold", 3.0)
        if score <= threshold:
            return True, f"stationary({score:.1f})"
        return False, f"changed({score:.1f})"

    def _draw_zupt_indicator(self, cached_img_msg: Image, header) -> Image:
        """
        Draw ZUPT indicator on a cached annotated image.

        Args:
            cached_img_msg: Cached sensor_msgs/Image to annotate
            header: New header to apply to the output message

        Returns:
            New Image message with ZUPT indicator drawn (if show_zupt_hud is True)
        """
        if not getattr(self, "show_zupt_hud", True):
            # Just update header without drawing
            out_msg = copy.deepcopy(cached_img_msg)
            out_msg.header = header
            return out_msg

        # Decode cached image
        img_data = np.frombuffer(cached_img_msg.data, dtype=np.uint8).reshape(
            cached_img_msg.height, cached_img_msg.width, -1
        )
        frame = img_data.copy()

        # Draw ZUPT indicator using Pillow
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_size = max(10, int(18 * self.hud_font_scale))
        font = None
        if os.path.isfile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        else:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        # Draw "CACHED" indicator at bottom-left in yellow
        zupt_text = "CACHED"
        try:
            bbox = draw.textbbox((0, 0), zupt_text, font=font)
            text_h = bbox[3] - bbox[1]
        except Exception:
            _, text_h = draw.textsize(zupt_text, font=font)
        x = 10
        y = pil_img.height - text_h - 10
        draw.text((x, y), zupt_text, font=font, fill=(255, 255, 0))

        # Convert back to BGR
        annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Create output Image message
        out_msg = Image()
        out_msg.header = header
        out_msg.height = annotated_frame.shape[0]
        out_msg.width = annotated_frame.shape[1]
        out_msg.encoding = "bgr8"
        out_msg.is_bigendian = 0
        out_msg.step = annotated_frame.shape[1] * 3
        out_msg.data = annotated_frame.tobytes()
        return out_msg

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
        self.bbox_annotator = sv.BoundingBoxAnnotator(
            color=self.color_palette,
            thickness=self.bbox_thickness,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=self.color_palette,
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
                    self._publish_label_set(self.texts)
                    self._rebuild_color_palette()
                except Exception as e:
                    rospy.logerr(f"Failed to apply auto prompts: {e}")

        # ZUPT: Check for image stationarity before inference
        self.zupt_frames_total += 1
        should_skip, skip_reason = self._should_skip_inference(cv_image)

        if should_skip:
            self.zupt_frames_skipped += 1
            rospy.logdebug_throttle(5.0, f"ZUPT: skip ({skip_reason})")

            if getattr(self, "zupt_republish_cached", True) and self.zupt_cached_detections:
                # Republish cached detections with updated timestamp
                cached_det = copy.deepcopy(self.zupt_cached_detections)
                cached_det.header = msg.header
                cached_det.header.frame_id = f"yolo_world_set:{self.label_set_id}"
                self.detection_pub.publish(cached_det)

                # Republish cached annotated image if visualization enabled
                if self.visualize and self.zupt_cached_annotated_image is not None:
                    cached_img = self._draw_zupt_indicator(
                        self.zupt_cached_annotated_image, msg.header
                    )
                    self.annotated_image_pub.publish(cached_img)
            return

        # Update reference image for next comparison
        with self.zupt_reference_image_lock:
            self.zupt_reference_image = cv_image.copy()
        self.zupt_last_inference_time = time.time()

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
        # Track detector latency for diagnostics
        self.last_detector_latency_ms = float(detector_latency_ms)
        self.last_detector_stamp = rospy.Time.now()

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
            annotated_frame = self.bbox_annotator.annotate(
                scene=cv_image.copy(),
                detections=detections,
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )

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
            if self.show_detector_hud:
                detector_text = f"Detector: {detector_latency_ms:.2f} ms"
                _draw.text((10, 30), detector_text, font=_font, fill=(255, 0, 0))

            if getattr(self, "prompt_source", None) == 2 and self.show_tagger_hud:
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
                self.show_scene_hud
                and isinstance(getattr(self, "last_scene_text", None), str)
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
        detection_array.header.frame_id = f"yolo_world_set:{self.label_set_id}"

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

        # ZUPT: Cache results for potential reuse when skipping inference
        if getattr(self, "zupt_enable", False):
            self.zupt_cached_detections = copy.deepcopy(detection_array)
            self.zupt_cached_label_set_id = self.label_set_id
            if self.visualize:
                self.zupt_cached_annotated_image = copy.deepcopy(annotated_image_msg)


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
