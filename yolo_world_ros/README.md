# YOLO‑World ROS 1 Wrapper

A ROS 1 package that wraps the YOLO‑World open‑vocabulary object detector and publishes detections as `vision_msgs/Detection2DArray` with an optional annotated `sensor_msgs/Image` stream.

This package is intended to live inside the YOLO‑World repository tree and uses `dynamic_reconfigure` to update the vocabulary and runtime inference settings without restarting the node.

- Subscribes: `sensor_msgs/Image` (`bgr8` or `rgb8`)
- Publishes: `vision_msgs/Detection2DArray`
- Publishes (optional): `sensor_msgs/Image` with bounding boxes and labels
- Publishes: `yolo_world_ros/LabelSet` (current effective labels and optional color palette)
- Publishes (optional): `~diagnostics` (`diagnostic_msgs/DiagnosticArray`) with detector/tagger latency
- Dynamic params: prompt source (manual/file/auto), tagger_url/fps/timeout, text prompts, score threshold, top‑k, AMP, visualization, palette and HUD settings

This README reflects the current node implementation in `yolo_world_ros/nodes/yolo_world_node.py`.

---

## Quick start

1. **Install ROS 1 and YOLO‑World**

   - ROS 1 Noetic on Ubuntu 20.04 is recommended.
   - Clone the YOLO‑World repository into a catkin workspace `src` folder, so that `yolo_world_ros` is inside the YOLO‑World repo and the repo is a valid Git checkout (used by the node to locate the repo root).

2. **Build in a catkin workspace**

   From your catkin workspace root:

   ```bash
   catkin_make
   source devel/setup.bash
   ```

3. **Publish a test image**

   ```bash
   rosrun image_publisher image_publisher /path/to/image.jpg
   ```

4. **Launch the node (adjust config and checkpoint if needed)**

   ```bash
   roslaunch yolo_world_ros yolo_world.launch input_image_topic:="$(rostopic list | grep '^/image_publisher_.*image_raw$')"
   ```

5. **View results**

   ```bash
   rosrun rqt_image_view rqt_image_view
   ```

   Select `/yolo_world/annotated_image`, or inspect detections:

   ```bash
   rostopic echo /yolo_world/detections
   ```

---

## Launch and parameters

Default launch file: `yolo_world_ros/launch/yolo_world.launch`

```xml
<node name="yolo_world_ros"
      pkg="yolo_world_ros"
      type="yolo_world_node.py"
      output="screen">
  <!-- config_file, checkpoint_file, device, topics ... -->
</node>
```

Parameters (can be overridden on the command line):

- `config_file`: Path to a YOLO‑World config (`.py`). Defaults to a v2_x config under `../configs/pretrain/...`.
- `checkpoint_file`: Path to model weights (`.pth`) under `../weights/...`.
- `device`: Inference device string, e.g., `cuda:0` or `cpu`.
- `input_image_topic`: `sensor_msgs/Image` topic to subscribe (default `/camera/rgb/image_raw`).
- `output_detections_topic`: `vision_msgs/Detection2DArray` (default `/yolo_world/detections`).
- `annotated_image_topic`: `sensor_msgs/Image` with overlays (default `/yolo_world/annotated_image`).

Example overrides:

```bash
roslaunch yolo_world_ros yolo_world.launch device:=cpu \
  config_file:=$(rospack find yolo_world_ros)/../configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
  checkpoint_file:=$(rospack find yolo_world_ros)/../weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth
```

---

## Dynamic reconfigure

The node uses `dynamic_reconfigure` with the configuration defined in `yolo_world_ros/cfg/YOLOWorld.cfg`.

Run:

```bash
rosrun rqt_reconfigure rqt_reconfigure
```

Select the node (`yolo_world_ros_node` or `yolo_world_ros`).

### Prompt and tagger parameters

Internally, YOLO‑World uses **text prompts** (e.g., `"person"`, `"car"`, `"dog"`) to define the open‑vocabulary label set. The dynamic reconfigure interface exposes these as “prompts”, but the ROS messages and topics use the term “label” for clarity.

- `prompt_source` (int):  
  - `0` = manual (`text_prompts`)  
  - `1` = file (`text_prompt_file`)  
  - `2` = auto (VLM tagger)
- `text_prompts` (str): Comma‑separated list, e.g., `"person, car, dog"` (used when `prompt_source=0`).
- `text_prompt_file` (str): Path to `.txt` or `.json` prompt file (used when `prompt_source=1`).
  - `.txt`: one label per line.
  - `.json`: list of lists, e.g., `[["person"], ["car"], ...]`.
- `tagger_url` (str): VLM tagger endpoint URL, default `http://localhost:59810/tag`.
- `tagger_fps` (double): Max request rate to the tagger (Hz), default `1.0`.
- `tagger_timeout` (double): HTTP timeout (seconds), default `10.0`.

When `prompt_source` is set to `AUTO` (2), the node periodically sends the latest image to the VLM tagger. The tagger is expected to return JSON with:

```json
{
  "objects": ["person", "car", "dog"],
  "scene": "street"
}
```

- `objects` (list of strings) becomes the active label list (plus an internal sentinel).
- `scene` (string) is shown in the HUD overlay if enabled.

### Inference and visualization parameters

- `score_threshold` (double): Confidence threshold (0..1).
- `top_k` (int): Keep top‑K predictions after thresholding.
- `use_amp` (bool): Enable mixed‑precision inference (PyTorch AMP).
- `visualize` (bool): Publish the annotated image.

### Visualization and palette parameters

These control the on‑image HUD and color palette used for bounding boxes and labels:

- `hud_font_scale` (double): Font scale for overlay metrics text.
- `bbox_thickness` (int): Bounding box line thickness (px).
- `label_font_scale` (double): Font scale for label text.
- `label_text_thickness` (int): Thickness for label text.
- `palette_lightness` (double): OKLCh lightness `L` (0.0–1.0).
- `palette_chroma` (double): OKLCh chroma `C` (0.0–0.5 typical gamut‑safe).

The node uses an OKLCh‑based palette generator to create visually distinct colors for each label. Changing `palette_lightness` or `palette_chroma` will rebuild the palette and republish the label set (including the palette).

### HUD visibility toggles

- `show_detector_hud` (bool): Show detector latency text (e.g., `Detector: 12.34 ms`).
- `show_tagger_hud` (bool): Show tagger latency text when `prompt_source=2`.
- `show_scene_hud` (bool): Show the `scene:` description from the VLM tagger at the top‑right.

### Diagnostics parameters

- `publish_diagnostics` (bool): Publish diagnostic messages on the private `~diagnostics` topic.
- `diagnostics_rate_hz` (double): Publish rate in Hz (default 2.0).
- `diagnostics_stale_sec` (double): Time (s) after which latency is considered stale (default 2.0).
- `detector_warn_ms` / `detector_error_ms` (double): Thresholds for detector latency.
- `tagger_warn_ms` / `tagger_error_ms` (double): Thresholds for tagger latency.

Notes:
- The Tagger diagnostic status is only published when `prompt_source=2` (AUTO).
- Levels: OK (< warn), WARN (< error), ERROR (>= error), STALE (no recent update).

Changes to prompts (manual, file, or auto) trigger an internal model `reparameterize()` call and a republish of the label set and palette.

---

## Topics

### Subscribed

- `input_image_topic` (`sensor_msgs/Image`)
  - Only `rgb8` and `bgr8` encodings are supported.
  - If your camera publishes other encodings (e.g., `mono8`, `compressed`), convert or republish to `rgb8`/`bgr8` first.

### Published

- `output_detections_topic` (`vision_msgs/Detection2DArray`)
  - `header.frame_id` is set to `yolo_world_set:<label_set_id>` where `<label_set_id>` is a monotonically increasing integer whenever the label set changes.
  - Each `Detection2D` includes:
    - `bbox` (`BoundingBox2D`) in image coordinates:
      - `center.x`, `center.y`
      - `size_x`, `size_y`
    - `results[0].id` (int): class index corresponding to the label set order.
    - `results[0].score` (float): confidence score.
  - To get the label string, map `results[0].id` to the corresponding label string from the `label_set` topic.

- `annotated_image_topic` (`sensor_msgs/Image`)
  - Overlay with bounding boxes and `"label score"` text.
  - Uses the OKLCh palette and HUD settings described above.
  - Enabled/disabled via the `visualize` dynamic parameter.

- `~diagnostics` (`diagnostic_msgs/DiagnosticArray`)
  - Private diagnostics topic containing one `DiagnosticStatus` for the detector and, when enabled, one for the tagger.
  - Each status reports `latency_ms` and a level (OK/WARN/ERROR/STALE) based on thresholds and staleness.

- `label_set` (`yolo_world_ros/LabelSet`)
  - Unified message that carries both the current labels and an optional color palette.
  - Fields:
    - `stamp` (`rospy.Time`): timestamp when the label set was published.
    - `id` (uint32): label set identifier (matches the suffix in `yolo_world_set:<id>`).
    - `labels` (`string[]`): current effective label strings (the internal sentinel `" "` entry is omitted).
    - `colors_hex` (`string[]`): per‑label colors as `#RRGGBB` hex strings. May be empty if no palette is provided.
    - `colors_bgr` (`uint8[]`): flattened list of BGR triplets `[b0, g0, r0, b1, g1, r1, ...]` aligned with `labels`. May be empty if no palette is provided.

The `label_set` publisher is latched, so new subscribers immediately receive the latest label set and palette.

---

## Using YOLO‑World labels and palette in downstream nodes (e.g., segmentation)

A common use case is to run a downstream segmentation node that:

- Consumes `Detection2DArray` from YOLO‑World.
- Consumes the `LabelSet` topic.
- Produces segmentation masks with colors that are **visually consistent** with the detection boxes.

### Mapping detections to labels

- Subscribe to `/yolo_world/detections` (`vision_msgs/Detection2DArray`).
- For each `Detection2D`:
  - `detection.results[0].id` is the **class index**.
  - This index corresponds to `label_set.labels[class_id]` from the latest `LabelSet` message.
- The detection array header encodes the label set id:

  ```text
  detection_array.header.frame_id == "yolo_world_set:<id>"
  ```

  where `<id>` matches `LabelSet.id`. This allows you to correlate detections with the correct label set if labels change over time.

### Using the shared color palette

- Subscribe to `label_set` (`yolo_world_ros/LabelSet`):

  ```python
  from yolo_world_ros.msg import LabelSet

  current_label_set = None

  def label_set_cb(msg: LabelSet):
      global current_label_set
      current_label_set = msg
  ```

- The palette is optional:
  - If `colors_hex` and `colors_bgr` are non‑empty and consistent with `labels`, you can use them.
  - If they are empty, fall back to your own palette.

- Example: build a list of BGR tuples aligned with `labels`:

  ```python
  def build_bgr_palette(label_set_msg: LabelSet):
      n = len(label_set_msg.labels)
      if len(label_set_msg.colors_bgr) != 3 * n:
          # Palette missing or inconsistent; return None to signal fallback
          return None

      colors = [
          (
              label_set_msg.colors_bgr[3 * i + 0],  # B
              label_set_msg.colors_bgr[3 * i + 1],  # G
              label_set_msg.colors_bgr[3 * i + 2],  # R
          )
          for i in range(n)
      ]
      return colors
  ```

- In your segmentation node, you can then use:

  ```python
  class_id = detection.results[0].id
  label = current_label_set.labels[class_id]
  color = colors[class_id]  # BGR tuple
  ```

  to color your segmentation masks consistently with the detection boxes.

### Handling label set changes

- `LabelSet.id` increments whenever the label set changes (manual/file/auto prompts).
- `Detection2DArray.header.frame_id` is set to `yolo_world_set:<id>` with the same id.
- A downstream node can:

  1. Cache the latest `LabelSet` by `id`.
  2. When processing a `Detection2DArray`, parse `frame_id` to extract the id.
  3. Ensure it has the matching `LabelSet` (topics are latched, so it should already have it).
  4. If the id does not match the cached one, update its internal mapping.

This ensures that even if labels change at runtime, detections and segmentation masks remain consistent with the correct label set.

### Example architecture for a segmentation node

1. Subscriptions:
   - `/yolo_world/detections` (`vision_msgs/Detection2DArray`)
   - `/yolo_world/label_set` (`yolo_world_ros/LabelSet`)
   - Your own image or feature topics as needed.

2. Internal state:
   - `current_label_set_id`
   - `current_labels`
   - `current_colors` (BGR palette or `None`)

3. On `LabelSet`:
   - Update `current_label_set_id = msg.id`
   - Update `current_labels = msg.labels`
   - Build `current_colors` from `msg.colors_bgr` (or set to `None` if empty/inconsistent).

4. On `Detection2DArray`:
   - Optionally parse `header.frame_id` to extract the label set id and check against `current_label_set_id`.
   - For each detection:
     - `class_id = detection.results[0].id`
     - `label = current_labels[class_id]`
     - `color = current_colors[class_id]` if available, otherwise use a fallback palette.
   - Use `color` to render segmentation masks or overlays.

This pattern gives you **visual coherency** between detection boxes and segmentation masks while allowing the label set to change dynamically.

---

## Prompt files

- **Text file (`.txt`)**: one label per line.

  Example:

  ```text
  person
  car
  dog
  ```

- **JSON file (`.json`)**: top‑level list where each item is a list and `item[0]` is the label string.

  Example:

  ```json
  [["person"], ["car"], ["dog"]]
  ```

The default `text_prompt_file` in `YOLOWorld.cfg` is:

```text
data/texts/gpt_indoor_general.json
```

You can override this via dynamic reconfigure or by setting the parameter at launch.

---

## Auto labels via VLM tagger

When `prompt_source` is set to `AUTO` (2):

- A background thread periodically:
  - Grabs the latest BGR image received on `input_image_topic`.
  - JPEG‑encodes it.
  - Sends it as `multipart/form-data` to `tagger_url` with key `"image"`.
- The tagger is expected to respond with JSON containing at least an `"objects"` list and optionally a `"scene"` string.
- If the returned `objects` differ from the current labels, the node:
  - Updates `self.texts` (internal prompt representation).
  - Calls `model.reparameterize(self.texts)`.
  - Rebuilds the color palette.
  - Publishes updated `LabelSet`.

The tagger loop runs at up to `tagger_fps` Hz and uses `tagger_timeout` as the HTTP timeout. Failures are logged with throttling to avoid log spam.

---

## Notes and tips

- **Throughput and latency**
  - Reduce input image resolution upstream.
  - Increase `score_threshold` or lower `top_k` to reduce the number of boxes.
  - Enable `use_amp` on modern GPUs for faster inference.
  - Disable `visualize` if you only need detections; this avoids the cost of drawing overlays.
  - Enable `publish_diagnostics` to stream detector/tagger latency on `~diagnostics` for monitoring without HUD overlays.

- **Image encoding**
  - Only `bgr8` and `rgb8` are supported.
  - Use `image_transport` or a small helper node to convert from other encodings.

- **Label management**
  - Use manual prompts for quick experiments.
  - Use file prompts for reproducible setups and large vocabularies.
  - Use auto prompts with a VLM tagger to adapt to the scene dynamically.
  - Downstream nodes should rely on `LabelSet` for the authoritative label list and palette.

- **Repository root**
  - On startup, the node uses GitPython to locate the repository root and `chdir` there. This ensures relative paths in the YOLO‑World config and weights work as expected.
  - Make sure the `yolo_world_ros` package is inside a Git clone of the YOLO‑World repository.

---

## License

This ROS wrapper follows the YOLO‑World project licensing.

- YOLO‑World core code and models are developed by Tencent/AILab‑CVC and are licensed under the terms specified in the main YOLO‑World repository (e.g., GPLv3.0 for code, plus any model‑specific terms).
- Ensure you comply with both the code license and any model usage restrictions when deploying this node.

---

## Acknowledgements

Thanks to:

- YOLO‑World authors and the OpenMMLab ecosystem (MMEngine, MMDetection).
- The ROS community and the maintainers of `vision_msgs`.
- Contributors to the OKLCh color and visualization utilities used for the palette and HUD.
