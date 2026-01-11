# TrafficVision

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

TrafficVision is an AI-powered traffic violation detection system inspired by Saher. It uses a sophisticated multi-stage YOLO pipeline to automatically detect seatbelt and mobile phone violations, combined with license plate detection and OCR for violator identification.

## Features

- **Multi-Stage Detection Pipeline**

  - Vehicle detection (cars, buses, trucks)
  - Violation detection (seatbelt, mobile phone usage)
  - License plate detection
  - OCR text extraction from license plates

- **Multiple Violation Types**

  - Person without seatbelt detection
  - Mobile phone usage detection
  - Helmet violations (for motorcycles)

- **Flexible Usage**

  - Command-line interface (CLI) for batch processing
  - Web interface powered by Gradio
  - Installable via `uvx` for one-off usage

- **Output Options**
  - Save violation images with bounding boxes
  - Save license plate images
  - JSON output with violation details
  - OCR text extraction

## Installation

### Quick Start with uvx (No Installation Required)

Process a single image:

```sh
uvx git+https://github.com/AFAskar/trafficvision path/to/image.jpg
```

Launch the web interface:

```sh
uvx --from git+https://github.com/AFAskar/trafficvision trafficvision-web
```

### Install from GitHub

```sh
pip install git+https://github.com/AFAskar/trafficvision
```

### Local Development

```sh
git clone https://github.com/AFAskar/trafficvision.git
cd trafficvision
pip install -e .
```

## Usage

### Command Line Interface

Basic usage:

```sh
trafficvision path/to/image.jpg
```

Process a directory of images:

```sh
trafficvision path/to/images/
```

Save all outputs (violation images and license plates):

```sh
trafficvision path/to/image.jpg --save
```

Save only license plate images:

```sh
trafficvision path/to/image.jpg --save-plates
```

Save only violation images with bounding boxes:

```sh
trafficvision path/to/image.jpg --save-violations
```

### Web Interface

Launch the Gradio web interface:

```sh
trafficvision-web
```

Then open your browser to the provided URL (typically `http://127.0.0.1:7860`)

### Python API

```python
from pathlib import Path
from trafficvision.pipeline import run_pipeline, get_violation_type

# Run the detection pipeline
images = [Path("path/to/image.jpg")]
violation_results, plate_results, ocr_results = run_pipeline(images)

# Get violation types
for violation in violation_results:
    violation_types = get_violation_type(violation.boxes)
    print(f"Detected violations: {violation_types}")

# Get OCR text from license plates
for ocr_text in ocr_results:
    print(f"License plate text: {ocr_text}")
```

## How It Works

TrafficVision uses a multi-stage detection pipeline:

1. **Vehicle Detection**: Uses YOLOv11n to detect vehicles (cars, buses, trucks) in the input image
2. **Violation Detection**: Analyzes detected vehicles for traffic violations using a custom-trained YOLO model
3. **License Plate Detection**: Locates license plates on vehicles with detected violations using a fine-tuned YOLO model
4. **OCR Processing**: Extracts text from license plates using Tesseract OCR

## Models

The system uses three pre-trained YOLO models:

- `yolo11n.pt` - YOLOv11 Nano for vehicle detection
- `seatbelt.pt` - Custom-trained model for violation detection
- `license-plate-finetune-v1s.pt` - Fine-tuned model for license plate detection

## Requirements

- Python 3.13+
- PyTorch
- Ultralytics YOLO
- Tesseract OCR
- Gradio (for web interface)
- Other dependencies listed in `pyproject.toml`

## Dataset

The seatbelt detection model was trained on the [Seat Belt Detection dataset](https://universe.roboflow.com/farishijazi/seat_belt_detection-78bmx) from Roboflow (CC BY 4.0 license).

## Output Format

### CLI Output

```
Violation 1:
 - Violation Types: ['person-noseatbelt', 'mobile']
 - OCR Text: ABC 1234
```

### JSON Output

The system can generate JSON output with the following structure:

```json
[
  {
    "ABC 1234": {
      "violation_type": ["person-noseatbelt"],
      "violation_bbox": [...],
      "violation_image": "violation_0.png",
      "license_plate": "plate_0.png"
    }
  }
]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the Saher traffic violation detection system
- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Dataset from [Roboflow Universe](https://universe.roboflow.com/)
- License Plate Detection Model from [morsetechlab](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/blob/main/license-plate-finetune-v1s.pt)
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract)

## Author

AFAskar

## Version

Current version: 0.1.13
