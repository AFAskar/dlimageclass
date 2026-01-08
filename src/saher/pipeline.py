from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
import pytesseract
from PIL import Image
import numpy as np
import json
from typing import Any

license_plate_detection_model = YOLO("license-plate-finetune-v1s.pt")

car_detection_model = YOLO("yolo11n.pt")

violation_detection_model = YOLO("seatbelt.pt")
VIOLATION_CLASS_NAMES = [
    "1",
    "2",
    "No-helmet",
    "Np",
    "bike",
    "car",
    "helmet",
    "mobile",
    "no-helmet",
    "person-noseatbelt",
    "person-seatbelt",
    "seatbelt",
]


def detect_cars(image_paths: list[Path] | list[np.ndarray]) -> list[Results]:
    # First, detect cars in the image
    car_results = car_detection_model.predict(
        source=image_paths, classes=[2, 5, 7], verbose=False
    )  # COCO classes for car, bus, truck
    return car_results


def get_Results_images(results: list[Results]) -> list[np.ndarray]:
    images = []
    for result in results:
        if not result.boxes:
            continue
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            image = result.orig_img[y1:y2, x1:x2]
            images.append(image)
    return images


def get_Results_orig_images(results: list[Results]) -> list[np.ndarray]:
    images = []
    # Get original images from results but only those that have detections
    for result in results:
        if result.boxes:
            images.append(result.orig_img)
    return images


def detect_violations(car_images: list[Path] | list[np.ndarray]) -> list[Results]:
    if not car_images:
        return []
    violations = violation_detection_model.predict(
        source=car_images, classes=[7, 9, 10, 11], verbose=False
    )
    return violations


def plate_detection(belt_images: list[np.ndarray]) -> list[Results]:
    if not belt_images:
        return []
    plate_results = license_plate_detection_model.predict(
        source=belt_images, verbose=False
    )
    return plate_results


def run_ocr_on_plate(plate_images: list[np.ndarray]) -> list[str]:
    if not plate_images:
        return []
    ocr_results = []
    for plate in plate_images:
        # Convert numpy array to PIL Image for pytesseract
        pil_image = Image.fromarray(plate)
        ocr_result = pytesseract.image_to_string(pil_image, config="--psm 7")
        ocr_results.append(ocr_result.strip())
    return ocr_results


def get_violation_type(boxes: Any) -> list[str]:
    violations = []
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = VIOLATION_CLASS_NAMES[class_id]
        if class_name == "person-noseatbelt":
            violations.append("person-noseatbelt")
        elif class_name == "mobile":
            violations.append("mobile")
    return violations


def clean_violations(violation_results: list[Results]) -> list[Results]:
    cleaned_results = []
    for result in violation_results:
        boxes = result.boxes
        if not boxes:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = VIOLATION_CLASS_NAMES[class_id]
            if class_name in ["person-noseatbelt", "mobile"]:
                cleaned_results.append(result)
    return cleaned_results


def run_pipeline(images_path: list[Path] | list[np.ndarray]):
    car_results = detect_cars(images_path)
    car_images = get_Results_images(car_results)
    violation_results = detect_violations(car_images)
    violation_results = clean_violations(violation_results)
    violation_images = get_Results_orig_images(violation_results)
    plate_results = plate_detection(violation_images)
    plate_images = get_Results_images(plate_results)
    ocr_results = run_ocr_on_plate(plate_images)
    return violation_results, plate_results, ocr_results


def json_output(
    violation_results: list[Results],
    plate_results: list[Results],
    ocr_results: list[str],
):
    # output violation type, bboxes of violation, license plate of violator and ocr result of plate
    output = []
    plate_images = get_Results_images(plate_results)
    for i in range(len(ocr_results)):
        plate_image = plate_images[i]
        plate_image_path = f"plate_{i}.png"
        Image.fromarray(plate_image).save(plate_image_path)
        violation_image = get_Results_images([violation_results[i]])[0]
        violation_image_path = f"violation_{i}.png"
        Image.fromarray(violation_image).save(violation_image_path)
        if not violation_results[i].boxes:
            continue
        boxes = violation_results[i].boxes
        output_item = {
            ocr_results[i]: {
                "violation_type": get_violation_type(boxes),
                "violation_bbox": boxes.summary(),
                "violation_image": violation_image_path,
                "license_plate": plate_image_path,
            }
        }
        output.append(output_item)
    if output:
        json.dump(output, open("pipeline_output.json", "w"))
    return output
