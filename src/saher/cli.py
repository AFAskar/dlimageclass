import typer
from pathlib import Path
from typing import Optional
import json
from rich.console import Console
from rich.table import Table
from PIL import Image
import cv2
import numpy as np

from saher.pipeline import run_pipeline, VIOLATION_CLASS_NAMES

app = typer.Typer(help="Saher Traffic Violation Detection CLI")
console = Console()


@app.command()
def detect(
    image_path: Path = typer.Argument(
        ...,
        help="Path to the image file to analyze",
        exists=True,
        dir_okay=False,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to save output images and results",
    ),
    save_annotated: bool = typer.Option(
        True,
        "--annotated/--no-annotated",
        help="Save annotated images with bounding boxes",
    ),
    save_plates: bool = typer.Option(
        True,
        "--plates/--no-plates",
        help="Save extracted license plate images",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Save results as JSON file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
):
    """
    Detect traffic violations in an image.

    Analyzes the image for cars, detects violations (no seatbelt, mobile phone use),
    extracts license plates, and performs OCR on the plates.
    """
    console.print(f"[cyan]Analyzing image:[/cyan] {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run pipeline
    with console.status("[bold green]Running detection pipeline..."):
        violation_results, plate_images, ocr_results = run_pipeline([image])

    if not violation_results or not plate_images:
        console.print("[yellow]No violations detected in the image.[/yellow]")
        raise typer.Exit(0)

    # Setup output directory
    if output_dir is None:
        output_dir = Path("saher_output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Process results
    console.print(f"\n[green]✓ Found {len(violation_results)} violation(s)[/green]\n")

    # Create results table
    table = Table(title="Violation Detection Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("License Plate", style="magenta")
    table.add_column("Violations", style="red")
    table.add_column("Confidence", style="yellow")

    results_data = []

    for i, (result, plate_img, ocr_text) in enumerate(
        zip(violation_results, plate_images, ocr_results)
    ):
        if not result.boxes:
            continue

        # Collect violation types and confidences
        violations = []
        confidences = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = VIOLATION_CLASS_NAMES[class_id]
            confidence = float(box.conf[0])

            if class_name in ["person-noseatbelt", "mobile"]:
                violations.append(class_name)
                confidences.append(confidence)

        if not violations:
            continue

        # Add to table
        violation_str = ", ".join(violations)
        conf_str = ", ".join([f"{c:.2%}" for c in confidences])
        table.add_row(
            str(i + 1), ocr_text if ocr_text else "N/A", violation_str, conf_str
        )

        # Save annotated image
        if save_annotated:
            annotated_image = result.plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            annotated_path = output_dir / f"violation_{i+1}_annotated.png"
            Image.fromarray(annotated_image).save(annotated_path)

            if verbose:
                console.print(f"  [dim]Saved annotated image: {annotated_path}[/dim]")

        # Save plate image
        if save_plates:
            plate_path = output_dir / f"plate_{i+1}.png"
            Image.fromarray(plate_img).save(plate_path)

            if verbose:
                console.print(f"  [dim]Saved plate image: {plate_path}[/dim]")

        # Collect data for JSON
        results_data.append(
            {
                "id": i + 1,
                "license_plate": ocr_text,
                "violations": [
                    {"type": v, "confidence": float(c)}
                    for v, c in zip(violations, confidences)
                ],
                "annotated_image": (
                    str(output_dir / f"violation_{i+1}_annotated.png")
                    if save_annotated
                    else None
                ),
                "plate_image": (
                    str(output_dir / f"plate_{i+1}.png") if save_plates else None
                ),
            }
        )

    # Display table
    console.print(table)

    # Save JSON
    if json_output:
        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results_data, f, indent=2)
        console.print(f"\n[green]✓ Results saved to:[/green] {json_path}")

    console.print(f"\n[green]✓ Output saved to:[/green] {output_dir}")


@app.command()
def batch(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing images to analyze",
        exists=True,
        file_okay=False,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to save output images and results",
    ),
    pattern: str = typer.Option(
        "*.jpg",
        "--pattern",
        "-p",
        help="File pattern to match (e.g., *.jpg, *.png)",
    ),
    save_annotated: bool = typer.Option(
        True,
        "--annotated/--no-annotated",
        help="Save annotated images with bounding boxes",
    ),
):
    """
    Process multiple images in batch mode.

    Analyzes all images matching the pattern in the input directory.
    """
    image_files = list(input_dir.glob(pattern))

    if not image_files:
        console.print(
            f"[red]No images found matching pattern '{pattern}' in {input_dir}[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(image_files)} image(s) to process[/cyan]\n")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("saher_batch_output")
    output_dir.mkdir(exist_ok=True, parents=True)

    all_results = []

    for idx, image_path in enumerate(image_files, 1):
        console.print(
            f"[cyan]Processing {idx}/{len(image_files)}:[/cyan] {image_path.name}"
        )

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            console.print(f"[red]  ✗ Failed to load image[/red]")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run pipeline
        violation_results, plate_images, ocr_results = run_pipeline([image])

        if not violation_results or not plate_images:
            console.print(f"[yellow]  No violations detected[/yellow]")
            continue

        console.print(f"[green]  ✓ Found {len(violation_results)} violation(s)[/green]")

        # Process each violation
        for i, (result, plate_img, ocr_text) in enumerate(
            zip(violation_results, plate_images, ocr_results)
        ):
            if not result.boxes:
                continue

            violations = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = VIOLATION_CLASS_NAMES[class_id]
                if class_name in ["person-noseatbelt", "mobile"]:
                    violations.append(class_name)

            if not violations:
                continue

            # Save annotated image
            if save_annotated:
                annotated_image = result.plot()
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                filename = f"{image_path.stem}_violation_{i+1}.png"
                annotated_path = output_dir / filename
                Image.fromarray(annotated_image).save(annotated_path)

            # Save plate
            plate_filename = f"{image_path.stem}_plate_{i+1}.png"
            plate_path = output_dir / plate_filename
            Image.fromarray(plate_img).save(plate_path)

            all_results.append(
                {
                    "source_image": image_path.name,
                    "license_plate": ocr_text,
                    "violations": violations,
                }
            )

    # Save summary JSON
    summary_path = output_dir / "batch_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]✓ Processed {len(image_files)} images[/green]")
    console.print(f"[green]✓ Total violations found: {len(all_results)}[/green]")
    console.print(f"[green]✓ Results saved to:[/green] {output_dir}")


@app.command()
def web():
    """
    Launch the Gradio web interface.
    """
    console.print("[cyan]Starting Gradio web interface...[/cyan]")
    from saher.web import main as web_main

    web_main()


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
