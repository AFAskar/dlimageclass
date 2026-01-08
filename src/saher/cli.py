from pathlib import Path
import typer
from saher.pipeline import run_pipeline, get_violation_type
from PIL import Image

app = typer.Typer()


@app.command()
def main(
    images_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Path to image or directory of images",
    ),
    save_plates: bool = typer.Option(
        False, "-sp", "--save-plates", help="Whether to save license plate images"
    ),
    save_violations: bool = typer.Option(
        False,
        "-sv",
        "--save-violations",
        help="Whether to save violation images with bounding boxes",
    ),
    save_all: bool = typer.Option(
        False, "-s", "--save", help="Whether to save all output images"
    ),
):
    from saher.pipeline import run_pipeline

    violations, plates, ocr_texts = run_pipeline([images_path])
    if save_all:
        save_plates = True
        save_violations = True

    for i, (violation, plate, ocr_text) in enumerate(
        zip(violations, plates, ocr_texts)
    ):
        typer.echo(f"Violation {i+1}:")
        violation_types = get_violation_type(violation.boxes)
        typer.echo(f" - Violation Types: {violation_types}")
        typer.echo(f" - OCR Text: {ocr_text}")
        typer.echo("")
        if save_plates:
            plate_path = images_path.parent / f"plate_{i+1}.png"
            Image.fromarray(plate.plot()).save(plate_path)
            typer.echo(f" - Saved Plate Image to: {plate_path}")
        if save_violations:
            violation_path = images_path.parent / f"violation_{i+1}.png"
            # save with bounding boxes
            Image.fromarray(violation.plot()).save(violation_path)
            typer.echo(f" - Saved Violation Image to: {violation_path}")
