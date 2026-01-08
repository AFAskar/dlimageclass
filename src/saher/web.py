import gradio as gr
import numpy as np
import cv2
from saher.pipeline import run_pipeline, get_violation_type
from PIL import Image


def process_image(image):
    """
    Process an image through the SAHER pipeline.

    Args:
        image: Input image (numpy array from Gradio)

    Returns:
        tuple: (violation_image, plate_image, results_text)
    """
    if image is None:
        return None, None, "No image provided"

    try:

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(f"Debug: Converted to BGR, shape: {image_bgr.shape}")

        violation_results, plate_results, ocr_results = run_pipeline([image_bgr])

        if not violation_results:
            return None, None, "No violations detected"

        violation_result = violation_results[0]
        if not violation_result.boxes or len(violation_result.boxes) == 0:
            return None, None, "No violations detected (no boxes in result)"

        violation_image = violation_result.plot()
        violation_image = cv2.cvtColor(violation_image, cv2.COLOR_BGR2RGB)

        plate_image = None
        if plate_results and len(plate_results) > 0:
            plate_image = plate_results[0].plot()
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)

        ocr_text = ocr_results[0] if ocr_results else "No text detected"
        violation_types = get_violation_type(violation_result.boxes)

        # Format results text
        results_text = f"Violation Types: {', '.join(violation_types) if violation_types else 'None'}\n"
        results_text += f"OCR Text: {ocr_text}"

        return violation_image, plate_image, results_text

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        return None, None, error_msg


def gradio_interface():
    """
    Create and launch the Gradio interface for SAHER pipeline.
    """
    with gr.Blocks(title="SAHER - Traffic Violation Detection") as demo:
        gr.Markdown("# SAHER Traffic Violation Detection System")
        gr.Markdown(
            "Upload an image to detect traffic violations, license plates, and read plate numbers."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="numpy")
                submit_btn = gr.Button("Process Image", variant="primary")

            with gr.Column():
                violation_output = gr.Image(
                    label="Violation Detection (with bounding boxes)"
                )
                plate_output = gr.Image(
                    label="License Plate Detection (with bounding boxes)"
                )
                text_output = gr.Textbox(label="Detection Results", lines=3)

        submit_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[violation_output, plate_output, text_output],
        )

        gr.Examples(
            examples=[], inputs=[image_input]  # Add example image paths here if needed
        )

    return demo


def launch():
    """
    Launch the Gradio interface.
    """
    demo = gradio_interface()
    demo.launch()
