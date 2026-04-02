from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import tempfile

def generate_pdf_report(result, input_image, detection_image, gradcam_image):
    styles = getSampleStyleSheet()

    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf = SimpleDocTemplate(temp_file.name, pagesize=A4)

    elements = []

    # Title
    elements.append(Paragraph("CliniScan AI Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Prediction
    elements.append(Paragraph(f"<b>Prediction:</b> {result['classification']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Confidence:</b> {result['confidence']:.2f}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Note
    elements.append(Paragraph(f"<b>Interpretation:</b> {result['note']}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Save images temporarily
    input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    detection_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    gradcam_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    input_image.save(input_path)
    detection_image.save(detection_path)
    gradcam_image.save(gradcam_path)

    # Add images
    elements.append(Paragraph("Input Image", styles['Heading3']))
    elements.append(RLImage(input_path, width=300, height=300))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Detection (YOLO)", styles['Heading3']))
    elements.append(RLImage(detection_path, width=300, height=300))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Grad-CAM", styles['Heading3']))
    elements.append(RLImage(gradcam_path, width=300, height=300))

    # Build PDF
    pdf.build(elements)

    return temp_file.name