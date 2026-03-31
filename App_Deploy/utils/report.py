from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(result, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("CliniScan Report", styles["Title"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Prediction: {result['label']}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {result['confidence']:.2f}", styles["Normal"]))

    doc.build(content)
    return filename