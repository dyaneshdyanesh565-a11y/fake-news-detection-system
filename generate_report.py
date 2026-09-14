from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

import json
from datetime import datetime


# --------------------------------------------------
# LOAD EVALUATION RESULTS
# --------------------------------------------------

with open("evaluation_results.json", "r") as file:
    evaluation = json.load(file)


# --------------------------------------------------
# REPORT FILE
# --------------------------------------------------

output_file = "Fake_News_Detection_Model_Report.pdf"

document = SimpleDocTemplate(
    output_file,
    pagesize=A4,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)


# --------------------------------------------------
# STYLES
# --------------------------------------------------

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleStyle",
    parent=styles["Title"],
    fontSize=24,
    leading=28,
    alignment=TA_CENTER,
    spaceAfter=20
)

heading_style = ParagraphStyle(
    "HeadingStyle",
    parent=styles["Heading2"],
    fontSize=16,
    leading=20,
    spaceBefore=15,
    spaceAfter=10
)

normal_style = ParagraphStyle(
    "NormalStyle",
    parent=styles["BodyText"],
    fontSize=10,
    leading=15,
    spaceAfter=8
)

center_style = ParagraphStyle(
    "CenterStyle",
    parent=styles["BodyText"],
    fontSize=10,
    leading=14,
    alignment=TA_CENTER
)


# --------------------------------------------------
# CONTENT
# --------------------------------------------------

story = []


# --------------------------------------------------
# TITLE
# --------------------------------------------------

story.append(
    Paragraph(
        "FAKE NEWS DETECTION",
        title_style
    )
)

story.append(
    Paragraph(
        "Machine Learning Model Evaluation Report",
        center_style
    )
)

story.append(Spacer(1, 20))


story.append(
    Paragraph(
        f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        center_style
    )
)

story.append(Spacer(1, 25))


# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

story.append(
    Paragraph(
        "1. Project Overview",
        heading_style
    )
)

story.append(
    Paragraph(
        "The Fake News Detection System is a machine learning "
        "application designed to classify news articles as Fake News "
        "or Real News using Natural Language Processing (NLP). "
        "The system converts article text into numerical TF-IDF "
        "features and applies a machine learning classifier to "
        "identify learned linguistic patterns.",
        normal_style
    )
)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

story.append(
    Paragraph(
        "2. Dataset Information",
        heading_style
    )
)

dataset_data = [
    ["Property", "Value"],
    ["Total Articles", str(evaluation["total_articles"])],
    ["Training Articles", str(evaluation["training_articles"])],
    ["Testing Articles", str(evaluation["testing_articles"])],
    ["Dataset Type", "Fake and Real News Articles"],
    ["Text Representation", "TF-IDF"],
]


dataset_table = Table(
    dataset_data,
    colWidths=[2.3 * inch, 3.5 * inch]
)

dataset_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ])
)

story.append(dataset_table)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

story.append(
    Paragraph(
        "3. Selected Machine Learning Model",
        heading_style
    )
)

story.append(
    Paragraph(
        f"The best-performing model selected during model comparison "
        f"was <b>{evaluation['model']}</b>. The model was selected "
        f"based on its F1 Score on the test dataset.",
        normal_style
    )
)


# --------------------------------------------------
# PERFORMANCE
# --------------------------------------------------

story.append(
    Paragraph(
        "4. Model Performance",
        heading_style
    )
)

performance_data = [
    ["Metric", "Score"],
    [
        "Accuracy",
        f"{evaluation['accuracy'] * 100:.2f}%"
    ],
    [
        "Precision",
        f"{evaluation['precision'] * 100:.2f}%"
    ],
    [
        "Recall",
        f"{evaluation['recall'] * 100:.2f}%"
    ],
    [
        "F1 Score",
        f"{evaluation['f1_score'] * 100:.2f}%"
    ]
]


performance_table = Table(
    performance_data,
    colWidths=[2.3 * inch, 3.5 * inch]
)

performance_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ])
)

story.append(performance_table)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

story.append(
    Paragraph(
        "5. Confusion Matrix",
        heading_style
    )
)

story.append(
    Paragraph(
        "The confusion matrix shows how correctly and incorrectly "
        "the model classified Fake News and Real News articles.",
        normal_style
    )
)

image_path = "static/confusion_matrix.png"

try:
    confusion_image = Image(
        image_path,
        width=5.2 * inch,
        height=3.9 * inch
    )

    story.append(confusion_image)

except Exception:
    story.append(
        Paragraph(
            "Confusion matrix image could not be loaded.",
            normal_style
        )
    )


# --------------------------------------------------
# CONFUSION MATRIX VALUES
# --------------------------------------------------

cm = evaluation["confusion_matrix"]

cm_data = [
    ["", "Predicted Fake", "Predicted Real"],
    ["Actual Fake", str(cm[0][0]), str(cm[0][1])],
    ["Actual Real", str(cm[1][0]), str(cm[1][1])]
]

cm_table = Table(
    cm_data,
    colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch]
)

cm_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ])
)

story.append(Spacer(1, 10))
story.append(cm_table)


# --------------------------------------------------
# TECHNOLOGY STACK
# --------------------------------------------------

story.append(
    Paragraph(
        "6. Technology Stack",
        heading_style
    )
)

technologies = [
    ["Technology", "Purpose"],
    ["Python", "Machine Learning and application development"],
    ["Flask", "Web application framework"],
    ["Pandas", "Dataset processing"],
    ["Scikit-learn", "Machine learning and NLP"],
    ["TF-IDF", "Text feature extraction"],
    ["ReportLab", "PDF report generation"],
    ["HTML / CSS", "User interface"],
]


technology_table = Table(
    technologies,
    colWidths=[2.3 * inch, 3.5 * inch]
)

technology_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
    ])
)

story.append(technology_table)


# --------------------------------------------------
# RESPONSIBLE USE
# --------------------------------------------------

story.append(
    Paragraph(
        "7. Responsible Use Notice",
        heading_style
    )
)

story.append(
    Paragraph(
        "<b>Important:</b> This system predicts whether submitted "
        "text resembles patterns learned from the training dataset. "
        "It does not independently verify facts, sources, people, "
        "events, or claims. A prediction should therefore not be "
        "treated as definitive proof that an article is true or false. "
        "For real-world verification, users should consult reliable "
        "sources and professional fact-checking organizations.",
        normal_style
    )
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

story.append(Spacer(1, 25))

story.append(
    Paragraph(
        "Fake News Detection & News Credibility Analysis System",
        center_style
    )
)

story.append(
    Paragraph(
        "Machine Learning • Natural Language Processing • Flask",
        center_style
    )
)


# --------------------------------------------------
# BUILD PDF
# --------------------------------------------------

document.build(story)

print("=" * 60)
print("PDF REPORT GENERATED SUCCESSFULLY")
print("=" * 60)
print()
print("Created:")
print(output_file)
print()
print("Open the PDF from your project folder.")
print("=" * 60)