from flask import Flask, render_template, request, send_file
import pickle
import re
import os
import json

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


app = Flask(__name__)


# ==================================================
# LOAD MODEL
# ==================================================

with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==================================================
# LOAD MODEL RESULTS
# ==================================================

if os.path.exists("model_results.json"):
    with open("model_results.json", "r") as file:
        model_results = json.load(file)
else:
    model_results = {}


# ==================================================
# TEXT STATISTICS
# ==================================================

def get_text_statistics(text):

    words = text.split()

    sentences = re.split(r"[.!?]+", text)
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    uppercase_words = [
        word
        for word in words
        if len(word) > 2 and word.isupper()
    ]

    exclamation_count = text.count("!")
    question_count = text.count("?")

    word_count = len(words)
    character_count = len(text)
    sentence_count = len(sentences)

    if sentence_count > 0:
        average_sentence_length = round(
            word_count / sentence_count,
            2
        )
    else:
        average_sentence_length = 0

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "uppercase_words": len(uppercase_words),
        "exclamation_marks": exclamation_count,
        "question_marks": question_count
    }


# ==================================================
# CREDIBILITY INDICATORS
# ==================================================

def analyze_credibility(text):

    indicators = []

    text_lower = text.lower()

    clickbait_phrases = [
        "you won't believe",
        "shocking",
        "breaking",
        "what happens next",
        "must see",
        "this will shock you",
        "secret",
        "viral",
        "unbelievable",
        "exposed"
    ]

    found_clickbait = []

    for phrase in clickbait_phrases:

        if phrase in text_lower:
            found_clickbait.append(phrase)

    if found_clickbait:

        indicators.append({
            "type": "warning",
            "title": "Clickbait-style wording",
            "description":
                "The article contains attention-grabbing phrases "
                "that may indicate sensational writing."
        })

    exclamation_count = text.count("!")

    if exclamation_count >= 3:

        indicators.append({
            "type": "warning",
            "title": "Excessive punctuation",
            "description":
                f"The article contains {exclamation_count} "
                "exclamation marks."
        })

    words = text.split()

    uppercase_count = sum(
        1
        for word in words
        if len(word) > 2 and word.isupper()
    )

    if uppercase_count >= 3:

        indicators.append({
            "type": "warning",
            "title": "Excessive uppercase wording",
            "description":
                "Several words are written in uppercase, "
                "which can be associated with emphasis or "
                "sensational style."
        })

    if len(words) < 30:

        indicators.append({
            "type": "info",
            "title": "Short content",
            "description":
                "The submitted text is very short, so the "
                "analysis may be less reliable."
        })

    if len(words) > 2000:

        indicators.append({
            "type": "info",
            "title": "Long content",
            "description":
                "The article is unusually long. The model "
                "analyzes the complete submitted text."
        })

    if not indicators:

        indicators.append({
            "type": "positive",
            "title": "No major style indicators detected",
            "description":
                "The text did not trigger the basic linguistic "
                "credibility indicators used by this system."
        })

    return indicators


# ==================================================
# INFLUENTIAL WORDS
# ==================================================

def get_influential_words(text):

    try:

        tfidf_values = vectorizer.transform([text])

        feature_names = (
            vectorizer.get_feature_names_out()
        )

        if not hasattr(model, "coef_"):
            return []

        coefficients = model.coef_[0]

        row = tfidf_values.toarray()[0]

        contributions = row * coefficients

        feature_data = []

        for index, value in enumerate(contributions):

            if value != 0:

                feature_data.append(
                    (
                        feature_names[index],
                        value
                    )
                )

        feature_data.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = feature_data[:10]

        results = []

        for word, score in top_features:

            if score > 0:
                direction = "Real News"
            else:
                direction = "Fake News"

            results.append({
                "word": word,
                "direction": direction,
                "score": round(
                    abs(float(score)),
                    4
                )
            })

        return results

    except Exception:

        return []


# ==================================================
# PDF REPORT GENERATION
# ==================================================

def create_pdf_report(
    article_text,
    result_data,
    statistics,
    indicators,
    influential_words
):

    output_file = "News_Analysis_Report.pdf"

    document = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

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

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "NEWS ANALYSIS REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Fake News Detection & Credibility Analysis",
            center_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated on: "
            + __import__("datetime")
            .datetime.now()
            .strftime("%d %B %Y, %I:%M %p"),
            center_style
        )
    )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "1. Prediction Result",
            heading_style
        )
    )

    prediction_data = [
        ["Property", "Result"],
        [
            "Classification",
            result_data["prediction"]
        ],
        [
            "Confidence Level",
            result_data["confidence_level"]
        ],
        [
            "Decision Score",
            str(result_data["decision_score"])
        ]
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[
            2.3 * inch,
            3.5 * inch
        ]
    )

    prediction_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.black
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                10
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(prediction_table)

    # --------------------------------------------------
    # ARTICLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "2. Submitted News Content",
            heading_style
        )
    )

    safe_text = (
        article_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            safe_text,
            normal_style
        )
    )

    # --------------------------------------------------
    # TEXT STATISTICS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "3. Text Statistics",
            heading_style
        )
    )

    statistics_data = [
        ["Statistic", "Value"],
        [
            "Word Count",
            str(statistics["word_count"])
        ],
        [
            "Character Count",
            str(statistics["character_count"])
        ],
        [
            "Sentence Count",
            str(statistics["sentence_count"])
        ],
        [
            "Average Sentence Length",
            str(statistics["average_sentence_length"])
        ],
        [
            "Uppercase Words",
            str(statistics["uppercase_words"])
        ],
        [
            "Exclamation Marks",
            str(statistics["exclamation_marks"])
        ],
        [
            "Question Marks",
            str(statistics["question_marks"])
        ]
    ]

    statistics_table = Table(
        statistics_data,
        colWidths=[
            2.8 * inch,
            3 * inch
        ]
    )

    statistics_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.black
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(statistics_table)

    # --------------------------------------------------
    # CREDIBILITY INDICATORS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "4. Credibility Indicators",
            heading_style
        )
    )

    for indicator in indicators:

        story.append(
            Paragraph(
                f"<b>{indicator['title']}</b>",
                normal_style
            )
        )

        story.append(
            Paragraph(
                indicator["description"],
                normal_style
            )
        )

    # --------------------------------------------------
    # INFLUENTIAL WORDS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "5. Influential Words & Phrases",
            heading_style
        )

    )

    if influential_words:

        words_data = [
            ["Word / Phrase", "Direction", "Score"]
        ]

        for item in influential_words:

            words_data.append([
                item["word"],
                item["direction"],
                str(item["score"])
            ])

        words_table = Table(
            words_data,
            colWidths=[
                2.4 * inch,
                1.8 * inch,
                1.2 * inch
            ]
        )

        words_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.black
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ])
        )

        story.append(words_table)

    else:

        story.append(
            Paragraph(
                "Influential feature information is not available "
                "for the selected model.",
                normal_style
            )
        )

    # --------------------------------------------------
    # RESPONSIBLE USE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "6. Responsible Use",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "<b>Important:</b> This system identifies patterns "
            "learned from its training dataset. It does not "
            "independently verify facts, sources, people, events, "
            "or claims. The prediction should not be treated as "
            "definitive proof that a news article is true or false.",
            normal_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Fake News Detection & News Credibility Analysis System",
            center_style
        )
    )

    document.build(story)

    return output_file


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    article_text = ""
    statistics = None
    indicators = []
    influential_words = []

    if request.method == "POST":

        article_text = request.form.get(
            "article_text",
            ""
        ).strip()

        if article_text:

            text_vector = vectorizer.transform(
                [article_text]
            )

            prediction = model.predict(
                text_vector
            )[0]

            decision_score = model.decision_function(
                text_vector
            )[0]

            if prediction == 1:
                result = "Real News"
            else:
                result = "Fake News"

            score_strength = abs(
                float(decision_score)
            )

            if score_strength >= 2:
                confidence_level = "High"

            elif score_strength >= 1:
                confidence_level = "Moderate"

            else:
                confidence_level = "Low"

            statistics = get_text_statistics(
                article_text
            )

            indicators = analyze_credibility(
                article_text
            )

            influential_words = get_influential_words(
                article_text
            )

            result = {
                "prediction": result,
                "decision_score": round(
                    float(decision_score),
                    4
                ),
                "confidence_level":
                    confidence_level
            }

        else:

            result = {
                "prediction": "No Text",
                "decision_score": 0,
                "confidence_level": "N/A"
            }

    return render_template(
        "index.html",
        result=result,
        article_text=article_text,
        statistics=statistics,
        indicators=indicators,
        influential_words=influential_words,
        model_results=model_results
    )


# ==================================================
# DOWNLOAD PDF
# ==================================================

@app.route("/download-report", methods=["POST"])
def download_report():

    article_text = request.form.get(
        "article_text",
        ""
    ).strip()

    if not article_text:
        return "No article text provided.", 400

    text_vector = vectorizer.transform(
        [article_text]
    )

    prediction = model.predict(
        text_vector
    )[0]

    decision_score = model.decision_function(
        text_vector
    )[0]

    if prediction == 1:
        prediction_label = "Real News"
    else:
        prediction_label = "Fake News"

    score_strength = abs(
        float(decision_score)
    )

    if score_strength >= 2:
        confidence_level = "High"
    elif score_strength >= 1:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    result_data = {
        "prediction": prediction_label,
        "decision_score": round(
            float(decision_score),
            4
        ),
        "confidence_level":
            confidence_level
    }

    statistics = get_text_statistics(
        article_text
    )

    indicators = analyze_credibility(
        article_text
    )

    influential_words = get_influential_words(
        article_text
    )

    pdf_file = create_pdf_report(
        article_text,
        result_data,
        statistics,
        indicators,
        influential_words
    )

    return send_file(
        pdf_file,
        as_attachment=True,
        download_name="News_Analysis_Report.pdf"
    )


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)

    print("\nModel:", type(model).__name__)

    if model_results:

        print(
            "Test Accuracy:",
            f"{model_results['results']['Linear SVM']['accuracy'] * 100:.2f}%"
        )

    print("\nStarting Flask server...")
    print("Open: http://127.0.0.1:5000")

    app.run(
        debug=True
    )