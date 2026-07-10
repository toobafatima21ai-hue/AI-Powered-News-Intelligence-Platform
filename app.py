import streamlit as st
import pandas as pd
import plotly.express as px
import yake

from transformers import pipeline

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI-Powered News Intelligence Platform",
    page_icon="📰",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="bert_news_classifier",
        tokenizer="bert_news_classifier"
    )

classifier = load_model()

# =====================================================
# LABELS
# =====================================================

labels = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech"
}

icons = {
    "World": "🌍",
    "Sports": "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬"
}

related_keywords = {
    "World": ["government", "country", "war"],
    "Sports": ["match", "player", "team"],
    "Business": ["market", "stock", "company"],
    "Sci/Tech": ["AI", "technology", "software"]
}

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Choose Page",
    [
        "Classifier",
        "Analytics"
    ]
)

theme = st.sidebar.radio(
    "Theme",
    [
        "Light",
        "Dark"
    ]
)

st.sidebar.success(f"Current Theme: {theme}")

# =====================================================
# CLASSIFIER PAGE
# =====================================================

if page == "Classifier":

    st.title("📰 AI-Powered News Intelligence Platform")

    st.write(
        """
        Predict news categories using a fine-tuned BERT model.
        Includes confidence visualization, explainable AI,
        prediction history, and batch classification.
        """
    )

    # ==============================================
    # SINGLE PREDICTION
    # ==============================================

    st.subheader("Single Headline Prediction")

    headline = st.text_input(
        "Enter News Headline"
    )

    if st.button("Predict"):

        if headline.strip() != "":

            results = classifier(
                headline,
                return_all_scores=True
            )

            scores = results[0]

            best_prediction = max(
                scores,
                key=lambda x: x["score"]
            )

            label = labels[
                best_prediction["label"]
            ]

            confidence = best_prediction["score"]

            # ======================================
            # RESULT
            # ======================================

            st.success(
                f"{icons[label]} Category: {label}"
            )

            st.info(
                f"Confidence: {confidence:.2%}"
            )

            if confidence < 0.70:

                st.warning(
                    "⚠ Low Confidence Prediction"
                )

            # ======================================
            # CONFIDENCE CHART
            # ======================================

            chart_data = []

            for item in scores:

                chart_data.append(
                    {
                        "Category": labels[item["label"]],
                        "Score": item["score"]
                    }
                )

            df_chart = pd.DataFrame(chart_data)

            st.subheader("Confidence Scores")

            st.bar_chart(
                df_chart.set_index("Category")
            )

            # ======================================
            # RELATED KEYWORDS
            # ======================================

            st.subheader("Related Keywords")

            st.info(
                ", ".join(
                    related_keywords[label]
                )
            )

            # ======================================
            # EXPLAINABLE AI
            # ======================================

            st.subheader("Important Keywords")

            kw_extractor = yake.KeywordExtractor()

            keywords_found = kw_extractor.extract_keywords(
                headline
            )

            for keyword, score in keywords_found[:5]:

                st.write(
                    f"• {keyword}"
                )

            # ======================================
            # SAVE HISTORY
            # ======================================

            st.session_state.history.append(
                {
                    "Headline": headline,
                    "Category": label,
                    "Confidence": round(
                        confidence * 100,
                        2
                    )
                }
            )

        else:

            st.warning(
                "Please enter a headline."
            )

    # ==============================================
    # HISTORY
    # ==============================================

    st.subheader("Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    if not history_df.empty:

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.subheader(
            "Prediction Distribution"
        )

        st.bar_chart(
            history_df["Category"].value_counts()
        )

    # ==============================================
    # BATCH PREDICTION
    # ==============================================

    st.subheader(
        "Batch Headline Classification"
    )

    batch_text = st.text_area(
        "Paste multiple headlines (one per line)"
    )

    if st.button("Batch Predict"):

        rows = []

        headlines = batch_text.split("\n")

        for item in headlines:

            if item.strip() != "":

                pred = classifier(item)

                category = labels[
                    pred[0]["label"]
                ]

                rows.append(
                    [
                        item,
                        category,
                        round(
                            pred[0]["score"] * 100,
                            2
                        )
                    ]
                )

        if len(rows) > 0:

            batch_df = pd.DataFrame(
                rows,
                columns=[
                    "Headline",
                    "Prediction",
                    "Confidence %"
                ]
            )

            st.dataframe(
                batch_df,
                use_container_width=True
            )

# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    st.title("📊 Dataset Analytics Dashboard")

    try:

        data = pd.read_csv(
            "train.csv",
            header=None
        )

        data = data.iloc[1:]

        counts = data[0].value_counts()

        mapping = {
            "1": "World",
            "2": "Sports",
            "3": "Business",
            "4": "Sci/Tech",
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech"
        }

        labels_list = []

        for cls in counts.index:

            labels_list.append(
                mapping.get(
                    cls,
                    str(cls)
                )
            )

        st.subheader(
            "Category Distribution"
        )

        fig = px.pie(
            values=counts.values,
            names=labels_list,
            title="News Categories"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Samples Per Category"
        )

        bar_df = pd.DataFrame(
            {
                "Category": labels_list,
                "Count": counts.values
            }
        )

        st.bar_chart(
            bar_df.set_index(
                "Category"
            )
        )

        st.metric(
            "Total Training Samples",
            len(data)
        )

    except Exception as e:

        st.error(
            f"Could not load train.csv\n\n{e}"
        )