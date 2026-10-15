import json
import streamlit as st
from ccus.predict import Predictor

st.set_page_config(page_title="Cybercrime Complaint Understanding",page_icon="🔎")
st.title("Cybercrime Complaint Understanding")
st.caption("Research prototype · English / Hinglish complaint classification")
path=st.sidebar.text_input("Trained model directory","artifacts/hingroberta")
threshold=st.sidebar.slider("Human review threshold",0.0,1.0,0.5)
st.sidebar.caption("Scores are model probabilities, not calibrated certainty.")
text=st.text_area("Describe the incident",height=200,placeholder="Mujhe Telegram par part-time job offer hua...")
@st.cache_resource
def load(path): return Predictor(path)
if st.button("Analyze complaint",type="primary"):
    if not text.strip(): st.warning("Enter a complaint first.")
    else:
        try:
            result=load(path).predict(text,threshold)
            st.subheader(result["category"])
            st.write(f"Model score: {result['score']:.1%}")
            if result["needs_review"]: st.warning("Low model score. Review the predicted category.")
            if result["truncated"]: st.warning("The complaint exceeded the model's token limit and was truncated.")
            st.dataframe(result["ranking"],hide_index=True)
            with st.expander("Preprocessed input"): st.write(result["processed_text"])
            st.download_button("Download classification report",json.dumps(result,indent=2,ensure_ascii=False),
                file_name="classification.json",mime="application/json")
        except FileNotFoundError:
            st.error("No trained model found. Train a baseline or transformer using the README commands.")
        except Exception as e:
            st.error(f"Could not analyze complaint: {type(e).__name__}: {e}")
st.caption("Predictions need human review. This prototype does not file complaints or determine legal liability.")
