import streamlit as st
import requests
import base64
import mimetypes

st.set_page_config(page_title="Garment Classifier Pro", page_icon="👕", layout="wide")
st.title("Garment Classifier: Baseline vs Optimized 👕")
st.write("Upload a clothing photo and see how optimization improves the AI's prediction!")

uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi ortala
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(uploaded_file, caption="Uploaded Garment", use_container_width=True)
    
    st.divider() # Araya şık bir çizgi çekelim
    
    if st.button("Run Comparison!", use_container_width=True):
        with st.spinner("Both models are analyzing..."):
            base64_img = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)
            if mime_type is None: mime_type = "image/jpeg"
            payload = {"input": {"image": f"data:{mime_type};base64,{base64_img}"}}
            
            try:
                response = requests.post("http://127.0.0.1:5000/predictions", json=payload)
                
                if response.status_code == 200:
                    result = response.json().get("output")
                    
                    # Ekranı ikiye böl (Yan yana gösterim)
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("🤖 Baseline Model")
                        st.info(f"Prediction: **{result['baseline_prediction'].title()}**")
                        
                    with res_col2:
                        st.subheader("🚀 Optimized Model")
                        st.success(f"Prediction: **{result['optimized_prediction'].title()}**")
                        
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error("Cannot reach Cog server. Make sure Cog is running.")