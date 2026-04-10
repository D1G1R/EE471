import streamlit as st
import requests
import base64
import mimetypes

st.set_page_config(page_title="Cat Classifier", page_icon="🐱")
st.title("Cat Breed Classifier 🐱")
st.write("Upload a cat photo and let the model find out what breed it is!")

uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded")
    
    if st.button("Classify!"):
        with st.spinner("Model analyzing..."):
            # Görseli base64 formatına çevirip Cog API'sine gönderiyoruz
            base64_img = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)
            if mime_type is None:
                mime_type = "image/jpeg"  # fallback
            payload = {"input": {"image": f"data:{mime_type};base64,{base64_img}"}}
            
            try:
                # Arka planda çalışan Cog sunucumuza istek atıyoruz
                response = requests.post("http://127.0.0.1:5000/predictions", json=payload)
                
                if response.status_code == 200:
                    result = response.json().get("output")
                    if isinstance(result, dict):
                        predicted_class = max(result, key=result.get)
                    else:
                        predicted_class = result
                    st.success(f"This cat breed: **{predicted_class.title()}**")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error("Cannot reach Cog server. Make sure Cog is running in the first terminal.")