1. **Terminalinizi Açın:**
   `cog.yaml` dosyasının bulunduğu ana dizine gidin.

2. **Birleşik Komutu Çalıştırın:**
   Aşağıdaki komut hem model sunucusunu (port 5000) hem de Streamlit arayüzünü (port 8501) aynı konteyner içinde başlatacaktır:

   ```bash
   cog run -p 8501 -p 5000 bash -c "python -m cog.server.http & sleep 10 && python -m streamlit run app.py --server.address 0.0.0.0"
