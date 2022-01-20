
PIP := pip3

install:
	$(PIP) install -r requirements.txt

run:
	streamlit run app.py
