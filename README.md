# Rozpoznávanie gest a držaných objektov v reálnom čase

Školský projekt v rámci 2. ročníka strednej školy (odbor Umelá inteligencia)[cite: 10].

## 📌 O projekte

V tomto projekte som vytvoril systém na rozpoznávanie gest a držaných objektov v reálnom čase pomocou webkamery[cite: 1, 4, 10]. Pomocou knižnice **MediaPipe Hands** extrahujem 21 kľúčových bodov ruky (42 súradníc `x, y`) a následne klasifikátor **K-Nearest Neighbors (KNN)** natrénovaný na vlastnom datasete určuje, či v ruke držím **pero**, **mobil** alebo **nič**[cite: 1, 2, 3, 4].

## 🛠️ Použité technológie

* Python 3.11[cite: 1, 10]
* OpenCV (`cv2`)[cite: 1]
* MediaPipe Hands (`0.10.14`)[cite: 1]
* scikit-learn (`KNeighborsClassifier`)[cite: 3]
* pandas & joblib[cite: 3]

## 📁 Štruktúra repozitára

* `objekt_zber.py` – skript na zber dát z webkamery a ukladanie súradníc ruky do CSV[cite: 1, 10]
* `objekt_trening.py` – skript na trénovanie a vyhodnotenie KNN klasifikátora[cite: 3, 10]
* `objekt_test.py` – skript na testovanie a rozpoznávanie gest v reálnom čase[cite: 4, 10]
* `dataset_hold_2.csv` – vlastný dataset súradníc kľúčových bodov ruky
* `hold_classifier_2.pkl` – uložený natrénovaný model klasifikátora[cite: 4]
* `requirements.txt` – zoznam potrebných Python knižníc[cite: 7]

## 🚀 Spustenie

1. Nainštaluj potrebné knižnice:
   ```bash
   pip install -r requirements.txt
