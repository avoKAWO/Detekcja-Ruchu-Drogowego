# 🛣️ Detekcja Ruchu Drogowego

Ten projekt dotyczy badania i implementacji systemu detekcji ruchu drogowego bazując na analizie zdjęć. Wykorzystuje sieci neuronowe YOLO w celu identyfikacji bezpieczeństwa na drodze.

---

## 📌 Spis treści

- [Opis projektu](#-opis-projektu)  
- [Cele](#-cele)  
- [Technologie](#-technologie)  
- [Uruchomienie](#-uruchomienie)  
- [Sprawozdanie](#-szczegółowe-sprawozdanie-z-projektu)  
- [Autorzy](#-autorzy)

---

## 📝 Opis projektu

Projekt oparty jest o analizę klatek z nagrań wideo z kamer samochodowych. Implementuje mechanizmy rozpoznawania i oznaczania elementów ruchudrogowego.

---

## 🎯 Cele

- Wykrywanie pojazdów, znaków, przechodniów itp. na podstawie klatek z nagrań wideo (zdjęcia)

---

## 💻 Technologie

- YOLOv12 (detekcja obiektów)
-  Python
-  Jupiter Notebooks
-  streamlit
-  supervision
-  ...

---


## 🚀 Uruchomienie

### 🧩 Instalacja zależności

```bash
git clone https://github.com/avoKAWO/Detekcja-Ruchu-Drogowego.git
cd Detekcja-Ruchu-Drogowego
pip install git+https://github.com/sunsmarterjie/yolov12.git
pip install supervision streamlit  
pip install opencv-python-headless 
pip install huggingface_hub 
```

### ▶️ Uruchamianie

```bash
streamlit run app.py
```

---

## 📊 Szczegółowe sprawozdanie z projektu

- [`raport.md`](/Documentation/raport.md)

---

## 👥 Autorzy

Projekt wykonany w ramach kursu akademickiego Podstawy Sztucznej Inteligencji.  
Autorzy:  
- *Karol Woda*  
- *Wojciech Zacharski*  
