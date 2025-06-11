# Detekcja obiektów ruchu drogowego
🤖 Podsumowanie projektu realizowanego w ramach zajęć z **Podstaw Sztucznej Inteligencji**

---

#### ✍️ Autorzy:
- Karol Woda
- Wojciech Zacharski

🎓 Wydział: Inżynierii Materiałowej i Informatyki Przemysłowej\
📘 Kierunek: Informatyka Techniczna\
⏳ Rok: 3\
🔖 Semestr: 6

#### 📆 Data oddania: 12.06.2025

---

## 🎯 Określenie tematu i celu projektu, analiza wymagań

W ramach naszego projektu zdecydowaliśmy się zrealizować problem detekcji obiektów ruchu drogowego. Naszym celem było wytrenowanie a następnie wdrożenie modelu w funkcjonalnej formie.

#### Klasy wykrywanych obiektów:

1. 🚗 samochód (car)
2. 🚛 Ciężarówka (Truck)
3. 🏍️ Motocykl (Motorcycle)
4. ⚠️ Znak ostrzegawczy (Warning Sign)
5. ⛔ Znak zakazu (Prohibition Sign)
6. ⏱️ Znak ograniczenia prędkości (Speed Limit Sign)
7. 🪧 Inny znak (Different Traffic Sign)
8. 🟢 Zielone światło drogowe (Green Traffic Light)
9. 🔴 Czerwone światło drogowe (Red Traffic Light)
10. 🚶‍♂️ Pieszy (Pedestrian)
11. 🚸 Przejście dla pieszych (Pedestrian Crossing)

#### Wymagania:
- ⚡ **Wydajność:** Model nie musi działać w czasie rzeczywistym, ale powinien w sensownym czasie dawać wyniki.
- 🏗️ **Skalowalność:** Model powinien działać lokalnie na podstawie dostarczonych zdjęć.
- 🔒️ **Bezpieczeństwo:** Na zdjęciach nie mogą być widoczne dane wrażliwe (np. numer rejestracyjny, twarz przechodnia itp.).
- 🧮 **Dokładność:** Elementy rozpoznane przez program powinny znajdować się na obrazku.

---

## 🗂️ Zbiór danych i ich przygotowanie

Wybraliśmy zbiór danych dostępny na licencji `Apache 2.0`: [Traffic Road Object Detection Polish 12k](https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k).
Pobraliśmy model, a następnie wykorzystując internetowe narzędzie [Roboflow](https://roboflow.com/brand)
Przygotowaliśmy zbiór danych. Zdjęcia w naszym wybranym datasecie były już pooznaczane od razu w liczbie 4027 zdjęć, jednak by przećwiczyć oznaczanie, postanowiliśmy samodzielnie oznaczyć kilka zdjęć:

![](Documentation_Photos/img.png)

Proces oznaczania:

![](Documentation_Photos/img_1.png)

Dalej stworzyliśmy w [Roboflow](https://roboflow.com/brand) nową wersję naszego datasetu, ustalajć proporcję na:
- 88% set treningowy (train set)
- 8% set walidacyjny (valid set)
- 4% set testowy (test set)

Ustaliliśmy by zdjęcia były przeskalowane do rozmiaru 640x640, nie nakładaliśmy żadnych modyfikacji zdjęć, gdyż zauważyliśmy, że orginalny autor datasetu już to zrobił.

![](Documentation_Photos/img_2.png)

![](Documentation_Photos/img_3.png)

---

## ⚙️ Wybór i implementacja modelu AI

Trenowaliśmy model w dwójnasób:
- wykorzystując interfejs [Roboflow](https://roboflow.com/brand)
- wykorzystując internetowe narzędzie [Google Collab](https://colab.google/)
- próbowaliśmy również uczyć model w środowisku lokalnym wykorzystując venv, jednak w wyniku trudności związanych z odpowiednim przygotowaniem środowiska, porzuciliśmy to podejście

Model uczyliśmy w obydwu przypadkach wykorzystując [YOLOv12](https://github.com/sunsmarterjie/yolov12)

### Trenowanie w Roboflow
W przypadku trrenowania w [Roboflow](https://roboflow.com/brand), proces sprowadzał się do uruchomienia procesu nauki. Uczeni zostało przedwcześnie zatrzymane z powodu problemó z łączem.

![](Documentation_Photos/img_5.png)

### Trenowanie w Collabie

Najpierw w collabie zainstalowaliśmy odpowiednie biblioteki i pakiety

Najważniejsze wykorzystane biblioteki i pakiety:
- ultralytics
- roboflow
- supervision
- flash-attn

Dalej przesłaliśmy dataset poprzez link, wykorzystując jedną z bibliotek. By to było możliwe utworzyliśmy klucz dostępu do datasetu `ROBOFLOW_API_KEY`, który nie podajemy jawnie w celach bezpieczeństwa.

Stosując się do [dokumentacji](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov12-object-detection-model.ipynb#scrollTo=7S97p_O7YPsa\) przeprowadziliśmy trenowanie modelu w 100 epokach:

![](Documentation_Photos/img_4.png)

Najlepszy wynik zapisaliśmy w formacie .pt: [`best.pt`](./../best.pt)

Niżej przedstawiamy kilka wybranych batchy treningowych i validacyjnych z procesu trenowania.

#### Batche treningowe:
- batch 0\
![](Documentation_Photos/train_batch0.jpg)
- batch 1\
![](Documentation_Photos/train_batch1.jpg)
- batch 2\
![](Documentation_Photos/train_batch2.jpg)
- batch 19980\
![](Documentation_Photos/train_batch19980.jpg)
- batch 19981\
![](Documentation_Photos/train_batch19981.jpg)
#### Batche validacyjne:
- batch 0
  - przewidziane adnotacje:\
  ![](Documentation_Photos/val_batch0_pred.jpg)
  - właściwe adnotacje:\
  ![](Documentation_Photos/val_batch0_labels.jpg)
- batch 1
  - przewidziane adnotacje:\
  ![](Documentation_Photos/val_batch1_pred.jpg)
  - właściwe adnotacje:\
  ![](Documentation_Photos/val_batch1_labels.jpg)
---

## 📈 Ocena wyników modelu i optymalizacja
### Ewaluacja modelu trenowanego w Roboflow:

Niestety możelowości ewaluacji modelu w darmowym pakiecie Roboflow są ograniczone, jednk na podstawie dostarczonych danych przez aplikacje jesteśmy w stanie wysunąć pewne wnioski o modelu.

![](Documentation_Photos/img_6.png)

1. mAP@50 (Mean Average Precision at IoU=0.50): 59.0%
- Co to znaczy: Średnia precyzja detekcji przy założeniu, że detekcja jest uznawana za trafną, jeśli współczynnik IoU (Intersection over Union) między wykrytą a rzeczywistą ramką przekracza 0.50.
- Interpretacja: Model osiąga 59% skuteczności w identyfikacji i lokalizacji obiektów przy umiarkowanej dokładności ramki. W kontekście detekcji w ruchu drogowym to przyzwoity wynik, ale jest miejsce na poprawę.

2. Precision (Precyzja): 73.2%
- Co to znaczy: Spośród wszystkich obiektów wykrytych przez model, 73.2% to prawidłowe wykrycia.
- Interpretacja: Model rzadko generuje fałszywe alarmy (fałszywie pozytywne wykrycia). Oznacza to, że jeśli coś zostało wykryte, to najprawdopodobniej rzeczywiście tam było (np. samochód, pieszy).

3. Recall (Czułość): 54.7%
- Co to znaczy: Spośród wszystkich rzeczywistych obiektów (np. pojazdów, pieszych) model wykrył 54.7%.

- Interpretacja: Model pomija prawie połowę rzeczywistych obiektów. To może być problematyczne w systemach bezpieczeństwa, np. w autonomicznych pojazdach, gdzie pominięcie pieszego może mieć poważne konsekwencje.

Ogólna interpretacja:

- Precyzja jest dobra (73.2%) – model nie zgłasza zbyt wielu błędnych wykryć.

- Czułość jest umiarkowana (54.7%) – model nie wykrywa wszystkich ważnych obiektów, co może stanowić zagrożenie w praktyce.

- mAP@50 na poziomie 59.0% sugeruje, że model działa poprawnie, ale nadal wymaga usprawnień

![](Documentation_Photos/img_7.png)

Powyższy wykres przedstawia średnią precyzję dla poszczególnych klas (mAP@50) w zbiorze walidacyjnym modelu detekcji YOLO. Metryka ta pokazuje, jak dobrze model rozpoznaje każdą z klas obiektów.

#### 📊 Podsumowanie wyników wg klas:
| Klasa                      | mAP\@50 | Interpretacja                                                      |
| -------------------------- | ------- | ------------------------------------------------------------------ |
| **car**                    | 88.0%   | Bardzo dobra detekcja pojazdów – model dobrze się uczy tej klasy.  |
| **Red Traffic Light**      | 85.0%   | Bardzo dobra detekcja czerwonych świateł.                          |
| **Different Traffic Sign** | 68.0%   | Dobre rozpoznawanie znaków ogólnych.                               |
| **Green Traffic Light**    | 63.0%   | Poprawne rozpoznawanie, ale słabsze niż czerwone światło.          |
| **Prohibition Sign**       | 64.0%   | Też przyzwoicie, ale może zależeć od jakości danych.               |
| **Speed Limit Sign**       | 26.0%   | **Bardzo niska skuteczność** – model ma problem z tą klasą.       |
| **Pedestrian**             | 38.0%   | Słaba detekcja pieszych – potencjalnie niebezpieczne.            |
| **Pedestrian Crossing**    | 52.0%   | Umiarkowana skuteczność, można poprawić.                           |
| **Truck**                  | 59.0%   | OK, ale niżej niż dla samochodów osobowych.                        |
| **Warning Sign**           | 55.0%   | Średnia jakość – może być mylony z innymi znakami.                 |
| **Motorcycle**             | 51.0%   | Umiarkowana skuteczność, może wynikać z mniejszej liczby przykładów. |


![](Documentation_Photos/img_8.png)

##### 📈 Model Performance – górny wykres

- mAP (ciemnofioletowa linia): średnia precyzja rośnie z epoki na epokę, z ~0.22 do ~0.58, co oznacza, że model się uczy i poprawia swoje wyniki.

- mAP@50:95 (jasnofioletowa linia): to bardziej rygorystyczna miara (średnia dla różnych wartości IoU). Również rośnie, choć wolniej – co sugeruje, że model poprawia precyzję, ale lokalizacja bounding boxów nadal może być niedokładna.

📌 Wniosek: Model konwerguje – warto byłoby trenować dłużej niż 12 epok, bo nie widać jeszcze wypłaszczenia krzywej mAP.

##### 🔍 Lossy (dolne wykresy) – mierzą błędy modelu
- 🟪 Box Loss
  - Mierzy błąd lokalizacji bounding boxów.
  - Widoczny jest spadek z ~1.9 do ~1.35, co oznacza, że model uczy się lepiej lokalizować obiekty.

- 🟪 Class Loss
  - Błąd klasyfikacji obiektów. 
  - Na początku dość wysoki (ponad 4.0), potem gwałtowny spadek do poziomu ~1.0 – bardzo dobry znak, oznacza lepsze rozróżnianie klas.

- 🟪 Object Loss
  - Mierzy błąd związany z pewnością, czy dany obiekt istnieje.
  - Po kilku fluktuacjach również stabilizuje się poniżej 1.1 – to dobry objaw, choć można próbować dalej optymalizować.

##### 🧠 Interpretacja ogólna
- Model efektywnie się uczy – wszystkie trzy metryki loss spadają, a mAP rośnie.
- Krzywe nie są jeszcze w pełni wypłaszczone – sugeruje to, że model nie osiągnął jeszcze maksimum możliwości.
- Brak oznak nadmiernego dopasowania (overfittingu) – gdyby mAP zaczął spadać przy dalszym spadku lossów, byłby to sygnał ostrzegawczy.

🔴 Wniosek: Dane są zgodne z tym co nadmieniłem wcześniej, że model został wstrzymany przedwcześnie z powodu problemów z łączem i przez to jest niedouczony. Nie ponawialiśmy próby uczenia, gdyż skupiliśmy się na dopracowaniu i dobrym wytrenowaniu modelu w Collabie
### Ewaluacja modelu trenowanego w Collabie:

#### Macierz korelacji
Nieznormalizowana:

![](Documentation_Photos/confusion_matrix.png)

Znormalizowana:

![](Documentation_Photos/confusion_matrix_normalized.png)

##### 📊 Co przedstawia ta macierz?
Znormalizowana macierz pomyłek (confusion matrix), to tablica pokazująca, jak często model poprawnie lub błędnie klasyfikował poszczególne obiekty w zestawie testowym.
- Wiersze: przewidywania modelu
- Kolumny: rzeczywiste etykiety

Wartości są znormalizowane (od 0 do 1), więc np. wartość 0.69 dla Different Traffic Sign oznacza, że w 69% przypadków model poprawnie przypisał tę etykietę.

##### 📌 Na co warto zwrócić uwagę?
✅ Mocne strony:
- Red Traffic Light: 0.87 poprawnych detekcji — świetny wynik.
- Car: 0.86 poprawnych detekcji — model bardzo dobrze rozpoznaje samochody.
- Pedestrian Crossing: 0.71 trafień — przyzwoity wynik.

⚠️ Słabe strony i potencjalne problemy:
- Motorcycle: tylko 0.50 poprawnych detekcji, a aż 0.48 przypisanych błędnie do background — model często nie rozpoznaje motocykli.

- Pedestrian: 0.51 trafień, a aż 0.29 przypisanych do background — możliwe problemy z rozpoznawaniem pieszych.

- Truck i Warning Sign: też stosunkowo niskie wartości (~0.59-0.62).

- Different Traffic Sign: chociaż 0.69 poprawnych, to aż 0.22 błędnie przypisanych do background — może oznaczać problemy z detekcją przy nieidealnych warunkach.

📌 Wysoka kolumna background:
Warto zauważyć, że wiele obiektów mylonych jest z tłem (background), co oznacza, że YOLO ma trudność z ich zauważeniem lub jest nadmiernie ostrożny przy predykcji (np. Motorcycle, Pedestrian).

#### Wykres pewności F1
![](Documentation_Photos/F1_curve.png)

##### 📊 Co pokazuje ten wykres?
Dla każdej klasy (i dla wszystkich razem — niebieska, pogrubiona linia) pokazuje zależność między:

- Confidence threshold (wartość pewności predykcji od 0 do 1)

- F1-score (harmoniczna średnia precision i recall)

W skrócie: jaki próg confidence da najlepszy balans między precyzją a czułością.

##### 📌 Najważniejsze obserwacje:
✅ Optymalny próg:
- Dla wszystkich klas razem najlepiej wychodzi F1 = 0.69 przy threshold 0.315
To znaczy, że ustawiając `confidence_threshold = 0.315` w YOLO mamy optymalny kompromis między precision a recall dla całego zbioru.

📈 Które klasy radzą sobie dobrze:
- Red Traffic Light, Car, Motorcycle osiągają wysokie F1-score (>0.8) przy tym lub nieco wyższym thresholdzie.
- Different Traffic Sign i Prohibition Sign też mają solidne krzywe.

⚠️ Problematyczne klasy:
- Pedestrian (czerwona linia) i Speed Limit Sign (szara linia) — niskie F1 na całym zakresie confidence, z maksymalnymi wartościami około 0.55 dla pedestrian i 0.45 dla speed limit.
- Warning Sign też wypada poniżej średniej.

#### Krzywa precyzja-pewność
![](Documentation_Photos/P_curve.png)

##### 📊 Co pokazuje ten wykres?
Dla każdej klasy oraz dla wszystkich razem (gruba niebieska linia):
- oś X — próg pewności (confidence)
- oś Y — precision (precyzja = liczba prawidłowych predykcji / liczba wszystkich predykcji)

W skrócie: jak rośnie precision wraz ze wzrostem confidence threshold.

##### 📌 Najważniejsze obserwacje:
✅ Ogólny trend:
- Dla wszystkich klas precision rośnie wraz z confidence threshold, co jest naturalnym zjawiskiem.
- Dla wszystkich klas razem (niebieska gruba linia) precision dochodzi do 1.00 przy threshold 0.975.
Oznacza to, że przy bardzo wysokim progu model popełnia prawie zero fałszywych detekcji, ale prawdopodobnie gubi sporo prawidłowych.

📈 Najlepsze klasy:
- Motorcycle, Red Traffic Light, Car — mają wysokie precision >0.9 już przy progach około 0.6–0.7.
- Prohibition Sign, Different Traffic Sign i Warning Sign również trzymają się dobrze.

⚠️ Problematyczne klasy:
- Pedestrian (czerwona linia) i Speed Limit Sign (szara linia) — precision wyraźnie niższe w całym zakresie, szczególnie pedestrian ledwo dobija do 0.7 przy wysokich progach.
- Truck — poprawny, ale nieco słabszy od topowych klas.

#### Krzywa precyzja-pewność
![](Documentation_Photos/PR_curve.png)

##### 📊 Co pokazuje ten wykres?
Wykres Precision-Recall pokazuje zależność między precyzją (precision) a czułością (recall) dla każdego z wykrywanych obiektów (klas) w różnych progach decyzyjnych.

- Precyzja (Precision) = Prawdziwe Pozytywy / (Prawdziwe Pozytywy + Fałszywe Pozytywy)\
→ Jak wiele z wykryć było trafnych.

- Czułość / Recall = Prawdziwe Pozytywy / (Prawdziwe Pozytywy + Fałszywe Negatywy)\
→ Jak wiele z rzeczywistych obiektów zostało wykrytych.

##### 📈 Jak interpretować krzywe?
- Krzywe PR dla każdej klasy (np. "car", "pedestrian", "speed limit sign") przedstawiają jakość wykrywania tej konkretnej klasy.

- Im bliżej górnego prawego rogu (1,1), tym lepsze wykrycie — oznacza to wysoką precyzję i recall jednocześnie.

- Linie bardziej "spłaszczone" lub schodzące stromo w dół pokazują trudność modelu w rozpoznaniu danej klasy.

##### 📊 Co mówi legenda?
W legendzie podano średnie wartości AP (Average Precision) dla każdej klasy oraz ogólne mAP@0.5 (mean Average Precision przy IoU = 0.5).

##### 🔧 Co to oznacza dla modelu YOLO?
- YOLO radzi sobie bardzo dobrze z wykrywaniem samochodów i świateł drogowych, ale gorzej z bardziej zmiennymi lub trudniejszymi klasami, jak znaki ograniczenia prędkości i piesi.

- Trudne klasy mogą wymagać:

  - Więcej zróżnicowanych danych treningowych.

  - Lepszych adnotacji.

  - Augmentacji danych (np. zmiany warunków oświetleniowych, kąta widzenia).

  - Dalszego fine-tuningu modelu.

##### 📌 Podsumowanie:
- Wykres PR pomaga zrozumieć dokładność i kompletność wykrycia dla każdej klasy.
- mAP@0.5 = 0.690 sugeruje, że model jest użyteczny, ale może być dalej optymalizowany.
- YOLO dobrze radzi sobie z bardziej wyrazistymi klasami (np. samochody), ale wymaga dopracowania przy mniej kontrastowych obiektach (np. znaki drogowe, piesi).

#### Analiza rozkładu adnotacji (bounding boxów) i ich klas
![](Documentation_Photos/labels.jpg)

##### 📊 Górny lewy wykres: Rozkład liczby instancji według klas
Co pokazuje?
- Liczbę wystąpień (instancji) każdej klasy w zbiorze treningowym.

Co z tego wynika?
- Najwięcej przykładów to: car (samochód) i Different Traffic Sign.
- Klasy takie jak Green Traffic Light, Speed Limit Sign, Truck mają znacznie mniej danych.
- Taki nierównomierny rozkład klas może prowadzić do:
  - Gorszego uczenia się klas rzadziej reprezentowanych.
  - Dominacji klas licznych w decyzjach modelu.

##### 📏 Górny prawy wykres: Rozmiary i pozycje bounding boxów
To zestaw nakładających się ramek (bounding boxes), pokazujący typowe pozycje i wymiary obiektów. Wskazuje:
- że większość obiektów znajduje się blisko centrum obrazu,
- że mają stosunkowo małe rozmiary (co jest typowe np. dla znaków drogowych widocznych z daleka).

##### 📍 Dolny lewy wykres: Rozkład pozycji bounding boxów (x/y)
Pokazuje zagęszczenie pozycji środków ramek na osi x i y (w skali względnej obrazu, 0–1):
- Skupienie wokół środka, szczególnie w pionie (y ≈ 0.6–0.8), sugeruje, że obiekty często pojawiają się na poziomie drogi (tam, gdzie są samochody, piesi itp.).
- Rzadsze występowanie przy krawędziach obrazu.

##### 📐 Dolny prawy wykres: Rozkład szerokości i wysokości bounding boxów
Wskazuje:
- większość ramek ma małą szerokość i wysokość (0.0–0.2), czyli obiekty są zazwyczaj małe w stosunku do całego obrazu.
- istnieje niewielka liczba bardzo dużych ramek, co może oznaczać duże obiekty blisko kamery (np. samochody z bliska).

##### 🔎 Jak to wpływa na trening YOLO?
✅ Co działa dobrze:
- Dużo danych dla klas takich jak "car" zapewnia dobrą jakość detekcji tej klasy (co widać na PR curve).
- Obiekty są dobrze skupione – YOLO lubi „gęste”, regularne dane.

❗ Problemy:
- Nierównowaga klas → niższe AP dla klas z małą liczbą instancji.
- Małe obiekty → YOLO może mieć problem z ich detekcją, jeśli nie są dobrze reprezentowane.
- Centralne skupienie obiektów → model może mieć trudność z obiektami na obrzeżach obrazu.

#### Pairplot (scatter matrix)
![](Documentation_Photos/labels_correlogram.jpg)

##### 🔍 Co pokazuje ten wykres?
1. Histogramy na przekątnej – Rozkłady pojedynczych cech:
- x – większość ramek znajduje się blisko środka obrazu (wartości około 0.5),

- y – skupienie wokół 0.6–0.8, czyli dolna część obrazu (zgodne z umiejscowieniem dróg),

- width i height – większość ramek ma małe rozmiary (poniżej 0.2), czyli obiekty są małe.

2. Scatterploty poza przekątną – Zależności między parami cech:

✅ width vs height
- Widoczna grupa małych obiektów (dół wykresu).
- Kilka większych ramek (mogą to być bliskie samochody).

✅ x vs width oraz y vs height
- Kształt rozproszony, ale niejednorodny – sugeruje, że rozmiary ramek mogą zależeć od pozycji w obrazie.

- np. większe obiekty częściej pojawiają się niżej (w y > 0.6), co jest typowe dla perspektywy kamery drogowej (samochód blisko → duży obiekt).

##### 🧠 Wnioski dla YOLO i trenowania modelu
1. Potwierdzenie: obiekty są głównie małe i znajdują się na dole obrazu.
- Model YOLO może mieć problemy z małymi obiektami – warto dobrać odpowiednią rozdzielczość wejściową i anchor boxy.
2. Rozrzut wielkości jest niewielki, ale istnieje.
- Użycie zbyt małej liczby anchor boxów może obniżyć skuteczność → warto dopasować anchory (np. przez k-means na width/height).
3. Nieliniowość zależności
- np. width i x nie są niezależne → model może korzystać z tych zależności przy predykcji.

#### Mean Average Precision (mAP)
![](Documentation_Photos/MeanAveragePrecision.png)

##### 📊 Co oznaczają te metryki?
| Metryka                    | Znaczenie                                                             |
| -------------------------- | --------------------------------------------------------------------- |
| **mAP\@0.5**               | Średnia precyzja przy IoU ≥ 0.5 (najczęściej cytowana metryka).       |
| **mAP\@0.75**              | Średnia precyzja przy bardziej wymagającym progu IoU ≥ 0.75.          |
| **mAP\@0.5:0.95**          | Średnia z mAP dla progów od 0.5 do 0.95 co 0.05 (COCO standard).      |
| **Small / Medium / Large** | Podział wg rozmiaru obiektu w pikselach (wg COCO lub YOLO definicji). |

##### 🔍 Co widać na wykresie?
📌 Ogólna jakość:
- mAP@0.5 = 0.65 – dobry wynik.
- mAP@0.5:0.95 = 0.40 – pokazuje spadek precyzji przy wyższych wymaganiach (normalne).
- mAP@0.75 = 0.46 – przeciętny wynik, ale akceptowalny.

📌 Zależność od wielkości obiektu:

| Rozmiar | mAP\@0.5:0.95 | mAP\@0.5 | mAP\@0.75 | Wnioski                                         |
| ------- | ------------- | -------- | --------- | ----------------------------------------------- |
| Small   | **0.30**      | 0.53     | 0.32      | 🔻Najgorszy wynik – małe obiekty są trudne.     |
| Medium  | **0.51**      | 0.76     | 0.67      | ✅ Najlepszy wynik – model radzi sobie świetnie. |
| Large   | **0.43**      | 0.58     | 0.48      | ❕ Lepsze niż small, ale gorsze niż medium.      |

##### 🧠 Wnioski i rekomendacje:
📉 Słaba detekcja małych obiektów
- to typowe dla YOLO.

📈 Najlepsza jakość dla obiektów średnich
- Obiekty średniej wielkości są dobrze reprezentowane i łatwiejsze do detekcji.

❗ Duże obiekty nie są wykrywane idealnie
- Możliwa przyczyna: niedoreprezentowanie lub zbyt duże rozmiary wykraczające poza ramkę obrazu.

✅ Podsumowanie:
- Model jest skuteczny, szczególnie dla obiektów średniej wielkości.
- Główna słabość to małe obiekty – można to poprawić dopasowując architekturę, anchory, augmentacje i rozdzielczość wejścia.

#### Zbiór wykresów uczenia
![](Documentation_Photos/results.png)

##### 🧠 Straty treningowe (górny rząd, 3 pierwsze wykresy):
1. train/box_loss – Strata związana z dokładnością lokalizacji (pozycji) wykrywanego obiektu.
- W kontekście ruchu drogowego: niższa strata oznacza, że model coraz lepiej określa ramki ograniczające pojazdy, pieszych, znaki itp.
2. train/cls_loss – Strata klasyfikacyjna, określająca jak dobrze model rozpoznaje klasy obiektów (np. samochód, rower, pieszy).
- Dobrze wytrenowany model powinien umieć rozróżnić różne typy uczestników ruchu.
3. train/dfl_loss – Strata „Distribution Focal Loss”, używana do precyzyjniejszego przewidywania granic ramek (bardziej szczegółowe dopasowanie).
- W przypadku ścisłej detekcji (np. pojazdów na zatłoczonej drodze), niski dfl_loss jest kluczowy.

Wszystkie trzy straty maleją, co świadczy o skutecznym procesie uczenia.

##### 🧠 Metryki precyzji (górny rząd, 2 ostatnie wykresy):
4. metrics/precision(B) – Precyzja: jak wiele wykryć było trafnych (czyli niskie False Positive).

- W ruchu drogowym ważna np. przy wykrywaniu pieszych — nie chcemy fałszywych alarmów.
5. metrics/recall(B) – Czułość: jak wiele prawdziwych obiektów zostało wykrytych (niskie False Negative).
- Kluczowa przy detekcji zagrożeń: lepiej wykryć za dużo niż przeoczyć np. pieszego.

Obie metryki rosną, co świadczy o poprawiającej się jakości detekcji.

##### ✅ Straty walidacyjne (dolny rząd, 3 pierwsze wykresy):
6. val/box_loss
7. val/cls_loss
8. val/dfl_loss

Odpowiedniki strat treningowych, mierzone na zbiorze walidacyjnym.
- Jeżeli wykresy treningowe i walidacyjne są zbieżne (a są), oznacza to brak przeuczenia (overfitting).

##### 📈 mAP – mean Average Precision (dolny rząd, 2 ostatnie wykresy):
9. metrics/mAP50(B) – Średnia precyzja przy IoU=0.5. Miara ogólnej skuteczności detekcji.
10. metrics/mAP50-95(B) – Średnia precyzja przy różnych progach IoU (0.5–0.95). Bardziej rygorystyczna metryka.
- Dobrze rosnące wartości mAP oznaczają, że model skutecznie i dokładnie wykrywa różne obiekty (np. pojazdy w ruchu, sygnalizację świetlną).

##### 📌 Wnioski ogólne:
- Model skutecznie się uczy: straty spadają, metryki rosną.
- Brak oznak przeuczenia: trening i walidacja postępują równolegle.
- mAP50 na poziomie ~0.8 i mAP50-95 na poziomie ~0.42 to dobre wyniki jak na detekcję złożonych scen ulicznych.

---

## 🚀 Wdrożenie modelu i monitorowanie

### Wdrożenie modelu trenowanego w Roboflow:

Utworzyliśmy `Workflow` który pozwala otrzymać oznaczone zdjęcie z dowolnego przesłanego zdjęcia:

![](Documentation_Photos/img_9.png)

Otrzymane oznaczenia:

![](Documentation_Photos/img_10.jpg)

[Roboflow](https://roboflow.com/brand) pozwala również dynamicznie dostosowywać Threshold:

![](Documentation_Photos/img_11.png)

[Link do naszego Workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiR2xwOWJmVVV2VVNReWI3SG9PbWMiLCJ3b3Jrc3BhY2VJZCI6ImVNZjdxdm9acGJjVXZXNGpuNnV5SURPeFhydDEiLCJ1c2VySWQiOiJlTWY3cXZvWnBiY1V2VzRqbjZ1eUlET3hYcnQxIiwiaWF0IjoxNzQ5NjQ5MzczfQ.tHYD4L38s1W3lf4Dnx4jB3e_FFSxJ8KrMWCDqtK1u9I)

### Wdrożenie modelu trenowanego w Collabie:

Najpierw przetestowaliśmy oznaczanie bezpośrednio w Collabie. Niżej przedstawiamy przykłady oznaczeń:

Bez przeskalowania:

![](Documentation_Photos/img_12.png)

Z przeskalowaniem:

![](Documentation_Photos/img_13.png)

Potem utworzyliśmy projekt lokalnie, z użyciem `venv`.

Zainstalowaliśmy potrzebne biblioteki i pakiety:
```
pip install git+https://github.com/sunsmarterjie/yolov12.git
pip install supervision streamlit  
pip install opencv-python-headless 
pip install huggingface_hub 
```

Napisaliśmy prostą aplikację webową w `streamlit` i uruchomiliśmy:

```commandline
streamlit run app.py
```

![](Documentation_Photos/img_10.png)

Oznaczone zdjęcie:

![](Documentation_Photos/img_14.png)

---