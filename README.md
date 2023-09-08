<p  style="font-size: 30px; text-align: center; "><b>Predicting Analysis Cardiovascular Disease</b></p>

Nama : Muhammad Reesa Rosyid
Email : mreesa669@gmail.com

<h1  align='center'>Project Domain</h1>

***

## Background
Prakiraan epidemiologi menunjukkan akan terjadi lonjakan global penyakit kardiovaskular, dengan meningkatnya jumlah individu yang berisiko tinggi, karena studi kohort baru-baru ini mengungkapkan bahwa sangat sedikit, sekitar 2% hingga 7%, pada populasi umum yang tidak memiliki faktor risiko, sementara lebih dari 70% individu yang berisiko menunjukkan beberapa faktor risiko [1]. Kejadian kardiovaskular tradisional terutama didorong oleh faktor risiko yang dapat dimodifikasi seperti hipertensi, dislipidemia, diabetes, merokok, obesitas, perilaku kurang gerak, dan riwayat keluarga dengan penyakit kardiovaskular. Meskipun pengelolaan faktor-faktor ini diketahui dapat menurunkan prevalensi penyakit kardiovaskular dan angka kematian terkait secara signifikan, namun angka kejadian kardiovaskular yang meningkat juga meningkat. sayangnya faktor risiko di kalangan anak-anak dan remaja ini semakin memperburuk masalah [2]. Penelitian tertentu telah menyoroti hubungan antara kasus-kasus tersebut dan faktor-faktor seperti obesitas dan peningkatan BMI (Indeks Massa Tubuh), dengan obesitas bertindak sebagai faktor risiko independen untuk kejadian Penyakit Jantung Koroner (PJK) dan Penyakit Kardiovaskular (CVD), serta peningkatan jaringan lemak. jaringan dikaitkan dengan kematian secara keseluruhan [3]. Analisis ini mencakup validasi tindakan perawatan diri tertentu seperti pilihan makanan dan aktivitas fisik, hambatan dalam perawatan diri, dan dampak perawatan diri terhadap peningkatan hasil, serta mengeksplorasi data pendukung untuk berbagai metode yang berpusat pada individu, keluarga, dan komunitas untuk meningkatkan perawatan diri; Meskipun terdapat interaksi yang rumit antara perawatan mandiri dan hasilnya, bukti kuat menggarisbawahi efektivitas perawatan mandiri dalam mewujudkan tujuan rencana pengobatan, menggarisbawahi pentingnya hal ini [4]. Oleh karena itu, muncul kebutuhan mendesak akan teknologi berbasis AI yang mampu memprediksi dengan cepat dan andal kemungkinan kondisi masa depan individu yang terkena penyakit kardiovaskular, dengan Internet of Things (IoT) yang mendorong kemajuan dalam prognosis penyakit kardiovaskular, dan pembelajaran mesin (ML). dimanfaatkan untuk menganalisis dan memprediksi hasil menggunakan data dari perangkat IoT [5].

<h1  align='center'>Business Understanding</h1>

***

## Problem Statements

1. Dari fitur-fitur pengaruh resiko penyakit cardiovascular, bagaimanakah pengaruhnya terhadap resiko cardiovascular?
2. Bagaimana cara membantu tenaga medis dalam menglasifikasi pasien berdasarkan resiko mereka sehingga perawatan dapat dipersonalisasi?

## Goals
1. Mengetahui pengaruh variabel resiko yang ada terdahap penyakit cardiovascular pada tahap exploratory data analysis.
2. Membuat model machine learning yang dapat mengklasifikasian pasien berdasarkan resiko mereka.

<h1  align='center'>Data Understanding</h1>

***

Kumpulan Data Prediksi Risiko Penyakit Kardiovaskular adalah kumpulan data medis dan demografi yang dikurasi untuk memfasilitasi pengembangan dan evaluasi model prediktif penyakit kardiovaskular (CVD). Penyakit kardiovaskular mencakup berbagai kondisi yang mempengaruhi jantung dan pembuluh darah, seperti penyakit arteri koroner, gagal jantung, stroke, dan hipertensi. Penyakit-penyakit ini menimbulkan beban kesehatan global yang signifikan, berkontribusi terhadap sejumlah besar kasus kesakitan dan kematian setiap tahunnya.

  

Kumpulan data ini bertujuan untuk mengatasi meningkatnya kebutuhan akan alat yang akurat dan andal untuk menilai risiko seseorang terkena penyakit kardiovaskular. Prediksi risiko yang dini dan tepat sangat penting untuk memungkinkan intervensi yang tepat waktu dan strategi medis yang dipersonalisasi, yang pada akhirnya mengarah pada peningkatan hasil pasien dan pengurangan dampak sosial dari penyakit-penyakit ini.

  

Dataset ini dapat diunduh pada halaman berikut: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Kumpulan data berisi tipe data float64(1), int64(12) dan 70000 catatan. Penjelasan fitur-fitur yang ada pada dataset dapat dilihat pada penjelasan fitur dibawah ini:

1. Age: Umur dihitung menggunakan hari.
2. Gender: Menunjukan jenis kelamin biologis individu, dengan pilihan 1 = laki-laki dan 2 = perempuan.
3. Height: Mencatat tinggi individu dalam cm.
4. Weight: Mencatat berat individu dalam kg.
5. Systolic blood pressure: Tekanan tertinggi yang dicapai saat otot jantung berkontraksi.
6. Diastolic blood pressure: Tekanan tertinggi saat otot jantung relaksasi.
7. cholesterol: Menunjukan tingkat kolesterol dengan pilihan 1 = Normal, 2 = Di atas normal, dan 3 = jauh di atas normal.
8. Glucose: Menunjukan tingkan glukosa dalam darah dengan pilihan 1 = Normal, 2 = Di atas normal, dan 3 = jauh di atas normal.
9. Smoking: Riwayat merokok 0 = Tidak merokok dan 1 = Merokok.
10. Alcohol intake: Riwayat alkohol 0 = Tidak minum alkohol dan 1 = Peminum alkohol.
11. Physical activity: Riwayat olahraga, 0 = Tidak pernah dan 1 = Pernah olahraga.
12. Cardio: Ada atau tidak potensi terkena penyakit cardiovascular.
## EDA
### Data Cleaning
Dataset tersebut akan dilakukan pengecekan missing value dan duplicate value. Dari hasil pengecekan tersebut dataset dapat dikatakan bersih dikarenakan tidak ada indikasi missing value dan duplicate value. Langkah selanjutnya akan meilhat distribusi data menggunakan histogram dan boxplot.
![Alt text](picture/distribusidata.png)
![Alt text](picture/boxplotbefore.png)

Dari distribusi data di atas, kemungkinan data outlier akan dibersihkan dengan Teknik IQR. IQR (Interquartile Range) adalah salah satu teknik statistik yang digunakan untuk mengukur sebaran data dalam sebuah kumpulan data atau distribusi. IQR biasanya digunakan dalam analisis data untuk mengidentifikasi nilai-nilai yang dianggap sebagai outliers atau pencilan. Teknik IQR melibatkan langkah-langkah berikut:
1. Mengurutkan Data: Pertama, data harus diurutkan dari yang terkecil hingga yang terbesar atau sebaliknya, tergantung pada preferensi.
2. Menentukan Quartile Pertama (Q1) dan Quartile Ketiga (Q3): Quartile pertama (Q1) adalah nilai yang berada di tengah-tengah data setelah diurutkan, yang membagi data menjadi dua bagian dengan 25% dari data di bawahnya. Quartile ketiga (Q3) adalah nilai yang berada di tengah-tengah data yang sama, tetapi membagi data menjadi dua bagian dengan 75% dari data di bawahnya.
3. Menghitung IQR: IQR adalah selisih antara Q3 dan Q1, yaitu IQR = Q3 - Q1.
4. Menentukan Batas Atas dan Batas Bawah: Batas atas (Upper Bound) adalah nilai Q3 ditambah 1,5 kali IQR, sedangkan batas bawah (Lower Bound) adalah nilai Q1 dikurangi 1,5 kali IQR.
5. Mengidentifikasi Pencilan (Outliers): Nilai-nilai yang berada di luar batas atas dan batas bawah dianggap sebagai pencilan atau outliers.

$$\text{IQR} = Q3 - Q1$$

Berikut ini merupakan hasil data yang sudah dihapus kemungkinan outliersnya.

![Alt text](picture/boxplotafter.png)

### Feature Age

Akan dilakukan penghitungan value pada feature umur agar diketahui rentang umur yang terdapat di dalam dataset.

```python
df['age'] = df['age'].apply(lambda x: x // 365)
df['age'] = df['age'].astype(int)
```

![Alt text](picture/agecounts.png)

Umur yang terdapat pada dataset tersebut ada di antara umur 33 sampai 64. Selanjutnya akan dilihat umur berapa orang biasanya terkena penyakit kardiovaskular.

![Alt text](picture/agecardiocounts.png)

Data di atas menunjukan bahwa pada umur 39 ke atas orang mulai terkena penyakit kardiovaskular dan akan meningkat seiring bertambahnya usia.

### Feature Gender

Akan dilihat perbandingan laki-laki atau perempuankah yang banyak mengalami penyakit cardiovascular.

![Alt text](picture/gender.png)

Dilihat dari perbandingan antara laki-laki dan perempuan yang terkena kardiovaskular, laki-laki lebih banyak yang mengalami penyakit kardiovaskular dari pada perempuan dengan perbandingan 64.7:35.5.

### Feature Cholesterol & Glucosa

Akan dilihat tingkat kolesterol dan glukosa pasien terhadap penyakit cardiovascular.

![Alt text](picture/chol.png)

![Alt text](picture/gluc.png)

Dilihat dari hasil pengukuran kadar gula darah dan kolesterol untuk pasien penderita kardiovaskular, semua hasil gula darah dan kolesterol berpeluang mengalami penyakit kardiovaskular.

### Feature Smoking, Alcohol, & Active

Pada feature kebiasaan smoking, alcohol, dan active akan dilihat apakah terdapat pengaruh terhadap penyakit kardiovaskular.

![Alt text](picture/smoke.png)

Dapat dilihat pada plot bar di atas, orang yang tidak merokok dan yang merokok sama-sama memiliki resiko penyakit kardiovaskular.

![Alt text](picture/alco.png)

Begitu pula pada peminum alcohol, peminum maupun tidak peminum sama-sama memiliki resiko terhadap penyakit kardiovaskular.

![Alt text](picture/activity.png)

Sedangkan pada feature active yang diukur apakah orang sering berolahraga atau tidak, nyatanya orang yang tidak pernah berolahraga memiliki faktor resiko terkena penyakit kardiovaskular yang lebih tinggi dari pada orang yang senang berolahraga.

### Feature BMI & Obessitas

BMI atau Body Mass Index (Indeks Massa Tubuh) adalah sebuah angka yang digunakan untuk mengukur hubungan antara berat badan seseorang dengan tinggi badannya. BMI dapat memberikan gambaran apakah berat badan seseorang proporsional dengan tinggi badannya atau apakah terdapat risiko kelebihan berat badan atau kurang berat badan.

Rumus menghitung BMI adalah:
$$\text{BMI} = \frac{\text{weight (kg)}}{\text{height (m)}^2}$$

Nilai-nilai BMI biasanya dikelompokkan ke dalam kategori-kategori sebagai berikut:
* BMI < 18.5: Kekurangan berat badan
* 18.5 ≤ BMI < 24.9: Berat badan normal
* 25 ≤ BMI < 29.9: Kelebihan berat badan
* BMI ≥ 30: Obesitas

Untuk itu akan dibuatkan feature baru yaitu BMI dari penghitungan rumus di atas.

```python
def hitung_bmi(tinggi, berat):
    tinggi_m = tinggi/100 # konversi ke m
    bmi = berat/ (tinggi_m**2)
    return bmi

# Applying the function
df['BMI'] = df.apply(lambda x: hitung_bmi(x['height'], x['weight']), axis=1)
```

Setelah mendapatkan hasil BMI akan di kategorikan BMI tersebut ke dalam feature baru bernama obes dengan empat golongan yaitu normal, kekurangan berat badan, kelebihan berat badan, dan obesitas.

```python
df['obes'] = df['BMI'].apply(lambda x: 3 if x >= 30 else (2 if 25 < x <= 29.9 else (1 if 18.5 <= x < 24.9 else 0)))
```

Selanjutkan akan dilihat dari empat golongan tersebut yang paling banyak terkena penyakit kardiovaskular.

![Alt text](picture/BMI.png)

Ternyata dari hasil perhitungan yang terkena penyakit kardiovaskular dari empat golongan tersebut, orang yang kelebihan berat badan dan obesitas jauh lebih beresiko terkena penyakit kardiovaskular.

### Feature Blood Pressure

Tekanan darah adalah ukuran tekanan yang dihasilkan oleh aliran darah melalui pembuluh darah dalam tubuh manusia. Tekanan darah terutama diukur dalam dua nilai, yaitu tekanan sistolik dan tekanan diastolik, dan dinyatakan dalam milimeter air raksa (mmHg). Biasanya, tekanan darah direkam dalam bentuk dua angka, seperti "120/80 mmHg".
1. Tekanan Sistolik: Ini adalah angka pertama dalam pengukuran tekanan darah dan mencerminkan tekanan pada saat jantung berkontraksi atau memompa darah ke dalam arteri. Angka ini mengukur tekanan tertinggi dalam siklus denyut jantung.
2. Tekanan Diastolik: Ini adalah angka kedua dalam pengukuran tekanan darah dan mencerminkan tekanan pada saat jantung beristirahat di antara kontraksi. Angka ini mengukur tekanan terendah dalam siklus denyut jantung.

Tekanan darah penting untuk kesehatan manusia karena dapat memberikan informasi tentang kesehatan sistem kardiovaskular. Tekanan darah yang tinggi (hipertensi) dapat meningkatkan risiko penyakit jantung, stroke, dan masalah kesehatan lainnya, sementara tekanan darah yang terlalu rendah (hipotensi) juga dapat menyebabkan masalah kesehatan.

Tekanan darah normal, hipotensi, dan hipertensi adalah berbagai kisaran tekanan darah yang digunakan untuk mengukur kesehatan kardiovaskular seseorang. Berikut adalah penjelasan singkat tentang ketiganya:
1. Tekanan Darah Normal:
    * Sistolik: Biasanya kurang dari 120 mmHg.
    * Diastolik: Biasanya kurang dari 80 mmHg.
2. Hipotensi (Tekanan Darah Rendah):
    * Sistolik: Kurang dari 90 mmHg.
    * Diastolik: Kurang dari 60 mmHg.
3. Hipertensi (Tekanan Darah Tinggi):
    * Sistolik: 121 mmHg atau lebih tinggi.
    * Diastolik: 81 mmHg atau lebih tinggi.

Dari informasi di atas akan dikelompokan blood pressure menggunakan acuan tersebut.

```python
df["bld_pres"] = df.apply(lambda row: 1 if 60 <= row["ap_lo"] <= 80 and 90 <= row["ap_hi"] <= 120
                                  else (2 if row["ap_lo"] > 80 and row["ap_hi"] > 120
                                        else (0 if row["ap_lo"] < 60 and row["ap_hi"] < 90
                                              else None)), axis=1)
```

Setelah dilihat lagi ternyata terdapat missing value karena terdapat tekanan darah sistolik atau diastolik yang salah satunya tinggi dan satunya lagi normal. Untuk itu akan dilakukan fillna dengan nilai 3. Nilai 3 ini berarti orang tersebut mengalami hipertensi sistolik atau hipertensi diastolik. Itu berarti salah satu tekanan darah sistolik atau diastoliknya saja yang tinggi.

Selanjutnya akan dibuat plot berupa perbandingan penderita kardiovaskular yang memiliki tekanan darah normal, hipertensi, dan hipertensi sistolik atau diastolik.

![Alt text](picture/bloodpress.png)

Dapat disimpulkan bahwa orang yang bertekanan darah tinggi lebih banyak yang menderita penyakit kardiovaskular dari pada orang yang bertekanan darah rendah dengan perbandingan 44.2% (tekanan darah normal) dan 55.8% (tekanan darah tinggi).

### Korelasi antara setiap feature

![Alt text](picture/corr.png)

Dapat dilihat pada plot segitiga korelasi di atas feature cardio memiliki korelasi positif pada feture usia, gender, weight, ap_hi, ap_lo, cholesterol, glucose, BMI, obes, dan bld_press. Hubungan dari feature tersebut dapat dikatakan cukup rendah hingga sedang. Sedangkan pada feature height, smoke, alco, dan active memiliki korelasi negatif sehingga dapat dikatakan variabel tersebut tidak berkorelasi dengan cardio.

<h1 align='center'>Data Preparation</h1>

***

## Split dan Scaling Data

Data akan di bagi dengan proporsi 80 data training dan 20 data testing. Data yang telah dilakukan pembagian selanjutnya akan melalui proses scaling.

Scaling data adalah proses mengubah rentang (skala) dari data numerik dalam suatu dataset sehingga memiliki skala yang seragam atau sebanding. Tujuan utama dari scaling data adalah untuk memastikan bahwa variabel-variabel dalam dataset memiliki pengaruh yang seimbang pada model atau analisis yang sedang dilakukan. Ini sangat penting dalam banyak algoritma pembelajaran mesin dan analisis data karena banyak algoritma sensitif terhadap perbedaan skala antara variabel-variabelnya.

Teknik yang dilakukan pada scaling adalah Min-Max Scaling (Normalization):
- Min-Max scaling mengubah data sehingga nilainya berada dalam rentang tertentu, seringkali [0, 1].
- Metode ini baik digunakan ketika ingin menjaga informasi relatif antara variabel tetapi menginginkan skala yang seragam.
Rumus Min Max Scaling:
    $$X_{\text{new}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

<h1 align='center'>Modeling</h1>

***

Alogritma yang akan dipilih sebagai pendeteksi resiko kardiovaskular adalah:
1. Logistic Regression: Ini adalah salah satu algoritma klasifikasi yang paling sederhana dan populer. Ini dapat digunakan untuk memprediksi probabilitas seseorang mengalami risiko kardiovaskular berdasarkan sejumlah fitur seperti usia, tekanan darah, kolesterol, dan lainnya.
2. K-Nearest Neighbors (KNN): KNN adalah algoritma yang sederhana namun efektif untuk klasifikasi. Ini bekerja dengan mencari k-tetangga terdekat dari setiap data poin dan memilih label mayoritas dari tetangga-tetangga tersebut.
3. Random Forest: Ini adalah algoritma ensemble yang kuat yang dapat digunakan untuk klasifikasi. Random Forest dapat mengatasi masalah overfitting dan memiliki kemampuan untuk menangani banyak fitur dengan baik. Ini dapat digunakan untuk memodelkan hubungan kompleks antara berbagai faktor risiko.
4. Gradient Boosting: Algoritma seperti XGBoost, LightGBM, atau CatBoost adalah pilihan yang baik jika ingin meningkatkan performa model. Mereka bekerja dengan menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi prediksi.

Ke empat algoritma tersebut akan dicoba dan akan dipilih satu algoritma yang memberikan akurasi paling baik dan waktu runtime yang tidak terlalu lama.

| Algorithm | Accuracy | Runtime |
|:--------:|:-------:|:--------:|
| Logistic Regression   | 0.729071  | 	0.126529   |
| 	K-Nearest Neighbors     | 0.691571  | 0.127531    |
| Random Forest    | 0.705786  | 8.856894    |
| XGBoost     | 0.733000  | 0.793133    |

Setelah dilakukan pengetesan dari ke empat algoritma tersebut, XGBoost memberikan akurasi dan waktu runtime paling baik dari pada ketiga algoritma lainnya dengan akurasi 73,3% dan waktu runtime 0.86s. 

Model xgboost akan dilakukan tunning hyperparameter dengan menggunakan GridSearchCV untuk mengetahui parameter mana yang dapat meningkatkan hasil akurasi model.

Didapatkan best parameter untuk model xgboost dengan Best Parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150} dan mendapatkan akurasi 74% dari yang awalnya hanya 73%.

<h1 align='center'>Evaluate Model</h1>

## Logloss

Log Loss (Logarithmic Loss) adalah metrik evaluasi yang digunakan untuk mengukur kinerja model klasifikasi yang menghasilkan probabilitas sebagai output. Log Loss mengukur sejauh mana probabilitas yang diberikan oleh model cocok dengan hasil aktual.

Log Loss didefinisikan sebagai:
$$\text{Log Loss} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right]$$

- N adalah jumlah sampel dalam dataset
- yi adalah label sebenarnya dari sampel ke-i (0 atau 1 dalam masalah klasifikasi biner).
- pi adalah probabilitas yang diberikan oleh model untuk sampel ke-i milik kelas positif (label 1).

Tujuan utama adalah untuk mengurangi Log Loss sebanyak mungkin, yang berarti model memberikan probabilitas yang mendekati 0 atau 1 untuk sampel yang benar-benar milik kelas positif atau kelas negatif.

![Alt text](picture/logloss.png)

Dapat dilihat dari log loss curve di atas, model menunjukan hasil yang baik karena memiliki nilai logloss akhir yang renda.

## Classification Report & Confussion Matrix

| | precision | recall | f1-score | support |
|:--------:|:-------:|:--------:|:--------:|:--------:|
|0|0.72|0.78|0.75|6988|
|1|0.76|0.70|0.73|7012|
||||||
| accuracy |||0.74|14000|
|macro avg|0.74|0.74|0.74|14000|
|weighted avg|0.74|0.74|0.74|14000|

![Alt text](picture/cm.png)

Berikut ini penjabaran masing-masing nilai pada report:
- Precision untuk kelas 0: 0,72
Precision untuk kelas 1: 0,76
Presisi adalah rasio kejadian kelas 0 yang diprediksi dengan benar terhadap total kejadian kelas 0 yang diprediksi. Dalam hal ini, dari semua kejadian yang diprediksi sebagai kelas 0, 72% dan 76% pada kelas 1 adalah benar.
- Recall untuk kelas 0: 0,78
Recall untuk kelas 1: 0,70
Perolehan kembali (juga dikenal sebagai sensitivitas atau tingkat positif sebenarnya) adalah rasio kejadian kelas 0 yang diprediksi dengan benar terhadap total kejadian aktual kelas 0. Dalam hal ini, 78% dari semua kejadian aktual kelas 0 diprediksi dengan benar dan untuk kelas 1 70% diprediksi dengan benar.
- Skor F1 untuk kelas 0: 0,75
Skor F1 untuk kelas 1: 0,73
Skor F1 adalah rata-rata harmonik antara presisi dan perolehan. Ini memberikan metrik tunggal yang menyeimbangkan presisi dan perolehan. Skor F1 sebesar 0,75 untuk kelas 0 menunjukkan keseimbangan yang baik antara presisi dan perolehan untuk kelas 0. Skor F1 untuk kelas 1 ialah sebesar 73% menunjukan pula keseimbangan yang baik antara presisi dan perolehan untuk kelas 1.
- Dukungan untuk kelas 0: 6988
Dukungan untuk kelas 1: 7012
Dukungannya adalah jumlah instance masing masing kelas sebenarnya dalam kumpulan data.
- "Akurasi" adalah ukuran kinerja model secara keseluruhan dan sama dengan 0,74, yang berarti bahwa model mengklasifikasikan 74% dari total contoh dengan benar.
- "Rata-rata makro" menghitung rata-rata presisi, perolehan, dan skor F1 di kedua kelas. Dalam hal ini, nilainya 0,74.
- "Rata-rata tertimbang" menghitung rata-rata presisi, perolehan, dan skor F1, yang diberi bobot berdasarkan jumlah instance untuk setiap kelas. Ini juga 0,74 dalam kasus ini.
- "Rata-rata tertimbang" menghitung rata-rata presisi, perolehan, dan skor F1, yang diberi bobot berdasarkan jumlah instance untuk setiap kelas. Ini juga 0,74 dalam kasus ini.
- Secara keseluruhan, tampaknya model memiliki performa yang cukup baik dengan presisi, perolehan, dan skor F1 yang serupa untuk kedua kelas. Akurasi sebesar 74% menunjukkan bahwa ia membuat prediksi yang akurat pada sebagian besar kasus. Namun, interpretasi spesifik dari metrik ini bergantung pada konteks masalah klasifikasi dan prioritas terkait presisi dan perolehan.

## ROC

ROC AUC (Receiver Operating Characteristic - Area Under the Curve) adalah metrik evaluasi yang digunakan untuk mengukur kinerja model klasifikasi, khususnya dalam konteks klasifikasi biner. Metrik ini mengukur sejauh mana model mampu membedakan antara dua kelas (biasanya kelas positif dan kelas negatif) dan seberapa baik model mengklasifikasikan instansi positif lebih tinggi daripada negatif.
- Receiver Operating Characteristic (ROC) adalah sebuah kurva yang menggambarkan hubungan antara tingkat True Positive Rate (TPR) dan tingkat False Positive Rate (FPR) pada berbagai ambang batas (threshold) pengambilan keputusan. TPR juga dikenal sebagai Sensitivity atau Recall, sedangkan FPR adalah 1 minus Specificity.
- Area Under the Curve (AUC) adalah area di bawah kurva ROC. Nilai AUC berkisar antara 0 hingga 1, di mana nilai 0.5 menunjukkan performa yang sama dengan model acak, dan nilai mendekati 1 menunjukkan performa yang sangat baik.

![Alt text](picture/roc.png)

Dilihat pada ROC-AUC plot tersebut nilai ROC AUC > 0.5 yang dimana dapat dikatakan baik dalam membedakan antara kelas positif dan negatif.

<h1 align='center'>References</h1>

***
[1]	B. Dahlöf, “Cardiovascular Disease Risk Factors: Epidemiology and Risk Assessment,” Am J Cardiol, vol. 105, no. 1, pp. 3A-9A, Jan. 2010, doi: 10.1016/J.AMJCARD.2009.10.007.

[2]	T. V. Jardim et al., “Multiple cardiovascular risk factors in adolescents from a middle-income country: Prevalence and associated factors,” PLoS One, vol. 13, no. 7, Jul. 2018, doi: 10.1371/journal.pone.0200075.

[3]	A. Khosravi et al., “The Relationship between Weight and CVD Risk Factors in a Sample Population from Central Iran (Based on IHHP).”

[4]	B. Riegel et al., “Self-care for the prevention and management of cardiovascular disease and stroke: A scientific statement for healthcare professionals from the American heart association,” J Am Heart Assoc, vol. 6, no. 9, Sep. 2017, doi: 10.1161/JAHA.117.006997.

[5]	S. Subramani et al., “Cardiovascular diseases prediction by machine learning incorporation with deep learning,” Front Med (Lausanne), vol. 10, 2023, doi: 10.3389/fmed.2023.1150933.
