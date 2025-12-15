import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# FUNGSI BMI & LABEL
# ============================================================
def hitung_bmi(weight, height):
    tinggi_m = height / 100
    return weight / (tinggi_m ** 2)

def label_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# ============================================================
# VISUALISASI POHON KEPUTUSAN SEDERHANA
# ============================================================
def extract_tree_thresholds(model):
    tree = model.tree_
    thresholds = tree.threshold[tree.threshold != -2]
    return sorted(set(round(t, 2) for t in thresholds))

def visualize_tree_pretty(thresholds):
    if len(thresholds) < 3:
        return "digraph BMI_Tree { A [label='Pohon terlalu dangkal']; }"

    dot = f"""
    digraph BMI_Tree {{
        rankdir=TB;
        node [shape=box, style="rounded,filled", fontname="Arial"];

        A [label="BMI ≤ {thresholds[0]}?"];
        B [label="BMI ≤ {thresholds[1]}?"];
        C [label="BMI ≤ {thresholds[2]}?"];

        U [label="Underweight", fillcolor="#C6EFCE"];
        N [label="Normal", fillcolor="#C6EFCE"];
        O [label="Overweight", fillcolor="#FFE699"];
        OB [label="Obese", fillcolor="#F4CCCC"];

        A -> U [label="Ya"];
        A -> B [label="Tidak"];

        B -> N [label="Ya"];
        B -> C [label="Tidak"];

        C -> O [label="Ya"];
        C -> OB [label="Tidak"];
    }}
    """
    return dot

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🤖 Klasifikasi BMI dengan Decision Tree")
st.write("Aplikasi ini menghitung BMI dan mengklasifikasikan kategori berat badan menggunakan machine learning.")

# ============================================================
# UPLOAD FILE
# ============================================================
file = st.file_uploader("Upload file CSV (Height, Weight, optional Label)", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.subheader("📄 Data Awal")
    st.dataframe(df)

    # Validasi kolom
    if not {"Height", "Weight"}.issubset(df.columns):
        st.error("Dataset harus memiliki kolom 'Height' dan 'Weight'")
        st.stop()

    # Hitung BMI
    df["BMI"] = df.apply(lambda row: hitung_bmi(row["Weight"], row["Height"]), axis=1)

    # Buat label jika belum ada
    if "Kategori" not in df.columns:
        st.warning("⚠ Kolom 'Kategori' tidak ditemukan → membuat label otomatis (WHO)")
        df["Kategori"] = df["BMI"].apply(label_bmi)

    st.subheader("📊 Data dengan BMI & Label")
    st.dataframe(df)

    # Encode label
    encoder = LabelEncoder()
    df["Kategori_encoded"] = encoder.fit_transform(df["Kategori"])

    # Split data
    X = df[["BMI"]]
    y = df["Kategori_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)


    # Prediksi seluruh data
    df["Prediksi"] = encoder.inverse_transform(model.predict(X))

    st.subheader("📋 Hasil Prediksi")
    st.dataframe(df[["Height", "Weight", "BMI", "Kategori", "Prediksi"]])

    # Distribusi BMI
    st.subheader("📈 Distribusi BMI")
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(df["BMI"], bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Histogram BMI")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    # Visualisasi pohon keputusan
    st.subheader("🌳 Struktur Pohon Keputusan")
    thresholds = extract_tree_thresholds(model)
    dot = visualize_tree_pretty(thresholds)
    st.graphviz_chart(dot)

# ============================================================
# INPUT MANUAL
# ============================================================
st.subheader("🧍 Prediksi BMI (Input Manual)")

berat = st.number_input("Berat (kg)", min_value=1.0)
tinggi = st.number_input("Tinggi (cm)", min_value=1.0)

if st.button("Prediksi BMI"):
    bmi = hitung_bmi(berat, tinggi)
    st.success(f"BMI Anda: {bmi:.2f}")

    if file is not None:
        pred = model.predict([[bmi]])
        kategori = encoder.inverse_transform(pred)
        st.info(f"Kategori (Machine Learning): **{kategori[0]}**")
    else:
        st.warning("Model belum dilatih karena belum ada data.")

