
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
#         APLIKASI BMI + DECISION TREE + VISUALISASI
# ============================================================

# ===========================
# Node Class
# ===========================
class Node:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, result=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.result = result

    def is_leaf(self):
        return self.result is not None


# ===========================
# BUILD TREE BMI
# ===========================
def build_tree():
    root = Node("BMI", 18.5)
    root.left = Node(result="Underweight")

    root.right = Node("BMI", 25)
    root.right.left = Node(result="Normal")

    root.right.right = Node("BMI", 30)
    root.right.right.left = Node(result="Overweight")
    root.right.right.right = Node(result="Obese")

    return root


# ===========================
# TRAVERSE TREE
# ===========================
def traverse_tree(node, data):
    if node.is_leaf():
        return node.result

    value = data.get(node.attribute)
    if value is None:
        return "BMI tidak ditemukan dalam data!"

    if value < node.threshold:
        return traverse_tree(node.left, data)
    else:
        return traverse_tree(node.right, data)


# ===========================
# PRINT TREE (string output)
# ===========================
def print_tree(node, indent=""):
    if node.is_leaf():
        return indent + f"└── {node.result}\n"

    s = indent + f"[{node.attribute} < {node.threshold}]\n"
    s += indent + "├── Left:\n"
    s += print_tree(node.left, indent + "│   ")
    s += indent + "└── Right:\n"
    s += print_tree(node.right, indent + "    ")
    return s


# ===========================
# BMI RUMUS
# ===========================
def hitung_bmi(berat, tinggi_cm):
    tinggi_m = tinggi_cm / 100
    return berat / (tinggi_m**2)


# ============================================================
#                     STREAMLIT UI
# ============================================================

st.title("📊 Aplikasi Klasifikasi BMI + Decision Tree")
st.write("Unggah file Anda sendiri lalu lakukan analisis BMI dan visualisasi.")

# ===========================
# FILE UPLOADER
# ===========================
file = st.file_uploader("bmi.csv")

if file is not None:
    df = pd.read_csv(file)
    st.subheader("📄 Data Anda")
    st.dataframe(df)

    # ============================================================
    # Hitung BMI jika kolom berat & tinggi tersedia
    # ============================================================
    required_cols = ["Height", "Weight"]

    if all(col in df.columns for col in required_cols):

        df["BMI"] = df.apply(lambda row: hitung_bmi(row["Height"], row["Weight"]), axis=1)

        st.subheader("📈 Distribusi BMI")
        fig, ax = plt.subplots(figsize=(7,5))
        ax.hist(df["BMI"], bins=20)
        ax.set_title("Histogram BMI")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

        # ============================================================
        # Decision tree klasifikasi untuk seluruh data
        # ============================================================
        tree = build_tree()

        df["Kategori_BMI"] = df["BMI"].apply(lambda x: traverse_tree(tree, {"BMI": x}))

        st.subheader("📋 Hasil Kategori BMI")
        st.dataframe(df[["Height", "Weight", "BMI", "Kategori_BMI"]])

    else:
        st.warning("Kolom wajib: 'berat' dan 'tinggi' belum ada dalam file.")

# ============================================================
# Input manual BMI
# ============================================================
st.subheader("🧍 Perhitungan BMI Pasien (Input Manual)")

berat = st.number_input("Berat (kg)", min_value=1.0)
tinggi = st.number_input("Tinggi (cm)", min_value=1.0)

if st.button("Hitung BMI"):
    bmi = hitung_bmi(berat, tinggi)
    tree = build_tree()
    kategori = traverse_tree(tree, {"BMI": bmi})

    st.success(f"BMI Anda: {bmi:.2f}")
    st.info(f"Kategori: **{kategori}**")

