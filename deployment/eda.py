import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title ='Churn Prediction - EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Membuat Title
    st.title('Banking Customer Churn Prediction')

    # Menambahkan gambar
    st.image('https://storage.googleapis.com/kaggle-datasets-images/4649694/7913901/e3757b426bb4e2dcb1928d74db505249/dataset-cover.jpg?t=2024-03-22-11-53-11',
             caption='Bank X Churn Prediction')

    # Menambahkan deskripsi
    st.write('Made by Alridho')
    st.write('# Latar Belakang')
    st.write('Sebagai seorang Data Scientist yang baru bergabung di Bank X, tanggung jawab utama saya adalah menganalisis perilaku nasabah untuk mengidentifikasi risiko churn. Bank X unggul dalam memberikan layanan perbankan yang inovatif, pengalaman nasabah yang terpersonalisasi, serta penawaran produk yang relevan, menjadikannya pilihan utama di industri perbankan. Dalam kondisi persaingan yang semakin ketat, kemampuan untuk mengenali nasabah yang berpotensi melakukan churn sangatlah krusial untuk meningkatkan tingkat retensi dan mempertahankan loyalitas mereka. Melalui data transaksi dan interaksi nasabah, serta penerapan algoritma Machine Learning, saya mengembangkan model untuk mengklasifikasikan potensi churn nasabah, yang bertujuan meningkatkan strategi retensi dan menjaga nasabah dengan nilai tinggi tetap bertahan.')
    st.write('# Objective')
    st.write('Proyek ini bertujuan mengembangkan model Machine Learning untuk mengklasifikasikan apakah nasabah berpotensi untuk churn atau tetap menjadi pelanggan aktif di masa depan. Model ini akan dibangun berdasarkan berbagai fitur seperti skor kredit, usia, lama berlangganan (tenure), saldo akun, jumlah produk yang dimiliki, dan status keaktifan nasabah. Fokus utama dari model ini adalah mengurangi jumlah False Negative, yaitu ketika nasabah yang berpotensi churn justru tidak terdeteksi oleh model. Mengurangi False Negative sangat penting agar bank dapat mengambil tindakan preventif terhadap nasabah yang mungkin akan meninggalkan layanan. Hasil klasifikasi ini akan membantu Bank X dalam melakukan intervensi yang lebih tepat waktu, meningkatkan strategi retensi, dan menjaga hubungan dengan nasabah yang bernilai tinggi agar tetap loyal kepada bank.')

    # Membuat garis lurus
    st.markdown('---')

    # Magic syntax
    '''
    Untuk page ini, penulis akan melakukan eksplorasi sederhana (EDA),
    Dataset yang digunakan adalah Banking Customer Churn Prediction dari kaggle.com
    '''

    # Show DataFrame
    df = pd.read_csv('Churn_Modelling.csv')
    st.dataframe(df)

    # Membuat Barplot
    st.write('#### Plot Status Nasabah')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x='Exited',data=df,palette=['blue', 'red'])
    st.pyplot(fig)

    # Membuat Piechart
    st.write('#### Piechart Selection')
    pilihan = st.selectbox('Pilih kolom:',('Geography','Gender'))
    fig = plt.figure(figsize=(5,5))
    df[pilihan].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie chart for {pilihan}')
    st.pyplot(fig)

    # Membuat Histogram berdasarkan input user
    st.write('#### Histogram Distribusi Data Berdasarkan Input User')
    pilihan = st.selectbox('Pilih kolom:',('CreditScore','Age','Balance'))
    fig = plt.figure(figsize=(5,5))
    sns.histplot(df[pilihan], bins=30, kde=True)
    st.pyplot(fig)

if __name__== '__main__':
    run()
