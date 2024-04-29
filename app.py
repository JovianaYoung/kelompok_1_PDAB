import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import altair as alt
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression

st.set_page_config(
    page_title="Datmin - CO2 EMISSIONS",
    page_icon="car",
)

# Load the CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function for EDA (Exploratory Data Analysis)
def perform_eda(df):
    st.title('EDA (Exploratory Data Analysis)')
    st.write(""" Pada halaman ini menampilkan visualisasi :- heatmap (untuk melihat korelasi antar fitur) 
    - Relationship (untuk melihat hubungan antar dua fitur)
    - Distribution (untuk melihat distribusi jumlah data dari masing-masing fitur)
    - Composition (untuk melihat komposisi fitur dalam bentuk piechart)
    - Comparison (untuk melihat perbandingan antara fitur-fitur)
    """)
    st.sidebar.subheader('Visualisasi Data')

    # Filter correlation only for CO2 emissions
    corr_with_co2 = df.corr()['CO2EMISSIONS'].sort_values(ascending=False)

    # Remove CO2 emissions from correlation
    corr_with_co2 = corr_with_co2.drop('CO2EMISSIONS')

    # Sidebar menu for visualization options
    visualization_option = st.sidebar.selectbox('Pilih Visualisasi', ['Heatmap Correlation', 'Relationship', 'Distribution', 'Composition', 'Comparison'])

    if visualization_option == 'Heatmap Correlation':
        st.subheader('Visualisasi Heatmap CO2 Emissions dengan Fitur Lainnya')

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_with_co2.to_frame(), annot=True, cmap='Reds', fmt=".2f", linewidths=0.5)
        st.pyplot()
        st.markdown('Berdasarkan visualisasi di atas dapat disimpulkan bahwa fitur yang paling mempengaruhi nilai gas CO2 Emissions adalah fuelconsumtion_comb dan fuelconsumption_hwy')
        st.set_option('deprecation.showPyplotGlobalUse', False)

    elif visualization_option == 'Relationship':
        st.subheader('Scatter Plot dengan Regresi Linier')
        # Scatter plot with linear regression
        fig, ax = plt.subplots()
        sns.regplot(x='ENGINESIZE', y='CO2EMISSIONS', data=df, color='blue', line_kws={"color": "red"}, ax=ax)
        ax.set_xlabel('ENGINE SIZE')
        ax.set_ylabel('CO2 Emission')
        ax.set_title('Hubungan Antara ENGINESIZE dan CO2 Emission\n(Ditampilkan dengan Garis Regresi)')
        # Show plot in Streamlit
        st.pyplot(fig)
        st.markdown('Dari hasil visualisasi diatas kami membandingkan fitur engine size dengan CO2 emissions dan bisa dilihat nilai dari engine size cukup mendekati garis regresi.')
        
        #Scatter plot with linear regression
        fig, ax = plt.subplots()
        sns.regplot(x='CYLINDERS', y='CO2EMISSIONS', data=df, color='orange', line_kws={"color": "red"}, ax=ax)
        ax.set_xlabel('CYLINDERS')
        ax.set_ylabel('CO2 Emission')
        ax.set_title('Hubungan Antara CYLINDERS dan CO2 Emission\n(Ditampilkan dengan Garis Regresi)')
        # Show plot in Streamlit
        st.pyplot(fig)
        st.markdown('Dari hasil visualisasi diatas kami membandingkan fitur cylinder dengan CO2 emissions dan bisa dilihat nilai cylinder menjauh dari garis regresi.')

        # Scatter plot with linear regression
        fig, ax = plt.subplots()
        sns.regplot(x='FUELCONSUMPTION_CITY', y='CO2EMISSIONS', data=df, color='green', line_kws={"color": "red"}, ax=ax)
        ax.set_xlabel('FUELCONSUMPTION_CITY')
        ax.set_ylabel('CO2 Emission')
        ax.set_title('Hubungan Antara FUELCONSUMPTION_CITY dan CO2 Emission\n(Ditampilkan dengan Garis Regresi)')
        # Show plot in Streamlit
        st.pyplot(fig)
        st.markdown('Dari hasil visualisasi diatas kami membandingkan fitur fuelconsumption_city dengan CO2 emissions dan bisa dilihat nilai fuelconsumption_city mendekati dari garis regresi.')

        fig, ax = plt.subplots()
        sns.regplot(x='FUELCONSUMPTION_HWY', y='CO2EMISSIONS', data=df, color='yellow', line_kws={"color": "red"}, ax=ax)
        ax.set_xlabel('FUELCONSUMPTION_HWY')
        ax.set_ylabel('CO2 Emission')
        ax.set_title('Hubungan Antara FUELCONSUMPTION_HWY dan CO2 Emission\n(Ditampilkan dengan Garis Regresi)')
        # Show plot in Streamlit
        st.pyplot(fig)
        st.markdown('Dari hasil visualisasi diatas kami membandingkan fitur fuelconsumption_hwy dengan CO2 emissions dan bisa dilihat nilai fuelconsumption_hwy mendekati dari garis regresi, tetapi tidak sedekat dengan fitur fuelconsumption_city')\
        

        fig, ax = plt.subplots()
        sns.regplot(x='FUELCONSUMPTION_COMB', y='CO2EMISSIONS', data=df, color='purple', line_kws={"color": "red"}, ax=ax)
        ax.set_xlabel('FUELCONSUMPTION_COMB')
        ax.set_ylabel('CO2 Emission')
        ax.set_title('Hubungan Antara FUELCONSUMPTION_COMB dan CO2 Emission\n(Ditampilkan dengan Garis Regresi)')
        st.pyplot(fig)
        st.markdown('Dari hasil visualisasi diatas kami membandingkan fitur fuelconsumption_comb dengan CO2 emissions dan bisa dilihat nilai fuelconsumption_comb mendekati dari garis regresi, tetapi terdapat beberapa nilai yang menjauh dari garis.')
        st.markdown('Kesimpulan dari semua hasil visualisasi diatas adalah semua fitur mempengaruhi nilai dari gas CO2 emissions')
        # Show plot in Streamlit
        # Additional scatter plots
        # ...

    elif visualization_option == 'Distribution':
        st.subheader('Distribusi Variabel')
        # Histogram for single variable
        selected_variable = st.selectbox('Pilih Variabel', df.columns)
        if selected_variable:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[selected_variable], kde=True)
            st.pyplot()
            
            # Markdown text based on selected variable
            if selected_variable == 'ENGINESIZE':
                st.markdown("Distribusi dari **Engine Size**. Ini adalah distribusi dari ukuran mesin kendaraan. Dan dapat dilihat nilai ukuran mesin 4 paling tinggi")
            elif selected_variable == 'CYLINDERS':
                st.markdown("Distribusi dari **Cylinders**. Ini adalah distribusi dari jumlah silinder kendaraan. Dan dapat dilihat jumlah silinder 4 dan 6 yang paling tinggi distribusi datanya ")
            elif selected_variable == 'FUELTYPE':
                st.markdown("Distribusi dari **Fuel Type**. Ini adalah distribusi dari jenis bahan bakar kendaraan. Dan bisa dilihat bahwa jenis bahan bakar bensin dan diesel yang paling banyak digunakan oleh kendaraan")
                
            elif selected_variable == 'FUELCONSUMPTION_CITY':
                st.markdown("Distribusi dari **Fuel Consumption in City**. Ini adalah distribusi dari konsumsi bahan bakar di kota.")
            elif selected_variable == 'FUELCONSUMPTION_HWY':
                st.markdown("Distribusi dari **Fuel Consumption on Highway**. Ini adalah distribusi dari konsumsi bahan bakar di jalan raya.")
            elif selected_variable == 'FUELCONSUMPTION_COMB':
                st.markdown("Distribusi dari **Fuel Consumption Combined**. Ini adalah distribusi dari konsumsi bahan bakar yang dikombinasikan.")
        else:
            st.write("Pilih sebuah variabel untuk melihat distribusi.")

        # Pairplot for multiple variables
        # ...

    elif visualization_option == 'Composition':
        st.subheader('Komposisi Variabel')
        # Pie chart for fuel type composition
        fueltype_composition = df['FUELTYPE'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(fueltype_composition, labels=fueltype_composition.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Komposisi Jenis Bahan Bakar')
        st.pyplot()
        st.markdown('Dapat dilihat dari visualisasi pie chart diatas bahwa jenis bahan bakar yang paling tinggi digunakan adalah bahan bakar diesel.')
    elif visualization_option == 'Comparison':
        st.subheader('Perbandingan Variabel')
        # Bar chart for comparison of fuel consumption
        fuel_consumption_columns = ['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']
        fuel_consumption_comparison = df[fuel_consumption_columns].mean()

        plt.figure(figsize=(10, 6))
        fuel_consumption_comparison.plot(kind='bar', color=['green', 'blue', 'purple'])
        plt.title('Perbandingan Konsumsi Bahan Bakar')
        plt.xlabel('Tipe Konsumsi Bahan Bakar')
        plt.ylabel('Rata-rata Konsumsi Bahan Bakar')
        plt.xticks(rotation=45)
        st.pyplot()

        st.markdown('Dari visualisasi di atas, kita dapat melihat rata-rata konsumsi bahan bakar untuk berbagai jenis penggunaan jalan. Dan dapat disimpulkan bahwa rata-rata konsumsi bahan bakar di kota yang paling tinggi')

# Function for prediction
def predict():
    df = pd.read_csv('data_final.csv')
    st.write("""
    # Prediksi Gas CO2 Emissions
    Gunakan model di bawah untuk memasukkan fitur kendaraan dan memprediksi emisi CO2 gas.
    """)
    st.subheader("Dataset CO2 Emissions yang telah dicleaning")
    st.write(df)  # Menampilkan keseluruhan dataset dari file data_final.csv

    x = df.drop('CO2EMISSIONS', axis=1)
    y = df['CO2EMISSIONS']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    loaded_model = joblib.load(open('model_linear (4).pkl', 'rb'))
    loaded_model.fit(x_train,y_train)

    # Input features
    ENGINESIZE = st.number_input('Input nilai ENGINESIZE')
    CYLINDERS = st.number_input('Input nilai CYLINDERS')
    fueltype_options = ['Z', 'E', 'X', 'D']
    FUELTYPE = st.selectbox('Select FUELTYPE', fueltype_options)
    FUELCONSUMPTION_CITY = st.number_input('Input nilai FUELCONSUMPTION_CITY')
    FUELCONSUMPTION_HWY = st.number_input('Input nilai FUELCONSUMPTION_HWY')
    FUELCONSUMPTION_COMB = st.number_input('Input nilai FUELCONSUMPTION_COMB')

    # Define input_data
    input_data = pd.DataFrame({
        'ENGINESIZE': [ENGINESIZE],
        'CYLINDERS': [CYLINDERS],
        'FUELTYPE': [FUELTYPE],
        'FUELCONSUMPTION_CITY': [FUELCONSUMPTION_CITY],
        'FUELCONSUMPTION_HWY': [FUELCONSUMPTION_HWY],
        'FUELCONSUMPTION_COMB': [FUELCONSUMPTION_COMB]
    })

    # Prediction
    if st.button('Prediksi Gas CO2 Emissions'):
        predicted_values = loaded_model.predict(input_data)
        st.write('Prediksi Gas CO2 Emissions', predicted_values)

# Main function
def main():
    # Load CSS
    local_css("style.css")

    # Display main menu
    st.sidebar.title('Menu')
    menu_options = ['Home', 'EDA', 'Predict']
    choice = st.sidebar.selectbox('Pilih Menu', menu_options)

    if choice == 'Home':
        st.write("""
        # Welcome to Dashboard Prediksi Gas CO2 Emissions :car:
        """)
        st.image('emisimobil.jpg', use_column_width=True)
        st.write("""
        Dashboard ini memungkinkan Anda untuk melihat hasil analisis dari data Gas CO2 Emissions kendaraan Mobil.
        Dengan dibuatnya dashboard ini, Anda dapat:
        - Melihat visualisasi yang insighful dari data Gas CO2 Emissions kendaraan Mobil.
        - Melihat hasil prediksi Gas CO2 Emissions kendaraan mobil dengan model regresi yang telah dibuat.

        ## Daftar Menu
        1. :house: Home
        2. :bar_chart: Visualisasi EDA
        3. :chart_with_upwards_trend: Predict
        
        ## Penjelasan Singkat Gas CO2
        Gas CO2 (karbon dioksida) adalah gas yang umumnya dihasilkan oleh aktivitas manusia seperti pembakaran bahan bakar fosil, industri, dan pertanian. Gas ini merupakan penyumbang utama terhadap pemanasan global dan perubahan iklim di seluruh dunia. Dalam konteks emisi gas CO2 dari kendaraan bermotor, hal ini menjadi perhatian serius karena mobil  merupakan sumber emisi CO2 yang signifikan. Dalam upaya mengurangi dampak negatifnya terhadap lingkungan, perlu dilakukan pemahaman dan analisis yang mendalam terhadap faktor-faktor yang mempengaruhi emisi CO2 dari kendaraan, seperti ukuran mesin, jumlah silinder, jenis bahan bakar, dan efisiensi bahan bakar.
        """, unsafe_allow_html=True)

    elif choice == 'EDA':
        # Load the dataset
        df = pd.read_csv('Data Cleaned (4).csv')
        perform_eda(df)
    elif choice == 'Predict':
        predict()

if __name__ == "__main__":
    main()
