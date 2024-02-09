import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocessing
import seaborn as sns

st.title('Housing Pricing Prediction')
st.image('https://images.unsplash.com/photo-1524813686514-a57563d77965?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8aG91c2luZ3xlbnwwfDB8MHx8fDA%3D', width=600)
tab1, tab2, tab3, tab4= st.tabs(["Overview", "Business & Data Understanding", "Data Cleaning & Analysis", "Modelling, Evaluation, Deployment"])
with tab1:
    st.write("""
    Prediksi harga rumah yang akurat dapat membantu pemilik rumah, pembeli potensial, dan agen real estat dalam pengambilan keputusan yang lebih baik terkait harga properti.

    Kawasan Ames, Iowa, dipilih karena memiliki pasar perumahan yang cukup stabil dengan beragam tipe rumah dan fitur-fitur yang berbeda. Data yang digunakan dalam penELitian ini berasal dari Ames Housing Dataset, yang mencakup berbagai variabel seperti luas tanah, luas bangunan, jumlah kamar, dan fitur-fitur lain yang dapat mempengaruhi harga rumah.

    Metode yang digunakan dalam pengembangan model machine learning meliputi Data Understanding, Data Cleaning & Analysis, Data Modelling, Model Evaluation dan Model Deployment. Algoritma yang digunakan adalah XGBoost Regressor.

    Hasil dari penelitian ini diharapkan dapat memberikan pemahaman yang lebih baik tentang faktor-faktor apa saja yang mempengaruhi harga rumah di kawasan Ames, Iowa, serta model machine learning yang dapat digunakan untuk memprediksi harga rumah dengan tingkat akurasi yang tinggi.
    """)
with tab2:
    st.header('Business Understanding')
    st.subheader('Problem Statements dan Goals')
    st.write("""
    - **Problem**
        1. Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga rumah?
        2. Berapa harga pasar rumah dengan karakteristik atau fitur tertentu?
    - **Goals**
        1. Mengetahui fitur yang paling berkorelasi dengan harga rumah
        2. Membuat model machine learning yang dapat memprediksi harga rumah dengan tingkat akurasi yang tinggi
    - **Metodology**

        Prediksi harga adalah tujuan utama yang ingin dicapai dalam penelitian ini. Harga merupakan variabel kontinu yang ingin diprediksi. Dalam analisis prediktif, ketika Anda memprediksi variabel kontinu, Anda sedang menghadapi permasalahan regresi. Oleh karena itu, metodologi yang digunakan dalam penelitian ini adalah membangun model regresi dengan harga rumah sebagai target
    - **Metrics**

        Metrik digunakan untuk mengevaluasi kinerja model dalam memprediksi harga. Dalam kasus regresi, metrik yang umum digunakan adalah Root Mean Square Error (RMSE). Metrik ini mengukur seberapa besar deviasi antara hasil prediksi dan nilai aktual
    """)
    st.header('Data Understanding')
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    st.subheader('Training Data')
    col1, col2 = st.columns([9,3])
    with col1:
        st.dataframe(data=train_df)
    with col2:
        column_types = pd.DataFrame(train_df.dtypes, columns=['Type'])
        st.dataframe(data=column_types)
    st.write(train_df.describe(include="all"))
    st.subheader('Test Data')
    col1, col2 = st.columns([9,3])
    with col1:
        st.dataframe(data=test_df)
    with col2:
        column_types = pd.DataFrame(test_df.dtypes, columns=['Type'])
        st.dataframe(data=column_types)
    st.write(test_df.describe(include="all"))
    numerical_feats = train_df.dtypes[train_df.dtypes != "object"].index
    categorical_feats = train_df.dtypes[train_df.dtypes == "object"].index
    st.code("""
    numerical_feats = train.dtypes[train.dtypes != "object"].index
    print("Number of Numerical features: ", len(numerical_feats))
    categorical_feats = train.dtypes[train.dtypes == "object"].index
    print("Number of Categorical features: ", len(categorical_feats))
    """, language='python')
    st.caption("Terdapat 37 fitur numerik dan 43 fitur kategorikal")
with tab3:
    st.header('Data Cleaning & Analysis')
    st.subheader('Remove ID')
    train_df.drop('Id', axis=1, inplace=True)
    test_df.drop('Id', axis=1,  inplace=True)
    st.code("""
    train_df.drop('Id', axis=1, inplace=True)
    test_df.drop('Id', axis=1,  inplace=True)
    """)
    st.subheader('Missing Values')
    st.image("image/train_missing_value.png")
    st.code("""
    train_missing_data = preprocessing.calculate_missing_data(train_df)
    plt.figure(figsize=(13, 5))
    ax = sns.barplot(x=train_missing_data['Feature'], y=train_missing_data['Percent'])
    ax.set_title("Train Data Missing Value")
    for index, value in enumerate(train_missing_data['Percent']):
        plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    """)
    st.image("image/test_missing_value.png")
    st.code("""
    test_missing_data = preprocessing.calculate_missing_data(test_df)
    plt.figure(figsize=(13, 5))
    ax = sns.barplot(x=test_missing_data['Feature'], y=test_missing_data['Percent'])
    ax.set_title("Test Data Missing Value")
    for index, value in enumerate(test_missing_data['Percent']):
        plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    """)
    st.caption("Terdapat 33 fitur yang memiliki missing value. Sama seperti train, terdapat 4 fitur  diantaranya yang memiliki missing value di atas 80% yaitu PoolQC, MiscFeature, Alley, dan Fence. Pada data test, FirplaceQU memiliki missing value yang lebih dari 50%. Maka dari itu, PoolQC, MiscFeature, Alley, Fence, dan FirplaceQU akan di drop karena memiliki missing value yang terlalu banyak.")
    code = """
    features_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    train_df = train_df.drop(columns=features_to_drop)
    test_df = test_df.drop(columns=features_to_drop)
    train_missing_data = train_missing_data[~train_missing_data['Feature'].isin(features_to_drop)]
    test_missing_data = test_missing_data[~test_missing_data['Feature'].isin(features_to_drop)]
    """
    st.subheader('Dropping features with many missing values')
    st.code(code, language='python')
    st.subheader('Fill Missing Values')
    st.write("**Categorical Feature**")
    st.write("**Train Data**")
    st.code("""
    feature_to_fill = train_missing_data.Feature.tolist()
    for feature in feature_to_fill:
        if feature in categorical_feats:
            mode_value = train_df[feature].mode()[0]
            train_df[feature].fillna(mode_value, inplace=True)
""", language='python')
    st.write("**Test Data**")  
    st.code("""
    feature_to_fill = test_missing_data.Feature.tolist()
    for feature in feature_to_fill:
        if feature in categorical_feats:
            mode_value = test_df[feature].mode()[0]
            test_df[feature].fillna(mode_value, inplace=True)
            """)
    st.write("**Numerical Feature**")
    st.write("**Train Data**")
    st.code("""
    train_missing_data = preprocessing.calculate_missing_data(train_df)
    feature_to_fill = train_missing_data.Feature.tolist()
    for feature in feature_to_fill:
        if feature in numerical_feats:
            skewness = train_df[feature].skew()
            if -1 < skewness < 1:
                fill_value = train_df[feature].mean()
            else:
                fill_value = train_df[feature].median()
            train_df[feature].fillna(fill_value, inplace=True)
""", language='python')
    st.write("**Test Data**")  
    st.code("""
     test_missing_data = preprocessing.calculate_missing_data(test_df)
    feature_to_fill = test_missing_data.Feature.tolist()
    for feature in feature_to_fill:
        if feature in numerical_feats:
            skewness = train_df[feature].skew()

            if -1 < skewness < 1:
                fill_value = train_df[feature].mean()
            else:
                fill_value = train_df[feature].median()

            # Isi data test menggunakan data train
            test_df[feature].fillna(fill_value, inplace=True)
            """)
    st.subheader('Log Transformation')
    st.write("Log transformation digunakan untuk mengurangi skewness pada fitur numerik. Skewness adalah ukuran seberapa simetris distribusi. Jika skewness lebih dari 1 atau kurang dari -1, maka distribusi tersebut dianggap sangat condong.")
    st.code("""
    sns.histplot(train['SalePrice'], kde=True)
    """, language='python')
    st.image("image/log_transform_on_target.png")
    st.caption("Dapat dilihat bahwa target SalePrice memiliki skew positif ke kanan, hal ini akan berdampak pada keakuratan regresi linear yang akan digunakan. Agar regresi linear dapat dilakukan dengan akuran, log transform akan dilakukan pada target.")
    st.code("""
    def log_transform(y):
    return np.log1p(y)

# Inverse log transotm akan digunakan pada hasil prediksi data test, untuk mengembalikan nilai ke nilai sebelum log transform
def inverse_log_transform(y_log):
    return np.expm1(y_log)
    """)
    st.code("""
    train['SalePrice'] = log_transform(train['SalePrice'])
    plt.figure(figsize=(10, 4))

sns.histplot(train['SalePrice'], kde=True)
""")
    st.image("image/log_transform_on_target_normal.png")
    st.caption("Setelah dilakukan log transform, distribusi target SalePrice menjadi lebih normal.")
    st.subheader('Handling Outliers')
    st.code("""
    def calculate_outliers(column):
        # Calculate the IQR
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1

        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter the column to count outliers
        outliers = column[(column < lower_bound) | (column > upper_bound)]

        # Calculate percentage of outliers
        percentage_outliers = (outliers.shape[0] / column.shape[0]) * 100

        return outliers.shape[0], percentage_outliers
    """)
    st.code("""
        outliers_info = []

for feature in numerical_feats:
  num_outliers, percentage_outliers = calculate_outliers(train[feature])

  if percentage_outliers > 0:
    outliers_info.append({
        'feature': feature,
        'outlier_count': num_outliers,
        'percentage_outliers': percentage_outliers
    })
    """)
    st.subheader('Correlation')
    st.write("**Numerical Feature**")
    st.code("""
    correlation_results = []
p_value_results = []

for column in numerical_feats:
    # Calculate Pearson correlation coefficient (r) and p-value (p)
    r, p = stats.pearsonr(train[column], train['SalePrice'])

    correlation_results.append(r)
    p_value_results.append(p)

corr_train = pd.DataFrame({
    'Feature': numerical_feats,
    'pearson_correlation': correlation_results,
    'p_value': p_value_results
})

corr_train = corr_train.reindex(corr_train['pearson_correlation'].abs().sort_values(ascending=False).index)
corr_train = corr_train.sort_values(by='p_value', ascending=True)

corr_train['correlation_description'] = corr_train['pearson_correlation'].apply(lambda r:
    'no correlation' if np.isnan(r) else
    'no correlation' if r == 0 else
    'strong positive' if r >= 0.8 else
    'moderate positive' if r >= 0.5 else
    'weak positive' if r >= 0.3 else
    'weak negative' if r >= -0.3 else
    'moderate negative' if r >= -0.5 else
    'strong negative'
)

corr_train['statistical_significance'] = corr_train['p_value'].apply(lambda p:
  'statistically_significant' if p < 0.05 else
  'not_statistically_significant'
)
""")
    st.image("image/correlatiion_numerical.png")
    st.write("""
    Terdapat 4 fitur yang memiliki korelasi yang rendah dan secara statistik tidak significant. Fitur-Fitur tersebut adalah

    1. YrSold
    2. OverallCond
    3. BsmtFinSF2
    4. BsmtHalfBath

    Fitur ini akan di drop karena tidak memiliki pengaruh terhadap target""")
    st.write("**Categorical Feature**")
    st.code("""
    categorical_feats = train.dtypes[train.dtypes == "object"].index

    li_cat_feats = list(categorical_feats)
    nr_rows = 13
    nr_cols = 3

    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*5,nr_rows*4))

    for r in range(0,nr_rows):
        for c in range(0,nr_cols):
            i = r*nr_cols+c
            if i < len(li_cat_feats):
                sns.boxplot(x=li_cat_feats[i], y='SalePrice', data=train, ax = axs[r][c])
                axs[r][c].tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    """)
    st.image("image/correlatiion_categorical.png")
    st.write("""
    Dari analisa BoxPlot, terdapat beberapa fitur yang memiliki korelasi yang signifikan dengan SalePrice. Hal ini ditandai dengan terpisahnya bagian IQR boxplot pada tiap unique value di fitur kategori tertentu. Fitur-fitur tersebut adalah:

    1. MSZoning
    2. Neighborhood
    3. Condition2
    4. MasVnrType
    5. ExterQual
    6. BsmtQual
    7. CentralAir
    8. Electrical
    9. KitchenQual
    10. SaleType

    Selain fitur di atas, fitur kategorikal lain akan di drop
    """)
    st.subheader('Convert Categorical Feature to Numeric Feature')
    st.code("""
    ordinal_feats = ['ExterQual', 'BsmtQual', 'KitchenQual']

quality_mapping = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

for feat in ordinal_feats:
    train[feat] = train[feat].map(quality_mapping)
    test[feat] = test[feat].map(quality_mapping)
    """)
    st.caption("Fitu Neighborhood	dan MasVnrType merupakan data kategorikal nominal, maka binary encoding akan diterapkan untuk mengubah data kategorikal menjadi numerical")
    st.code("""
    nominal_feats = ['Neighborhood', 'MasVnrType']

for feature in nominal_feats:
    encoder = ce.BinaryEncoder(cols=[feature])

    # Fit transform pada train, test menggunakan fit
    train_encoded = encoder.fit_transform(train[feature])

    train_df.drop(columns=[feature], inplace=True)
    train_df = pd.concat([train_df, train_encoded], axis=1)

    # Menggunakan encoder hasil train
    test_encoded = encoder.transform(test_df[feature])

    test_df.drop(columns=[feature], inplace=True)
    test_df = pd.concat([test_df, test_encoded], axis=1)
    """)
    st.subheader('Scaling')
    st.code("""
    scaler = StandardScaler()

train_numerical = train.iloc[:, :-1].copy()  # Exclude the last column
test_numerical = test.copy()
train_scaled = scaler.fit_transform(train_numerical)

# Pada test gunakan transform supaya menggunakan scaler yang sama dengan train
test_scaled = scaler.transform(test_numerical)
train_scaled_df = pd.DataFrame(train_scaled, columns=train_numerical.columns)
train_scaled_df['SalePrice'] = train['SalePrice']

test_scaled_df = pd.DataFrame(test_scaled, columns=test_numerical.columns)
""")
with tab4:
    st.header('Prediction')
    col1, col2 = st.columns(2)
    with col1:
        # Dictionary mapping options to values
        genre_mapping = {
            '1-STORY 1946 & NEWER ALL STYLES': 20,
            '1-STORY 1945 & OLDER': 30,
            '1-STORY W/FINISHED ATTIC ALL AGES': 40,
            '1-1/2 STORY - UNFINISHED ALL AGES': 45,
            '1-1/2 STORY FINISHED ALL AGES': 50,
            '2-STORY 1946 & NEWER': 60,
            '2-STORY 1945 & OLDER': 70,
            '2-1/2 STORY ALL AGES': 75,
            'SPLIT OR MULTI-LEVEL': 80,
            'SPLIT FOYER': 85,
            'DUPLEX - ALL STYLES AND AGES': 90,
            '1-STORY PUD (Planned Unit Development) - 1946 & NEWER': 120,
            '1-1/2 STORY PUD - ALL AGES': 150,
            '2-STORY PUD - 1946 & NEWER': 160,
            'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER': 180,
            '2 FAMILY CONVERSION - ALL STYLES AND AGES': 190,
        }

        # Select box for movie genre
        selected_genre = st.selectbox(
            label="What's your Buliding Class?",
            options=list(genre_mapping.keys())  # Use keys from the dictionary as options
        )
        selected_value = genre_mapping[selected_genre]
    with col2:
        lotFrontage = st.slider(
    label='Linear feet of street connected to property',
    min_value=20, max_value=313, value=60, step=1)
    col1, col2 = st.columns(2)
    with col1:
        yearBuilt = st.slider(
            label='Original construction date',
            min_value=1872, max_value=2010, value=2000, step=1)
    with col2:
        yearRemodAdd = st.slider(
            label='Remodel date',
            min_value=1950, max_value=2010, value=2000, step=1)
    col1, col2 = st.columns(2)
    with col1:
        MasVnrArea = st.slider(
            label='Masonry veneer area in square feet',
            min_value=0, max_value=1600, value=100, step=1)
    with col2:
        TotalBsmtSF = st.slider(
            label='Total square feet of basement area',
            min_value=0, max_value=6110, value=1000, step=1)
    col1, col2 = st.columns(2)
    with col1:
        BsmtFinSF1 = st.slider(
            label='Type 1 finished square feet',
            min_value=0, max_value=5644, value=1000, step=1)
   
    with col2:
        GrLivArea = st.slider(
            label='Above grade (ground) living area square feet',
            min_value=334, max_value=5642, value=1000, step=1)
    col1, col2 = st.columns(2)
    overall_quality_mapping = {
        'Very Poor': 1,
        'Poor': 2,
        'Fair': 3,
        'Below Average': 4,
        'Average': 5,
        'Above Average': 6,
        'Good': 7,
        'Very Good': 8,
        'Excellent': 9,
        'Very Excellent': 10
    }
    with col1:
        lotArea =st.slider(
            label='Lot size in square feet',
            min_value=7.20, max_value=12.20, value=8.00, step=0.20)
    with col2:
        OverallQual = st.selectbox(
            label='Overall material and finish quality',
            options=list(overall_quality_mapping.keys()))
        OverallQual = overall_quality_mapping[OverallQual]
    col1, col2 = st.columns(2)
    quality_mapping = {'Fair': 1, 'Average': 2, 'Good': 3, 'Excelent': 4}
    with col1:
        ExterQual = st.selectbox(
            label='Evaluates the quality of the material on the exterior',
            options=list(quality_mapping.keys()))
        ExterQual = quality_mapping[ExterQual]
    with col2:
        BsmtQual = st.selectbox(
            label='Evaluates the height of the basement',
            options=list(quality_mapping.keys()))
        BsmtQual = quality_mapping[BsmtQual]
    col1, col2 = st.columns(2)
    with col1:
        BsmtUnfSF = st.slider(
            label='Unfinished square feet of basement area',
            min_value=0, max_value=2336, value=1000, step=1)
    with col2:
        BsmtFullBath = st.slider(
            label='Basement full bathrooms',
            min_value=0, max_value=3, value=1, step=1)
    col1, col2 = st.columns(2)
    with col1:
        firstFlrsf = st.slider(
            label='First Floor square feet',
            min_value=334, max_value=4692, value=1000, step=1)
    with col2:
        secondFlrsf = st.slider(
            label='Second floor square feet',
            min_value=0, max_value=2065, value=1000, step=1)
    KitchenQual = st.selectbox(
        label='Kitchen quality',
        options=list(quality_mapping.keys()))
    KitchenQual = quality_mapping[KitchenQual]
    lowQualFinSF = st.slider(
            label='Low quality finished square feet (all floors)',
            min_value=0.00, max_value=6.35, value=5.00, step=0.1)
    col1, col2 = st.columns(2)
    with col1:
        halfBath = st.slider(
            label='Half baths above grade',
            min_value=0, max_value=2, value=1, step=1)
    with col2:
        fullBath = st.slider(
            label='Full bathrooms above grade',
            min_value=0, max_value=3, value=1, step=1)
    bedroomAbvGr = st.slider(
            label='Bedrooms above grade (does NOT include basement bedrooms)',
            min_value=0, max_value=8, value=4, step=1)
    kitchenAbvgGr = st.slider(
            label='Kitchens above grade',
            min_value=0, max_value=3, value=1, step=1)
    totrmsAbvGrd = st.slider(
            label='Total rooms above grade (does not include bathrooms)',
            min_value=2, max_value=14, value=7, step=1)
    fireplaces = st.slider(
            label='Number of fireplaces',
            min_value=0, max_value=3, value=1, step=1)
    garageCars = st.slider(
            label='Size of garage in car capacity',
            min_value=0, max_value=4, value=2, step=1)
    col1, col2 = st.columns(2)
    with col1:
        garageArea = st.slider(
            label='Size of garage in square feet',
            min_value=0, max_value=1418, value=744, step=1)
    with col2:
        garageYrBlt = st.slider(
            label='Year garage was built',
            min_value=1900, max_value=2010, value=2000, step=1)
    woodDeckSF = st.slider(
            label='Wood deck area in square feet',
            min_value=0, max_value=857, value=400, step=1)
    openPorchSF = st.slider(
            label='Open porch area in square feet',
            min_value=0, max_value=547, value=300, step=1)
    enclosedPorch = st.slider(
            label='Enclosed porch area in square feet',
            min_value=0, max_value=552, value=300, step=1)
    threeSsnPorch = st.slider(
            label='Three season porch area in square feet',
            min_value=0.00, max_value=6.23, step=0.1, value=1.00)
    screenPorch = st.slider(
            label='Screen porch area in square feet',
            min_value=0.00, max_value=6.17, step=0.1, value=1.00)
    poolArea = st.slider(
            label='Pool area in square feet',
            min_value=0.00, max_value=6.60, value=3.60, step=0.1)
    miscVal = st.slider(
            label='Value of miscellaneous feature',
            min_value=0.00, max_value=9.64, value=7.00, step=0.1)
    month_mapping = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
    }
    monthSold = st.selectbox(
        label='Month Sold',
        options=list(month_mapping.keys()))
    monthSold = month_mapping[monthSold]
    neighborhood_mapping = {
    'Bloomington Heights': "Blmngtn",
    'Bluestem': "Blueste",
    'Briardale': "BrDale",
    'Brookside': "BrkSide",
    'Clear Creek': "ClearCr",
    'College Creek': "CollgCr",
    'Crawford': "Crawfor",
    'Edwards': "Edwards",
    'Gilbert': "Gilbert",
    'Iowa DOT and Rail Road': "IDOTRR",
    'Meadow Village': "MeadowV",
    'Mitchell': "Mitchel",
    'North Ames': "NAmes",
    'Northpark Villa': "NPkVill",
    'Northwest Ames': "NWAmes",
    'Northridge': "NoRidge",
    'Northridge Heights': "NridgHt",
    'Old Town': "OldTown",
    'South & West of Iowa State University': "SWISU",
    'Sawyer': "Sawyer",
    'Sawyer West': "SawyerW",
    'Somerset': "Somerst",
    'Stone Brook': "StoneBr",
    'Timberland': "Timber",
    'Veenker': "Veenker"
    }
    Neighborhood = st.selectbox(
        label='Physical locations within Ames city limits',
        options=list(neighborhood_mapping.keys()))
    Neighborhood = neighborhood_mapping[Neighborhood]
    masvnrtype_mapping = {
    'Brick Common': "BrkCmn",
    'Brick Face': "BrkFace",
    'Cinder Block': "CBlock",
    'None': "None",
    'Stone': "Stone"
    }
    MasVnrType = st.selectbox(
        label='Masonry veneer type',
        options=list(masvnrtype_mapping.keys()))
    MasVnrType = masvnrtype_mapping[MasVnrType]
    if st.button('Submit'):
        dataframe = {
            "MSSubClass": [selected_value],
            "LotFrontage": [lotFrontage],
            "LotArea": [lotArea],
            "OverallQual": [OverallQual],
            "YearBuilt": [yearBuilt],
            "YearRemodAdd": [yearRemodAdd],
            "MasVnrArea": [MasVnrArea],
            "ExterQual": [ExterQual],
            "BsmtQual": [BsmtQual],
            "BsmtFinSF1": [BsmtFinSF1],
            "BsmtUnfSF": [BsmtUnfSF],
            "TotalBsmtSF": [TotalBsmtSF],
            "1stFlrSF": [firstFlrsf],
            "2ndFlrSF": [secondFlrsf],
            "LowQualFinSF": [lowQualFinSF],
            "GrLivArea": [GrLivArea],
            "BsmtFullBath": [BsmtFullBath],
            "FullBath": [fullBath],
            "HalfBath": [halfBath],
            "BedroomAbvGr": [bedroomAbvGr],
            "KitchenAbvGr": [kitchenAbvgGr],
            "KitchenQual": [KitchenQual],
            "TotRmsAbvGrd": [totrmsAbvGrd],
            "Fireplaces": [fireplaces],
            "GarageYrBlt": [garageYrBlt],
            "GarageCars": [garageCars],
            "GarageArea": [garageArea],
            "WoodDeckSF": [woodDeckSF],
            "OpenPorchSF": [openPorchSF],
            "EnclosedPorch": [enclosedPorch],
            "3SsnPorch": [threeSsnPorch],
            "ScreenPorch": [screenPorch],
            "PoolArea": [poolArea],
            "MiscVal": [miscVal],
            "MoSold": [monthSold],
            "Neighborhood": [Neighborhood],
            "MasVnrType": [MasVnrType]
        }
        result = preprocessing.make_prediction(dataframe)
        result = round(float(result[0]),2)
        st.text("Harga prediksi rumah adalah ${}".format(result))
        


