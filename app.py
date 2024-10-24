import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import altair as alt
import seaborn as sns
import klib

# Function to handle categorical encoding
def encode_categorical(df, categorical_columns):
    if categorical_columns:
        st.info(f"Encoding categorical columns: {', '.join(categorical_columns)}")
        # Apply One-Hot Encoding to categorical columns
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df

# Function to convert numerical columns to proper types
def clean_numerical_columns(df, numerical_columns):
    # Convert numerical columns to suitable types
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    return df

# Function to display column information
def display_column_info(df):
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            "Column Name": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Data Type (Dtype)": df.dtypes.values
        })
        st.dataframe(col_info)

st.title("Clustering Algorithms")

st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None: 
    df = pd.read_csv(uploaded_file)[:600]
    df = klib.data_cleaning(df)  # Optional step using klib for basic cleaning
    st.write("Dataset preview:")
    st.dataframe(df)

    # Show column information in a clear way
    display_column_info(df)

    # Step 2: Select numerical and categorical columns
    st.sidebar.header("Select Features for Clustering")
    
    # Select numerical columns
    numerical_columns = st.sidebar.multiselect("Select numerical columns", df.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns.tolist())
    
    # Select categorical columns
    categorical_columns = st.sidebar.multiselect("Select categorical columns", df.select_dtypes(include=['object', 'category']).columns.tolist())
    
    if len(numerical_columns) + len(categorical_columns) < 2:
        st.warning("Please select at least 2 columns for clustering.")
    else:
        # Step 3: Clean and encode data
        df_selected = df[numerical_columns + categorical_columns].copy()
        
        # Clean numerical columns
        df_selected = clean_numerical_columns(df_selected, numerical_columns)
        
        # Encode categorical columns
        df_selected = encode_categorical(df_selected, categorical_columns)

        # Handle missing values (drop rows with NaN values)
        if df_selected.isnull().sum().sum() > 0:
            st.warning("Some non-numeric values were found and have been converted to NaN. Rows with NaN will be removed.")
            df_selected.dropna(inplace=True)

        if df_selected.empty:
            st.error("No valid numeric data available for clustering after cleaning.")
        else:
            # Capture the new column names after encoding
            encoded_columns = df_selected.columns.tolist()

            # Step 4: Choose clustering algorithm
            st.sidebar.header("Select Clustering Algorithm")
            algorithm = st.sidebar.selectbox("Choose Algorithm", 
                                             ("KMeans", "DBSCAN", "Agglomerative Clustering", 
                                              "Mean Shift", "Spectral Clustering"))

            # Preprocess data: scale the selected columns
            X = df_selected.values
            X_scaled = StandardScaler().fit_transform(X)

            # Step 5: Apply chosen algorithm
            if algorithm == "KMeans":
                k = st.sidebar.slider("Select number of clusters", 2, 10, value=3)
                model = KMeans(n_clusters=k)
            elif algorithm == "DBSCAN":
                eps = st.sidebar.slider("Select epsilon", 0.1, 10.0, value=0.5)
                min_samples = st.sidebar.slider("Select min samples", 1, 10, value=5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algorithm == "Agglomerative Clustering":
                n_clusters = st.sidebar.slider("Select number of clusters", 2, 10, value=3)
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif algorithm == "Mean Shift":
                model = MeanShift()
            elif algorithm == "Spectral Clustering":
                n_clusters = st.sidebar.slider("Select number of clusters", 2, 10, value=3)
                model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')

            # Fit the model
            labels = model.fit_predict(X_scaled)
            df_selected['Cluster'] = labels

            # Step 6: Attribute selection for visualization
            st.subheader("Attribute Selection for Visualization")
            visualization_columns = st.multiselect(
                "Select two attributes for scatter plot visualization:",
                numerical_columns + categorical_columns,
                default=numerical_columns[:2] if len(numerical_columns) >= 2 else categorical_columns[:2]
            )

            # Step 7: Visualize the clusters using Altair (Streamlit's built-in plotting)
            if len(visualization_columns) == 2:
                scatter = alt.Chart(df_selected).mark_circle(size=60).encode(
                    x=visualization_columns[0],
                    y=visualization_columns[1],
                    color=alt.Color('Cluster:N', legend=alt.Legend(title="Cluster")),
                    tooltip=[visualization_columns[0], visualization_columns[1], 'Cluster']
                ).interactive()

                st.altair_chart(scatter, use_container_width=True)

                st.write("Clustering results:")
                st.write(df_selected)
            else:
                st.warning("Please select exactly 2 attributes for visualization.")
