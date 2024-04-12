import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, boxcox, f_oneway
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler, FunctionTransformer, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.drop('Id', axis=1, inplace=True)
    df.dropna(axis=0)
    return df

def initial_exploration(df, name, continuous, discrete):
    print(df.shape)
    print(df.index)
    print(continuous)
    print(discrete)

    print("First few rows of the DataFrame:")
    print(df.head())

    print("\nData types of each column:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSummary statistics being made...")
    df.describe().to_csv(f'../results/data_analysis/desc_{name}.csv', index=True)
    
    print("\nPlotting Histograms...")
    if not os.path.exists(f'../results/data_analysis/histograms_{name}'):
        os.mkdir(f'../results/data_analysis/histograms_{name}')
    numerical_features = df[continuous]
    for column in numerical_features.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(x=df[column])
        title = f'Histogram of {column}'
        plt.title(title, fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig(f'../results/data_analysis/histograms_{name}/{title}.png')
    
    print("\nCounting Discretes...")
    if not os.path.exists(f'../results/data_analysis/countplots_{name}'):
        os.mkdir(f'../results/data_analysis/countplots_{name}')
    categorical_features = df[discrete]
    for column in categorical_features.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(x=df[column])
        title = f'Count plot of {column}'
        plt.title(title, fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.savefig(f'../results/data_analysis/countplots_{name}/{title}.png')

def correlation(df, name, continuous, discrete):
    # continuous
    print("\nCorrelating...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[continuous].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix for Continuous Features', fontsize=16)
    plt.savefig(f'../results/data_analysis/correlation_{name}.png')
    
    corr_matrix = df[continuous].corr()
    corr_matrix.to_csv(f'../results/data_analysis/correlation_{name}.csv', index=True)
    print("\nTabulated Correlation Coefficients being made...")
    
    # discrete
    print("\nBoxing...")
    boxplot_stats = pd.DataFrame(columns=['Feature', 'Median', 'Mean', 'Std Deviation', 'Min', '25th Percentile', 
                                          '50th Percentile (Median)', '75th Percentile', 'Max'])
    
    if not os.path.exists(f'../results/data_analysis/boxplots_{name}'):
        os.mkdir(f'../results/data_analysis/boxplots_{name}')
    for column in discrete:
        median = df.groupby(column)['SalePrice'].median()
        mean = df.groupby(column)['SalePrice'].mean()
        std_dev = df.groupby(column)['SalePrice'].std()
        minimum = df.groupby(column)['SalePrice'].min()
        q1 = df.groupby(column)['SalePrice'].quantile(0.25)
        q2 = df.groupby(column)['SalePrice'].quantile(0.5)
        q3 = df.groupby(column)['SalePrice'].quantile(0.75)
        maximum = df.groupby(column)['SalePrice'].max()
        
        boxplot_stats = boxplot_stats.append({'Feature': column, 'Median': median, 'Mean': mean, 
                                              'Std Deviation': std_dev, 'Min': minimum, '25th Percentile': q1, 
                                              '50th Percentile (Median)': q2, '75th Percentile': q3, 'Max': maximum}, 
                                              ignore_index=True)
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(x=df[column])
        title = f'Box Plot of {column} vs SalePrice'
        plt.title(title, fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.ylabel('SalePrice', fontsize=14)
        plt.xticks(rotation=45)
        plt.savefig(f'../results/data_analysis/boxplots_{name}/{column}.png')
        
    boxplot_stats.to_csv(f'../results/data_analysis/boxplots_{name}.csv', index=False)
    
    print("ANOVA Testing...")
    anova_results = pd.DataFrame(columns=['Feature', 'F-value', 'p-value'])
    
    for feature in discrete:
        category_groups = [group['SalePrice'] for _, group in df.groupby(feature)]
        f_value, p_value = f_oneway(*category_groups)
        anova_results = anova_results.append({'Feature': feature, 'F-value': f_value, 'p-value': p_value}, 
                                             ignore_index=True)

    anova_results.to_csv(f'../results/data_analysis/anova_{name}.csv', index=False)
    
def preprocessing(df, continuous, discrete):
    # Imputeing
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_columns]), columns=numeric_columns)
    df_categorical = pd.get_dummies(df[categorical_columns])
    df = pd.concat([df_numeric.reset_index(drop=True), df_categorical.reset_index(drop=True)], axis=1)
    
    # Print index information
    print("Index information after Imputeing:")
    print(df.index)
    
    # Outlier Detection and Removal
    outlier_scaler = RobustScaler()
    scaled_features_outlier = outlier_scaler.fit_transform(df[continuous])
    df_outlier_scaled = pd.DataFrame(scaled_features_outlier, columns=continuous)
    
    # Print index information
    print("Index information after Outlier Detection and Removal:")
    print(df_outlier_scaled.index)
    
    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[continuous])
    df_scaled = pd.DataFrame(scaled_features, columns=continuous)
    
    # Print index information
    print("Index information after Feature Scaling:")
    print(df_scaled.index)
    
    # Feature Transformation
    for column in continuous:
        _, p_value = shapiro(df[column])
        
        if p_value < 0.05:
            if df[column].min() > 0:
                df[column] = np.log(df[column])
            else:
                df[column] = boxcox(df[column] + np.abs(df[column].min()) + 1)[0]
                
    # Feature Encoding
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(df[discrete])
    encoded_columns = encoder.get_feature_names_out(discrete)
    df_encoded = pd.DataFrame(encoded_features.tolist(), columns=encoded_columns)
    
    # Print index information
    print("Index information after Feature Encoding:")
    print(df_encoded.index)
    
    # Normalization
    normalizer = Normalizer()
    normalized_features = normalizer.fit_transform(df[continuous])
    df_normalized = pd.DataFrame(normalized_features, columns=continuous)

    # Concatenate the preprocessed features
    df_preprocessed = pd.concat([df_outlier_scaled.reset_index(drop=True), 
                                df_scaled.reset_index(drop=True), 
                                df_encoded.reset_index(drop=True), 
                                df_normalized.reset_index(drop=True)], axis=1)
    
    # Print index information
    print("Index information after Concatenation:")
    print(df_preprocessed.index)
    
    # Headers Management
    discrete.extend(encoded_columns)
    newc = []
    newd = []
    for h in continuous:
        if h not in newc:
            newc.append(h)
    for h in discrete:
        if h not in newd:
            newd.append(h)
    continuous = newc
    discrete = newd
    
    return df_preprocessed, continuous, discrete

def main():
    discrete = [
        'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold'
    ]
    continuous = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice'
    ]
    file_path = "../data/train.csv"
    df = load_dataset(file_path)
    initial_exploration(df, 'raw', continuous, discrete)
    correlation(df, 'raw', continuous, discrete)
    df_processed, continuous, discrete = preprocessing(df, continuous, discrete)
    for h in ['BedroomAbvGr', 'MSSubClass', 'OverallQual', 'BsmtFullBath', 'Fireplaces', 'HalfBath', 'TotRmsAbvGrd', 'YearBuilt', 'BsmtHalfBath', 'FullBath', 'YrSold', 'YearRemodAdd', 'GarageCars', 'MoSold', 'OverallCond', 'KitchenAbvGr']:
        discrete.remove(h)
    df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]
    initial_exploration(df_processed, 'processed', continuous, discrete)
    correlation(df_processed, 'processed', continuous, discrete)
    
    if not os.path.exists('../data/processed'):
        os.mkdir('../data/processed')
    df_processed.to_csv('../data/processed/train_processed.csv')
    
    df = load_dataset('../data/test.csv')
    df_processed, c, d = preprocessing(df, discrete=[
        'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold'
    ], continuous=[
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
    ])
    df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]
    df_processed.to_csv('../data/processed/test_processed.csv')

if __name__ == "__main__":
    main()
