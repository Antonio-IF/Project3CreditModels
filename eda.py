"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: eda.py : python script with eda functionality                                               -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/Antonio-IF/Project3CreditModels                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------

class ExploratoryDataAnalysis:

    # ------------------------------

    def __init__(self, data, target_column='ESTATUS'):
        self.validate_data(data)
        self.data = data
        self.target_column = target_column
        self.number_columns = self.data.select_dtypes(include=['number']).columns
        self.object_columns = self.data.select_dtypes(include=['object']).columns

    # ------------------------------

    def validate_data(self, data):
        if data.empty:
            raise ValueError("\nDataFrame must not be empty.")

    # ------------------------------

    def data_summary(self):

        print(f"\n{'='*50}\nData summary...\n{'='*50}")

        try:
            print(f"\nHead:\n")
            print(self.data.head(5))

            print(f"\n{'-'*50}\n\nTail:\n")
            print(self.data.tail(5))

            print(f"\n{'-'*50}\n\nShape:\n")
            print(self.data.shape)

            print(f"\n{'-'*50}\n\nColumns:\n")
            print(len(self.data.columns))
            print(self.data.columns)

            print(f"\n{'-'*50}\n\nData Types:\n")
            print(self.data.dtypes)

            print(f"\n{'-'*50}\n\nNumeric columns:\n")
            print(len(self.number_columns))
            print(self.number_columns)

            print(f"\n{'-'*50}\n\nObject columns:\n")
            print(len(self.object_columns))
            print(self.object_columns)

            print(f"\n{'-'*50}\n\nUnique values:\n")
            for col in self.data.columns:
                unique_values = self.data[col].nunique()
                print(f"{col}: {unique_values}")

                if unique_values < 20:
                    print(f"Labels in {col}: {sorted(self.data[col].unique())}")

            print(f"\n{'-'*50}\n\nInformation:\n")
            print(self.data.info())

            print(f"\n{'-'*50}\n\nDescribe:\n")
            print(self.data.describe())

            print(f"\n{'-'*50}\n\nMissing values:\n")
            print(self.data.isnull().sum())

            print(f"\n{'-'*50}\n\nOutliers:\n")
            for col in self.number_columns:
                if self.data[col].dtype in ['int64', 'float64']:
                    z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                    print(f"Outliers in {col}: {np.where(z_scores > 3)}")
    
        except Exception as e:
            print(f"\nError during the data summary: {e}")

    # ------------------------------

    def data_statistics(self):

        print(f"\n{'='*50}\nData statistics...\n{'='*50}")

        try:
            numeric_data = self.data[self.number_columns]

            print(f"\nMean:\n")
            print(numeric_data.mean())

            print(f"\n{'-'*50}\n\nMedian:\n")
            print(numeric_data.median())

            print(f"\n{'-'*50}\n\nMode:\n")
            print(numeric_data.mode().iloc[0])

            print(f"\n{'-'*50}\n\nMinimum:\n")
            print(numeric_data.min())

            print(f"\n{'-'*50}\n\nMaximum:\n")
            print(numeric_data.max())

            print(f"\n{'-'*50}\n\nVariance:\n")
            print(numeric_data.var())

            print(f"\n{'-'*50}\n\nStandard Deviation:\n")
            print(numeric_data.std())

        except Exception as e:
            print(f"\nError during the data statistics: {e}")

    # ------------------------------

    def setup_subplots(self, data_columns, plot_type, title_prefix, hue=None, anaysis=None):

        n_rows = (len(data_columns) + 3) // 4

        if n_rows == 0:
            print("\nNo rows to plot.")
            return
    
        fig, axs = plt.subplots(n_rows, 4, figsize=(20, n_rows * 5))
        axs = axs.flatten()

        for i, col in enumerate(data_columns):
            try:
                ax = axs[i]

                if plot_type == 'hist':
                    sns.histplot(self.data[col], bins=30, kde=True, ax=ax, color='skyblue')
                
                elif plot_type == 'box':
                    sns.boxplot(x=self.data[col], ax=ax, color='lightgreen')
                
                elif plot_type == 'count':
                    if anaysis == 'yes':
                        sns.countplot(x=col, data=self.data, ax=ax, hue=hue, palette='viridis', legend=False)

                    else:
                        sns.countplot(x=col, data=self.data, ax=ax, hue=self.data[col], palette='viridis', legend=False)

                elif plot_type == 'violin':
                    sns.violinplot(x=self.target_column, y=self.data[col], data=self.data, ax=ax, palette='pastel', hue=self.target_column)

                ax.set_title(f'{title_prefix} {col}', fontsize=14)
                ax.set_xlabel('')
                ax.set_ylabel('Count' if plot_type == 'count' else 'Value')

            except Exception as e:
                print(f"\nError plotting {col}: {e}")

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_count_plots(self):

        print(f"\n{'='*50}\nCountplot...\n{'='*50}")

        filtered_object_columns = [col for col in self.object_columns if self.data[col].nunique() < 20]
        self.setup_subplots(filtered_object_columns, 'count', 'Count of')

    # ------------------------------

    def plot_histograms(self):

        print(f"\n{'='*50}\nHistogram...\n{'='*50}")

        self.setup_subplots(self.number_columns, 'hist', 'Distribution of')

    # ------------------------------

    def plot_box_plots(self):

        print(f"\n{'='*50}\nBoxplot...\n{'='*50}")

        self.setup_subplots(self.number_columns, 'box', 'Box plot of')

    # ------------------------------

    def plot_correlation_matrix(self):

        print(f"\n{'='*50}\nCorrelation matrix...\n{'='*50}")

        try:
            numeric_data = self.data[self.number_columns]

            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
            plt.title('Correlation Matrix:', fontsize=16)
            plt.show()

        except Exception as e:
            print(f"\nError plotting correlation matrix: {e}")

    # ------------------------------

    def plot_histograms_bivariates(self):

        print(f"\n{'='*50}\nBivariate Histogram...\n{'='*50}")

        filtered_object_columns = [col for col in self.object_columns if col != self.target_column and self.data[col].nunique() < 20]

        self.setup_subplots(filtered_object_columns, 'count', f'{self.target_column} VS', self.target_column, 'yes')

    # ------------------------------

    def plot_violins(self):

        print(f"\n{'='*50}\nViolin...\n{'='*50}")

        self.setup_subplots(self.number_columns, 'violin', f'{self.target_column} VS', self.target_column)

    # ------------------------------

    def perform_EDA(self):

        self.data_summary()
        self.data_statistics()
        self.plot_count_plots()
        self.plot_histograms()
        self.plot_box_plots()
        self.plot_correlation_matrix()
        self.plot_histograms_bivariates()
        self.plot_violins()
