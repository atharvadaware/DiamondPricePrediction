from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    
    def get_data_transformation(self):
        try:
            logging.info("Data Transformation Initiated")

            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns=['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns=['cut', 'color', 'clarity']

            # Define custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )


            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")


            preprocessor=ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])


            return preprocessor


        except Exception as e:
            logging.info("Data Transformation")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load datasets
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data successfully.")

            # Get preprocessor object
            preprocessing_obj=self.get_data_transformation()

            target_column_name="price"
            drop_columns=[target_column_name,'id']

            # Separate input features and target 
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]


            logging.info("Applying preprocessor object on training and testing dataframes.  ")

            # Apply preprocessing
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Combine preprocessed features and target
            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Save preprocessor object")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )


            return (
                train_arr,
                test_arr
            )


        except Exception as e:
            raise CustomException(e,sys)