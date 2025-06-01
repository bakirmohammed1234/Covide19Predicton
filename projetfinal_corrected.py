import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

import joblib, dill

df=pd.read_csv("D:\\MlAIM\\BigData\\Covid Data.csv")

from sklearn.base import BaseEstimator, TransformerMixin


class DuplicatesRemover(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        if y is not None and X.index.equals(y.index):
            X_y = pd.concat([X, y], axis=1)
            self.unique_indices = X_y.drop_duplicates().index
        print('1. \tDuplicatesRemover fitted')
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        if y is not None and X.index.equals(y.index):
            X_transformed = X.loc[self.unique_indices]
            y_transformed = y.loc[self.unique_indices]
            print("Transform - X shape:", X_transformed.shape)
            print("Transform - y shape:", y_transformed.shape)
            return X_transformed, y_transformed
            print('1. \tDuplicatesRemover transform')
        print('1. \tDuplicatesRemover transform')
        return X  # Simply pass X through if no y

    def fit_transform(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        self.fit(X, y)
        return self.transform(X, y)


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)
    #     return self.transform(X)

class SpecificFeaturesConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        print('2. \tSpecificFeaturesConverter fitted')
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        X['CLASIFFICATION_FINAL'] = X['CLASIFFICATION_FINAL'].apply(lambda x: 0 if x > 3 else x)
        X.loc[X['SEX'] == 2, 'PREGNANT'] = 0
        X.loc[X['PATIENT_TYPE'] == 1, 'ICU'] = 0
        X.loc[X['PATIENT_TYPE'] == 1, 'INTUBED'] = 0
        print('2. \tSpecificFeaturesConverter transform')
        if y is not None:
            return X, y
        return X

    def fit_transform(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        self.fit(X, y)
        return self.transform(X, y)


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class PlaceholderReplacer(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        print('3. \tPlaceholderReplacer fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        cols_to_convert = X.columns[X.columns != 'AGE']
        X.loc[:, X.columns != 'AGE'] = X.loc[:, X.columns != 'AGE'].astype(float)
        X[cols_to_convert] = X[cols_to_convert].astype(float)
        X[cols_to_convert] = X[cols_to_convert].replace([97, 98, 99], np.nan)
        print('3. \tPlaceholderReplacer transform')
        return pd.DataFrame(X)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class BinaryConverter(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        print('4. \tBinaryConverter fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        for col in X.columns:
            if col != 'AGE':
                if X[col].dtype == 'int64' or X[col].dtype == 'float64':
                    X[col] = X[col].astype(float)  # Explicitly cast to float for compatibility
                    X[col] = X[col].replace(2, 0)
                    X[col] = X[col].replace([97, 98, 99], np.nan)
        # X.loc[:, X.columns != 'AGE'] = X.loc[:, X.columns != 'AGE'].replace(2, 0)
        # X.loc[:, X.columns != 'AGE'] = X.loc[:, X.columns != 'AGE'].replace([97, 98, 99], np.nan)
        print('4. \tBinaryConverter transform')
        return pd.DataFrame(X)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform
        
class MinMaxScaler_own(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.scaled_columns = None
        
    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        self.scaler.fit(X)
        self.scaled_columns = X.columns  # Store the original column names
        print('5.1. \tMinMaxScaler_own fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        print('X shape = ', X.shape)
        scaled_data = self.scaler.transform(X)
        scaled_data = pd.DataFrame(scaled_data, columns=self.scaled_columns, index=X.index)
        print('5.1. \t(Num features) MinMaxScaler_own transform')
        scaled_data.head(1)
        return scaled_data


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class IterativeImputer_num_own(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        print('5.1.1. \tIterativeImputer_num_own fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        impute_num_cols = X.columns
        cols_to_convert = X.columns[X.columns != 'AGE']
        X[cols_to_convert] = X[cols_to_convert].astype(float)
        X[cols_to_convert] = X[cols_to_convert].replace([97, 98, 99], np.nan)
        print('5.1.1 \t(Num features) IterativeImputer_num_own transform')
        pd.DataFrame(X, columns = impute_num_cols).head(1)
        return pd.DataFrame(X, columns = impute_num_cols)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

# class OneHotEncoderWithNames(BaseEstimator, TransformerMixin):
#     def __init__(self, drop=None, sparse_output=False, handle_unknown='ignore'):
#         self.drop = drop
#         self.sparse_output = sparse_output
#         self.handle_unknown = handle_unknown
#         self.encoded_columns = None
#         self.encoder = OneHotEncoder(drop=self.drop, sparse_output=self.sparse_output, handle_unknown=self.handle_unknown)

#     def fit(self, X, y=None):
#         self.encoder.fit(X)
#         self.encoded_columns = self.encoder.get_feature_names_out(X.columns)
#         print('5.2. \tOneHotEncoderWithNames fitted')
#         self.fitted_ = True
#         return self

#     def transform(self, X):
#         check_is_fitted(self, 'fitted_')
#         encoded_data = self.encoder.transform(X)
#         if hasattr(encoded_data, "toarray"):
#             encoded_data = encoded_data.toarray()
#         encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_columns, index=X.index)
#         print('5.2. \t(Cat features) OneHotEncoderWithNames transform')
#         return encoded_df

class OneHotEncoderWithNames(BaseEstimator, TransformerMixin):
    def __init__(self, drop=None, sparse_output=False, handle_unknown='ignore'):
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.encoded_columns = None
        self.encoder = OneHotEncoder(drop=self.drop, sparse_output=self.sparse_output, handle_unknown=self.handle_unknown)

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        self.encoder.fit(X)
        self.encoded_columns = self.encoder.get_feature_names_out(X.columns)
        print('5.2. \tOneHotEncoderWithNames fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, 'fitted_')
        unknown_category_warnings = []

        for col in X.columns:
            unique_categories = set(X[col].unique()) - set(self.encoder.categories_[0])
            if unique_categories:
                unknown_category_warnings.append((col, unique_categories))
        
        if unknown_category_warnings:
            print("Unknown categories found during transform: ", unknown_category_warnings)

        encoded_data = self.encoder.transform(X)
        if hasattr(encoded_data, "toarray"):
            encoded_data = encoded_data.toarray()
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_columns, index=X.index)
        print('5.2. \t(Cat features) OneHotEncoderWithNames transform')
        return encoded_df


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class IterativeImputer_cat_own(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        print('5.2.1. \tIterativeImputer_cat_own fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        impute_cat_cols = X.columns
        cols_to_convert = X.columns[X.columns != 'AGE']
        X[cols_to_convert] = X[cols_to_convert].astype(float)
        X[cols_to_convert] = X[cols_to_convert].replace([97, 98, 99], np.nan)
        print('5.2.1 \t(Cat features) IterativeImputer_cat_own transform')
        return pd.DataFrame(X, columns = impute_cat_cols)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

# Define cat_features and num_features before the train_test_split ***
cat_features = ['MEDICAL_UNIT', 'CLASIFFICATION_FINAL']
num_features = df.columns.difference(cat_features + ['DATE_DIED']).tolist()

num_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler_own()),
    ('imputer', IterativeImputer_num_own())
])

cat_transformer = Pipeline(steps=[
    # ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ('encoder', OneHotEncoderWithNames(drop='first', sparse_output=False, handle_unknown='ignore')), # Use custom encoder
    ('imputer', IterativeImputer_cat_own())
])

preprocessor = ColumnTransformer(transformers=[
    ('num_features', num_transformer, num_features),
    ('cat_features', cat_transformer, cat_features)
])

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

class CustomPipeline(Pipeline):
    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, 'fit_transform'):
                print(f"Fitting and transforming: {name}")
                result = transformer.fit_transform(X, y)
                if isinstance(result, tuple) and len(result) == 2:
                    X, y = result
                else:
                    X = result
                print(f"After {name} - X shape: {X.shape}, y shape: {y.shape if y is not None else 'N/A'}")
            else:
                print(f"Fitting: {name}")
                X = transformer.fit(X).transform(X)
                print(f"After {name} - X shape: {X.shape}")
        
        final_step_name, final_estimator = self.steps[-1]
        print(f"Fitting final estimator: {final_step_name}")
        final_estimator.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, 'fitted_')
        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, 'transform'):
                print(f"Transforming: {name}")
                X = transformer.transform(X)
                print(f"After {name} - X shape: {X.shape}")
        return X

    def predict(self, X):
        import pandas as pd
        import numpy as np
        import seaborn as sns

        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer

        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.pipeline import Pipeline, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn import set_config

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        import joblib, dill
        X = self.transform(X)
        final_step_name, final_estimator = self.steps[-1]
        return final_estimator.predict(X)

def create_pipeline():
    import pandas as pd
    import numpy as np
    import seaborn as sns

    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer

    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    from sklearn.pipeline import Pipeline, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn import set_config

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    import joblib, dill
    pipeline = CustomPipeline(steps=[
        ('duplicates_remover', DuplicatesRemover()),
        ('specific_feature_converter', SpecificFeaturesConverter()),
        ('placeholder_replacer', PlaceholderReplacer()),
        ('binary_converter', BinaryConverter()),
        ('preprocessor', preprocessor),
        ('final_imputer', SimpleImputer(strategy='mean'))  
    ])
    return pipeline

# Now, separate the dependent and independent variables:
X = df.drop('DATE_DIED', axis=1)
y = df['DATE_DIED']

# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply pre-processing_pipeline to X_train and y_train
pipeline = create_pipeline()

# Fit the pipeline on training data
X_train_transformed = pipeline.fit_transform(X_train, y_train)
duplicates_remover = pipeline.named_steps['duplicates_remover']
y_train_transformed = y_train.loc[duplicates_remover.unique_indices].apply(lambda x: 0 if x == '9999-99-99' else 1).reset_index(drop=True)

# Define the models
models = {
    'LogisticRegression': LogisticRegression(verbose=2),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier()
}

# Define cross-validation strategies
cv = {
    'kfold': KFold(n_splits=2, shuffle=True, random_state=42),
    'skfold': StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
}

# Variables to store the best model with score
best_score = 0
best_model = None

# Loop through models and cross-validation strategies
for model_name, model in models.items():
    for cv_name, cv_strategy in cv.items():
        print(f"Training {model_name} with {cv_name}...")
        scores = cross_val_score(model, X_train_transformed, y_train_transformed, cv=cv_strategy)
        mean_score = scores.mean()
        print(f"{model_name} with {cv_name} scored {mean_score}")

        # Check if this model is the best so far
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_name

print(f"Best Model is - {best_model} with score {best_score}")


param_grid = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200]
}


from sklearn.model_selection import RandomizedSearchCV
rscv = RandomizedSearchCV(
    estimator=LogisticRegression(), 
    param_distributions=param_grid, 
    n_iter=20, 
    return_train_score=True, 
    scoring='accuracy', 
    verbose=1, 
    cv= 4
)

# Fit the training pipeline on training data
pipeline = create_pipeline()
X_train_transformed = pipeline.fit_transform(X_train, y_train)
duplicates_remover = pipeline.named_steps['duplicates_remover']
y_train_transformed = y_train.loc[duplicates_remover.unique_indices].apply(lambda x: 0 if x == '9999-99-99' else 1).reset_index(drop=True)

rscv.fit(X_train_transformed, y_train_transformed)

best_params = rscv.best_params_

# Define final pipeline
def create_final_pipeline(best_params):
    import pandas as pd
    import numpy as np
    import seaborn as sns

    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer

    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    from sklearn.pipeline import Pipeline, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn import set_config

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    import joblib, dill
    final_pipeline = CustomPipeline([
        ('duplicates_remover', DuplicatesRemover()),
        ('specific_feature_converter', SpecificFeaturesConverter()),
        ('placeholder_replacer', PlaceholderReplacer()),
        ('binary_converter', BinaryConverter()),
        ('preprocessor', preprocessor),
        ('final_imputer', SimpleImputer(strategy='mean')),  
        ('logistic_regression', LogisticRegression(n_jobs=-1, **best_params))
    ])
    return final_pipeline

# Create the final pipeline using the function
final_pipeline = create_final_pipeline(best_params)

# Fit the final pipeline on the training data
final_pipeline.fit(X_train, y_train.apply(lambda x: 0 if x == '9999-99-99' else 1).astype(int))


with open("Covid-19_Death_Predict_Pipeline.pkl", "wb") as f:
    dill.dump(final_pipeline, f)

print("Pipeline saved successfully.")

with open("Covid-19_Death_Predict_Pipeline.pkl", "rb") as f:
    pipeline = dill.load(f)

# Predict the target:
y_pred = pipeline.predict(X_test)
print("Predictions completed.")

y_test_transformed = y_test.apply(lambda x: 0 if x == '9999-99-99' else 1)
print('accuracy_score:', accuracy_score(y_test_transformed, y_pred))
print('\nconfusion_matrix:\n', confusion_matrix(y_test_transformed, y_pred))
print('\nclassification_report:\n', classification_report(y_test_transformed, y_pred))

X_test.shape

y_test_transformed


y_pred0 = pipeline.predict(X_test[0:1])

