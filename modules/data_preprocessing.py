import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the dataset."""
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data: drop columns, create dummies, and scale features."""
    # Dropping unnecessary columns
    df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    
    # Define numerical and categorical columns
    num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 
                'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked',
                'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                'TrainingTimesLastYear']
    
    cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 
                'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 
                'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']
    
    # Create dummy variables
    to_get_dummies_for = ['BusinessTravel', 'Department', 'Education', 'EducationField', 
                          'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 
                          'JobRole', 'MaritalStatus']
    
    df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)
    
    # Map categorical columns
    dict_OverTime = {'Yes': 1, 'No': 0}
    dict_attrition = {'Yes': 1, 'No': 0}
    
    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)
    
    # Separate features and target variable
    Y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    
    # Scale the features
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, Y

def split_data(X, Y, test_size=0.3):
    """Split the data into train and test sets."""
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=1, stratify=Y)
    return x_train, x_test, y_train, y_test

def metrics_score(actual, predicted):
    """Print the classification report and confusion matrix."""
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()