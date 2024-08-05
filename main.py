from modules.imports import *
from modules.data_preprocessing import *
from modules.utils import *
from modules.model_training import *

def main():
    file_path = 'C:\\Users\\kaurr\\OneDrive\\Desktop\\BISI\\2208\\Employee attrition using SVM\\data\\HR_Employee_Attrition.xlsx'
    df = load_data(file_path)
    X, Y = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(X, Y)
    logistic_regression(x_train, y_train, x_test, y_test)
    svm_model(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
