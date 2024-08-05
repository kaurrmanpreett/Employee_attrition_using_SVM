from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from .utils import metrics_score

def logistic_regression(x_train, y_train, x_test, y_test):
    """Build and evaluate a Logistic Regression model."""
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    print("Logistic Regression Model")
    print("Train Data Performance:")
    y_pred_train = lg.predict(x_train)
    metrics_score(y_train, y_pred_train)
    print("Test Data Performance:")
    y_pred_test = lg.predict(x_test)
    metrics_score(y_test, y_pred_test)

def svm_model(x_train, y_train, x_test, y_test):
    """Build and evaluate a Support Vector Machine model."""
    svm = SVC(kernel='linear')
    model = svm.fit(x_train, y_train)
    print("Support Vector Machine Model")
    print("Train Data Performance:")
    y_pred_train_svm = model.predict(x_train)
    metrics_score(y_train, y_pred_train_svm)
    print("Test Data Performance:")
    y_pred_test_svm = model.predict(x_test)
    metrics_score(y_test, y_pred_test_svm)
