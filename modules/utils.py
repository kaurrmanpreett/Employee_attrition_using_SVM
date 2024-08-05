from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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
