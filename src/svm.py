from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from src.dataset import dataset

def run_kernel_svm(X, y, kernel='rbf', C=1.0, gamma='scale', test_size=0.2, random_state=None):
    """
    Function to train and evaluate a kernel SVM model.

    Parameters:
    - X: Features (numpy array or pandas DataFrame).
    - y: Target labels (numpy array or pandas Series).
    - kernel: Specifies the kernel type to be used ('linear', 'poly', 'rbf', 'sigmoid').
    - C: Regularization parameter.
    - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels.
    - test_size: Fraction of data to be used for testing.
    - random_state: Seed for reproducibility.

    Returns:
    - model: Trained SVM model.
    - accuracy: Accuracy of the model on the test set.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y,shuffle=True)

    # Create the SVM model with the specified kernel and parameters
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and return accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

if __name__ == "__main__":
    main_data_dir = r"D:"
    palimpsest_name = r"Verona_msXL"
    main_dir = os.path.join(main_data_dir, palimpsest_name)
    classes_dict = {"undertext": 1, "not_undertext": 0}
    modalities = ["M"]
    folios_train = ["msXL_335v_b"]
    win = 0
    box = None
    features_train,_ =  dataset(main_dir,folios_train,[],classes_dict,modalities,win)
    print("Features shape", features_train[0].shape)
    print("Labels shape", features_train[1].shape)
    model,accuracy = run_kernel_svm(features_train[0], features_train[1])