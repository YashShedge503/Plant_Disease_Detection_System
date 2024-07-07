import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

trainimg, testimg = [], []
trainlb, testlb = [], []

def preprocess_images(input_folder_path, output_folder_path, label, no):
    # Define the folder path
    image_names = [f"{label}({i}).jpg" for i in range(no)]
    
    for image_name in image_names:
        # Get the file path
        image_path = os.path.join(input_folder_path, image_name)
        
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            continue

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        image = cv2.resize(image, (256, 256))
        
        # Improve contrast
        image = cv2.equalizeHist(image)
        
        # Save image
        output_file = os.path.join(output_folder_path, image_name)
        cv2.imwrite(output_file, image)

def preprocess():
    # Preprocessing image
    # Potato Healthy Images
    input0 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Sample/0_Potato___healthy"
    output0 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Preprocessed/0_Potato___healthy"
    preprocess_images(input0, output0, 0, 152)

    # Potato Early Blight Images
    input1 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Sample/1_Potato___Early_blight"
    output1 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Preprocessed/1_Potato___Early_blight"
    preprocess_images(input1, output1, 1, 1000)
       
    # Potato Late Blight Images
    input2 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Sample/2_Potato___Late_blight"
    output2 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Preprocessed/2_Potato___Late_blight"
    preprocess_images(input2, output2, 2, 1000)

def load_images_from_folder(folder, label, no, split_ratio=0.8):
    images = []
    labels = []
    for i in range(no):
        img_path = os.path.join(folder, f"{label}({i}).jpg")
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)

    return train_test_split(images, labels, train_size=split_ratio, random_state=42)

def split():
    global trainimg, testimg, trainlb, testlb

    # Healthy
    path0 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Preprocessed/0_Potato___healthy"
    train_img0, test_img0, train_lb0, test_lb0 = load_images_from_folder(path0, 0, 152)
    trainimg.extend(train_img0)
    testimg.extend(test_img0)
    trainlb.extend(train_lb0)
    testlb.extend(test_lb0)

    # Early Blight
    path1 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Preprocessed/1_Potato___Early_blight"
    train_img1, test_img1, train_lb1, test_lb1 = load_images_from_folder(path1, 1, 1000)
    trainimg.extend(train_img1)
    testimg.extend(test_img1)
    trainlb.extend(train_lb1)
    testlb.extend(test_lb1)

    # Late Blight
    path2 = "C:/Users/Sanved/Desktop/Sanved/Codes/C++/OpenCV/Images/Preprocessed/2_Potato___Late_blight"
    train_img2, test_img2, train_lb2, test_lb2 = load_images_from_folder(path2, 2, 1000)
    trainimg.extend(train_img2)
    testimg.extend(test_img2)
    trainlb.extend(train_lb2)
    testlb.extend(test_lb2)

def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    hog_features = []
    for image in images:
        features = hog.compute(image)
        hog_features.append(features.flatten())
    
    return np.array(hog_features, dtype=np.float32)

if __name__ == "__main__":
    preprocess()
    split()

    print(len(trainimg), len(trainlb))
    print(len(testimg), len(testlb))

    # Extract HOG features for training and testing data
    X_train = extract_hog_features(trainimg)
    y_train = np.array(trainlb)
    X_test = extract_hog_features(testimg)
    y_test = np.array(testlb)

    # Train SVM model
    model = svm.SVC(kernel='rbf')
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Number of Correct Predictions: {np.sum(y_pred == y_test)}")
    print(f"Number of Incorrect Predictions: {np.sum(y_pred != y_test)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
