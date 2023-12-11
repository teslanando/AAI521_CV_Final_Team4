# AAI521_CV_Final_Team4
University of San Diego, Computer Vision final project. 

# Video presentation: 
    https://www.youtube.com/watch?v=U32sg_eIZ4s

Dataset for FaceRec: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition 

# Make sure you have the necessary dependencies installed: 
  !pip install face_recognition opencv-python numpy scikit-learn 

# FaceRec Model
This model is able to perform face recognition via webcam or by comparing two already saved images.

  1. Run all the cells from FaceRec.ipynb in order.
  2. During EDA, you can see what kind of data you are working with. In this case, our dataset contains 17,534 celebrity pictures, but you can use your own dataset for training.
  3. Perform some preprocessing to make it faster. We randomly selected 10 pictures from the dataset and created a new folder named dataset_new.
  4. Then from dataset_new we selected the first 5 images and dropped them into another folder named training_set (the model will perform well even with 1 picture per celebrity)


  Usage:

  Instantiate the FaceRec class.
  Load known face encodings by calling the load_encoding_images method with the path to the folder containing training images.
  Use the detect_known_faces method to recognize faces in a video stream or frames.

  Example:

    # Instantiate FaceRec
    face_recognizer = FaceRec()

    # Load known face encodings
    face_recognizer.load_encoding_images('path/to/training/images')

    # Perform face recognition on a video frame
    frame = cv2.imread('path/to/video/frame.jpg')
    face_locations, face_names = face_recognizer.detect_known_faces(frame)

  Display the results
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Draw rectangle around the face and display the name
    # (Example: Use OpenCV drawing functions)

Metrics Calculation:
If you have ground truth labels and predicted labels, you can use the calculate_metrics method to obtain accuracy, precision, and confusion matrix.

    true_labels = [...]  # Ground truth labels
    predicted_labels = [...]  # Predicted labels
    accuracy, precision, confusion_matrix = face_recognizer.calculate_metrics(true_labels, predicted_labels)

Instantiate the FaceRec class.

  Load known face encodings by calling the load_encoding_images method with the path to the folder containing training images.
  Run the script to start real-time face recognition. 
  **Press the 'Esc' key to exit the loop** and save a snapshot of the last frame.

  ![image](https://github.com/teslanando/AAI521_CV_Final_Team4/assets/133830351/2a8b3a8d-0f30-4a4c-b1bd-2b1d3620660b)


# Image Comparision Tool

  1. Run the script to open a Tkinter window with a button to load and compare two images.
  2. Click the button, select two images using the file dialog, and observe the results.

Sample Results:

![image](https://github.com/teslanando/AAI521_CV_Final_Team4/assets/133830351/47a86792-ead5-47fa-88fa-e52f5bcb88ba)

![image](https://github.com/teslanando/AAI521_CV_Final_Team4/assets/133830351/7b7502a9-80da-488a-a754-73ee73c4c95a)


----

# FaceRecognizer model
Our second implementation similarly leveraged face-recognition library, but with the goal of streamlining the face detection process.  
We used [this dataset](https://vis-www.cs.umass.edu/lfw/) for training and validation  

**Note that you need to have CMake and a C compiler to be able to install the dependencies required to run this project.**  

We recommend to create a venv with **Python 3.9** as the project might not run properly with newer Python versions.  
Be sure to install the packages in *./face_recognizer/requirements.txt*
It expects this file heirarchy:


```|face_recognizer/
│
├── output/
│
├── training/
│   └── ben_affleck/
│       ├── img_1.jpg
│       └── img_2.png
│
├── validation/
│   ├── ben_affleck1.jpg
│   └── michael_jordan1.jpg
│
├── detector.py
├── requirements.txt
└── unknown.jpg

| data_cleaning.ipynb
```

Provided that data_cleaning script is run, the model can be trained by running the command line prompt `python detector.py --train`  
For validation, run `python detector.py --validate`  
For testing a single image, run `python detector.py --test -f img_path`  
You can choose the method that face-recognition library would use to perform the training and testing by specifying the argument `-m [hog, cnn]`  
cnn is optimized for GPU-enabled environments and hog (histogram of oriented gradients) works best for CPU.

You can choose run `python detector.py --help` to get a comprehensive list of command line arguments that you can use.  







