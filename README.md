# Abstract
The development of better detectors, x-ray sources, as well as faster and cheaper computer processing resources, has made x-ray imaging have numerous applications: From airport security, through medical industry to food processing.
One way to get more out of x-ray imaging is to use dual-energy x-rays.
Dual-energy increases the possibilities with x-ray to measure the contents of the scanned objects. 
For many applications large amounts of data are collected since x-ray is introduced to do quality control.
This is also the case for the food processing industry, where applications of x-ray imaging can be used in-line to scan the full production.
The large amounts of data collected requires automated processing to be effective.
This thesis explores Machine Learning in the data processing pipeline of a real world machine: The Meat Master II, which generates dual-energy x-ray images.

Concretely, the challenge is to detect foreign objects in the images.
To detect the foreign objects a Convolutional Neural Network was trained.
Furthermore, the use of synthetic data was explored.
We find that it is possible to train a Convolutional Neural Network to $98.74\%$ accuracy on detecting foreign objects, using a sliding window algorithm to preprocess the data from the Meat Master II.
We observed a significant drop in accuracy when the model is evaluated on similar but yet unseen data. 
This is a significant issue since it is hard to guarantee that the training data represents the full test distribution.
It is possible to alleviate this drop in performance, as measured by the Area Under the ROC-curve, by using our synthetic data in the form of artificial foreign objects.
The accuracy is currently not good enough for real world use, but with a larger dataset to train on, it should be possible to successfully introduce machine learning to automate the detection of foreign objects in meat using machine learning.

