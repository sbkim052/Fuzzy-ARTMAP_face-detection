# Real-time Face Registration and Classification System using Fuzzy ARTMAP
In this paper, we propose a real-time face registration and classification system using Fuzzy adaptive resonance theory (ART) based map architecture (Fuzzy ARTMAP). Facial recognition can be divided into face detection and face classification. Firstly, face detection was implemented by a pretrained Caffe Model using OpenCV’s dnn module. Secondly, with the detected faces, face classification was completed using Fuzzy ARTMAP. The advantages Fuzzy ARTMAP has are the capability of fast
computation based on fuzzy operation and high accuracy of classification with a small amount of data. The proposed algorithm enables not only fast face classification but also fast face registration with only a small amount of data in real-time. Experimental results showed that the system was able to generate new categories in real-time and performed high accuracy of facial recognition.

### This is an official implementation of https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09410561

## requirements
~~~
- imutils
- numpy
- skelarn
~~~

## video links
- https://www.youtube.com/watch?v=5-aRup4vgIQ&feature=youtu.be
- https://www.youtube.com/watch?v=2uepMaYM3v8&t=3s

## how to run
### place your datasets in ./dataset
### extract embeddings from training data in ./dataset
~~~
python extract_embeddings.py --dataset ./dataset --embeddings ./output/embeddings.pickle --detector ./face_detection_model --embedding-model openface_nn4.small2.v1.t7
~~~
- --dataset : dataset directory
- --embeddings : directory to save embeddings
- --detector : face detection model
- --embedding-model : feature extraction model
### run artmap for face classification
~~~
python train_fuzzy.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --embeddings ./output/embeddings.pickle --le ./output/le.pickle
~~~
- --le : label of the embeddings
- --embeddings : embeddings
- --detector : face detection model
- --embedding-model : feature extraction model

### press 'r' while running train_fuzzy.py
you can register your own face in a real-time

## reference 
- Carpenter, G. A(1992), Fuzzy ARTMAP: A neural network architecture for incremental supervised learning of analog multidimensional maps, IEEE Transactions on Neural Networks, vol 3(5)
- Carpenter, G.A., Grossberg, S., and Rosen, D.B. (1991). Fuzzy art: Fast stable learning and categorization of analog patterns by an adaptive resonance system. Neural networks, 4(6), 759–771.
