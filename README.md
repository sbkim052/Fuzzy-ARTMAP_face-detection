# Fuzzy-ARTMAP_face-detection

## reference 
Carpenter, G. A(1992), Fuzzy ARTMAP: A neural network architecture for incremental supervised learning of analog multidimensional maps, IEEE Transactions on Neural Networks, vol 3(5)

##
imutils
numpy


## video link
https://www.youtube.com/watch?v=5-aRup4vgIQ&feature=youtu.be

## how to run
1. extract embeddings for training data

"python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7"

2. run artmap for face detection 

"python train_fuzzy.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle"

## ARTMAP.py
algorithms for ART + ARTMAP
