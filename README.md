## Endoscopy Image Retrieval

``Lead contributor/s:`` **Sharib Ali** 

Contact: <sharib.ali@eng.ox.ac.uk>

## About 

Endoscopy is a routine imaging technique used for both diagnosis and minimally invasive surgical treatment. While the endoscopy video contains a wealth of information, tools to capture this information for the purpose of clinical reporting are rather poor. In date, endoscopists do not have any access to tools that enable them to browse the video data in an efficient and user friendly manner. Fast and reliable video retrieval methods could for example, allow a clinician to review data from previous exams and therefore improve their ability to monitor disease progression. Deep learning provides new avenues of compressing and indexing video in an extremely efficient manner. In this study, we propose to use an autoencoder for efficient video compression and fast retrieval of video images. To boost the efficiency of video image retrieval and to address data variability like multi-modality and view-point changes, we propose the integration of a Siamese network.


![Alt text](images/blockDiagram.png?raw=true "Title")

## Topics:

1. Endoscopic image retrieval using [AutoEncoders](https://github.com/sharibox/endoscopyImageRetrieval/tree/master/endo-autoEncoder)
2. [Siamese](https://github.com/sharibox/endoscopyImageRetrieval/tree/master/endo-siamese) network for pruning image retrieval 


### TODO: Check compression Vs Accuracy using same size final encoded embeddings

1. Change CNN size of AE and use CNN-VAE with the same fileter size
2. Use Global average pooling and FCN for last layers
