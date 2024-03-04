# Video Content Detection Model
A Deep Learning-Based Approach for Inappropriate Content Detection and Classification of Videos

The exponential growth of videos on YouTube and other platforms has attracted billions of viewers among which the majority belongs to a young demographic.
Malicious uploaders also find this platform as an opportunity to spread upsetting visual content, such as using animated cartoon videos to share inappropriate content with children.
Therefore, an automatic real-time video content filtering mechanism is highly suggested to be integrated into social media platforms.
In this study, a novel deep learning-based architecture is proposed for the detection and classification of inappropriate content in videos. 
For this, the proposed framework employs an ImageNet pre-trained convolutional neural network (CNN) model known as EfficientNet-B7 to extract video descriptors, 
which are then fed to bidirectional long short-term memory (BiLSTM) network to learn effective video representations and perform multiclass video classification. 
An attention mechanism is also integrated after BiLSTM to apply attention probability distribution in the network. 
These models are evaluated on a manually annotated dataset of 111,156 cartoon clips collected from YouTube videos.
Experimental results demonstrated that EfficientNet-BiLSTM (accuracy D 95.66%) performs better than attention mechanism based EfficientNet-BiLSTM (accuracy D 95.30%) framework. 
Secondly, the traditional machine learning classifiers perform relatively poor than deep learning classifiers. 
Overall, the architecture of EfficientNet and BiLSTM with 128 hidden units yielded state-of-the-art performance (f1 score D 0.9267). 
Furthermore, the performance comparison against existing state-of-the-art approaches verified that BiLSTM on top of CNN captures better contextual information of video descriptors in network architecture,
and hence achieved better results in child inappropriate video content detection and classification.
