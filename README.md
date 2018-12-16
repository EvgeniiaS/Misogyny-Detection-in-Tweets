# Misogyny-Detection-in-Tweets
Despite all efforts towards gender equality in the last hundred years, we still live in a male-dominated society. Nowadays, we are witnessing different kinds of abusive behavior toward women. Social media made it easier than ever to spread misogyny, and all major platforms such as Facebook or Twitter failed to control the content. Automatic detection of such posts is crucial for implementing a system aimed at eliminating hate speech towards women and potentially other similar problems. 

The goal of this project was to build a system that can identify misogynous tweets with high accuracy. For training, I used the labeled English tweet corpus from the shared task on Automatic Misogyny Classiﬁcation at IberEval 2018 (https://amiibereval2018.wordpress.com/). The corpus is composed of 3,997 tweets manually labeled by human annotators as either misogynous or non-misogynous. The final model is an ensemble of the following classifiers: 

1. An ensemble of Linear Regression, Support Vector Machine (SVM), Random Forest, Gradient Boosting, and Stochastic Gradient Descent models, trained on tweets represented as a bag of words and emoji. The accuracy on the test set is 82%. 
2. A recurrent neural network with embedding, based on pretrained 300-dimensional word vectors, and 2 LSTM layers. The accuracy on the test set is 78%. 
3. An ensemble of Random Forest and Gradient Boosting models, trained on the following features:
     i. Presence of words from a swearing list (https://www.noswearing.com/dictionary/z). 
     ii. LIWC features - reading a tweet and counts words that reflect different emotions, thinking styles, social concerns, parts of speech (http://liwc.wpengine.com/how-it-works/).  
     iii. Presence of links, @-mentions, #. 
     iv. Extracting eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive) using NRC emotion lexicon (https://saifmohammad.com/WebPages/NRCEmotion-Lexicon.htm). 
     Accuracy on the test set is 77%.
     
Accuracy of the final model on the test data is 81.75%. So, a bag of words and emoji approach was the most efficient. But the neural network with word-level embedding looks promising considering that it obtained very reasonable result despite the limited size of the training dataset.
