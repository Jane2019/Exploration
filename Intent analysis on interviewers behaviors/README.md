## Intent analysis on interviewers behaviors

This is a 3-month part-time project, aiming to understand how interviewers use STAR structure to answer behavioral question. As a NLP assistant, I am responsible to train a MLP neural network using Keras to predict the intent of each sentence interviewer gives. The model will give possibilities of the sentence belonging to which part of STAR. Deployed the model on Flask by other member in the team. The model has a 89% accuracy in prediction that can help interviewer by giving him feedback on whether he should say more/less on which part in STAR.

### Procedure

- Collected answers to behavioral questions and labeled 2000+ sentences in terms of STAR structure
- Built a Multilayer Perceptron (MLP) to predict intent of each sentence and gained 82% accuracy
- Processed error analysis by viewing classification report, confusion matrix and PR/AUC/ROC curves
- Improved model accuracy to 89% after cleaning text by NLTK and switching to Word Sentence Encoder
