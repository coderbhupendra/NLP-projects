# NLP-projects
<<<<<<< HEAD
Natural Language Processing related project done in IIT Madras. Programming language used :Python and theano framework

In this there are three projects: 1)language_translation_en : In this i made a rnn_encoder_decoder model to train the parallel corpus of English and German. rnn_encoder_decoder.py this contain the Rnn_encoder_decoder neural model. train.py uses above Rnn_encoder_decoder neural model to train my language dataset. so train.py and rnn_encoder_decoder are important files.

Other code files are used for making training and test dataset , shuffling them , checking is the dataset formed is correct or not and to see if model works correctly or not. data_rnn_classify.py is used to make a training and test data set for wrong correct(w/c) approach lstm_model_check.py is used for checking the final result of prediction for w/c approach

>>>>>>> 2f254283309ec255c830a8999c0f9002e3ba690c

For both languages one ngram_model is made inorder to learn the Emdedding Vector for words in dictionary , dictionary contain top 10,000 frequent words of corpus , for ngram_model i used a RNN neural network to predict the n+1 th word given n previous words.And these embedding vectors will be used by rnn_encoder_decoder model.

2)Similary language_translation_de does the samething for German dataset. 3)language_translation is used to translate a English sentence to German , using above models only.
