## NLP-projects

**Natural Language Processing related project done in IIT Madras. Programming language used: Python and theano framework**
language_translation contains files for preparing data set , vocabulary b
check_data.py to check if the data prepared is correct or not
data_no.py To calulate what percentage for words are covered by Vocabulary 
data_rnn_classify.py to randomly shuffle the wrong sentences and correct sentences
preprocessdata.py for preparing vocalbulary  of top 10,000 from prepared data set'
rnn_encoder_decoder.py '''RNN Enocoder Decoder model for encoding a sentence and decoding it into another language
lstm_model_check.py lstm model to check model by printing original(English) and translated sentences(German)
In this there are three experiments *language_translation_en : 
1.rnn_encoder_decoder model to train the parallel corpus of English and German. 
2.rnn_encoder_decoder.py this contain the Rnn_encoder_decoder neural model.
3.train.py uses above Rnn_encoder_decoder neural model to train my language dataset. 
folder vocab contains vocabulary
so train.py and rnn_encoder_decoder are important files.

language_translation_de/ 
data_no.py To calulate what percentage for words are covered by Vocabulary 
data_rnn_classify.py To randomly shuffle the wrong sentences and correct sentences , that will be used to classying sentences
check_data.py to check if the data prepared is correct or not
train.py Wrapper script to train an RNN-encoder-decoder model for phrase translation probabilities
euclidean.py To calulate the closeness fo two word embedding using euclidean distnace
data_lstm_check.py To remove sentences which are too short and combine X and Y labels for lstm'
sort_data.py To prepare data by sorting sentences according to no of words, it will improve the training time
rnn_e_d_test_check.py RNN-encoder-decoder model used for testing learned model
phasepair.py To prepare dataset pairwise for English and its corresponding German Sentence
Other code files are used for making training and test dataset , shuffling them , checking is the dataset formed is correct or not and to see if model works correctly or not. data_rnn_classify.py is used to make a training and test data set for wrong correct(w/c) approach lstm_model_check.py is used for checking the final result of prediction for w/c approac



For both languages one ngram_model is made inorder to learn the Emdedding Vector for words in dictionary , dictionary contain top 10,000 frequent words of corpus , for ngram_model i used a RNN neural network to predict the n+1 th word given n previous words.And these embedding vectors will be used by rnn_encoder_decoder model.

-Similary language_translation_de does the samething for German dataset. 3)language_translation is used to translate a English sentence to German , using above models only.
