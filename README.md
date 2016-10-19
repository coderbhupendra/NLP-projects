## Natural Language Processing related project done in IIT Madras. Programming language used: Python and theano framework

language_translation contains files required for preparing data set , vocabulary and training model:
===================================================================================================

1. **check_data.py** to check if the data prepared is correct or not.

2. **data_no.py** To calulate what percentage for words are covered by Vocabulary. 

3. **data_rnn_classify.py** to randomly shuffle the wrong sentences and correct sentences.

4. **preprocessdata.py** for preparing vocalbulary  of top 10,000 from prepared data set.

5. **rnn_encoder_decoder.py** RNN Enocoder Decoder model for encoding a sentence and decoding it into another language.

6. **lstm_model_check.py** lstm model to check model by printing original(English) and translated sentences(German).

language_translation_de contains following files:
=================================================

1. **data_no.py** To calulate what percentage for words are covered by Vocabulary.

2. **data_rnn_classify.py** To randomly shuffle the wrong sentences and correct sentences , that will be used to classying sentences.

3. **check_data.py** To check if the data prepared is correct or not.

4. **train.py** Wrapper script to train an RNN-encoder-decoder model for phrase translation probabilities.

5. **euclidean.py** To calulate the closeness fo two word embedding using euclidean distnace.

6. **data_lstm_check.py** To remove sentences which are too short and combine X and Y labels for lstm.

7. **sort_data.py** To prepare data by sorting sentences according to no of words, it will improve the training time.

8. **rnn_e_d_test_check.py** RNN-encoder-decoder model used for testing learned model.

9. **phasepair.py** To prepare dataset pairwise for English and its corresponding German Sentence.

language_translation_en contains following files: 
=================================================

1. **rnn_encoder_decoder.py** this contain the Rnn_encoder_decoder neural model to train the parallel corpus of English and German.

2. **train.py** uses above Rnn_encoder_decoder neural model to train my language dataset. 

3. folder vocab contains vocabulary.

4. train.py and rnn_encoder_decoder contains main code.

** Files with the same name in other folder have the same description.** 

How to procede
==============
Firstly, prepare datset for training and test, shuffling them.Then check if the dataset formed is correct or not and to see if model works correctly or not. 

wrong-correct(w/c) Experiment
=============================
In this we make syntax of sentnces wrong by shuffling words. Then we try to train model to learn to differentiate between wrong sentence vs. correct sentences.

First,run data_rnn_classify.py to make a training and test data set for wrong-correct(w/c) approach.

Then run lstm_model_check.py to checking the final result of prediction for w/c approach.

N-gram Experiment:
=================
In this we try to show how words with the same context meaning (like push and move) end up having similar word embedding.

Run train_ngram.py to train N-gram model and learn word embedding for vocabulary.

For both languages one ngram_model is made inorder to learn the Emdedding Vector for words in dictionary, dictionary contain top 10,000 frequent words of corpus.

**Note**:For ngram_model I used a RNN neural network to predict the n+1 th word given n previous words. And these embedding vectors will be used by rnn_encoder_decoder model.

Similary language_translation_de does the samething for German dataset. 

language_translation is used to translate a English sentence to German , using above models only.
