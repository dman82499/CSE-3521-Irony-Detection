


#CSE 3521 Project: Logistic Regression comparison



#now we have all the necessary parameters for prediction


#starting actual code
import numpy

import matplotlib.pyplot

#I realize that the proper pythonic way is 'import numpy as np
#and import matplotlib.pyplot as plt, but I just prefer to call
#functions by their actual names


#first, we make a small little template dictionary class to store all unique words with a counter to them,
# basically like Collections.counter
#the counter will serve as the X feature vector for a given sentence
class dict_num_count(dict):

    def append(self, word, generate_more_features = True):
        if word not in self.keys():
            if generate_more_features:
                self[word] = 1
        else:

            self[word] += 1

    def reset(self):
        for x in self.keys():
            self[x] = 0



def create_word_feature_vector():

    #making the universal word dictionary, or the actual feature vector



    task_a_train_file = open("SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskA_emoji.txt", 'r', encoding='utf8')
    task_a_train_lines = task_a_train_file.readlines()

    
    #doing some parsing here to figure out what goes where 

    total_sentences = len(task_a_train_lines)
    task_a_train_lines.remove(task_a_train_lines[0])
    #in this first line of code, we just initialize the bag of words vector
    th_dict = dict_num_count()
    th_answers = []
    #we create a vector of the answers as well, with
    #1 as positive, 0 as neutral, and -1 as negative

    



    for words2 in task_a_train_lines:
        wordl2 = words2.split()
        th_answers.append(int(wordl2[1]))
        wordl2.remove(wordl2[1])
        for w in wordl2:
            th_dict.append(w)






    th_dict.reset()


    return task_a_train_lines, th_dict, th_answers

def initialize_weight_vectors(template:dict, num_len):
    #making a contigous array just in case it could possible speed up update times
    unique_words = template.values()
    total_weight_array = numpy.ascontiguousarray(numpy.ones((num_len, len(unique_words))))
    #weights initialized to .0005, value just taken from inpsiration of the 3/25 in class
    total_weight_array *=  .0005
    return total_weight_array
    
    

def calculate_feature_vector(th_dict, sentence, generate_features = True):
    #calculates for only one sentence, a feature vector X_i, where i is the training sample number
    words1 = sentence.split()
    for w1 in words1:
        th_dict.append(w1, generate_features)

    value_copy = list(th_dict.values())
    th_dict.reset()
    return value_copy


'''A function for splitting up the correct y vector into different classes'''
def answer_split(ans):
    #theoretically this is a bit slow since these could be initialized with numpy and in the same for loop
    #as the origenal function, but this way makes me feel more confident in the approach
    positive_ans = []
    negative_ans = []
    for x in ans:
        if x == 1:
            positive_ans.append(1)
        else:
            positive_ans.append(0)
    for x2 in ans:
        if x2 == 0:
            negative_ans.append(1)
        else:
            negative_ans.append(0)

    return numpy.array(positive_ans), numpy.array(negative_ans)





def make_LR_for_sentences(unique_word_dict, sentences, weight_vectors, gen_information = True):
    #first we make our x_vector
    #the entire X vector is a NxM matrix where N are the p arameters (bag of words)
    #and M contains each sentence
    X = []
    for x in sentences:
        x_i = calculate_feature_vector(unique_word_dict, x, gen_information)
        X.append(x_i)
    X = numpy.array(X)


    positive_weight_vector = weight_vectors[0]
    positive_z = numpy.dot(X, positive_weight_vector)
    positive_z = 1/(1+numpy.exp(-(positive_z)))



    negative_weight_vector= weight_vectors[1]
    negative_z = numpy.dot(X, negative_weight_vector)
    negative_z = 1 / (1 + numpy.exp(-(negative_z)))

    return X, positive_z, negative_z





def train(template:dict_num_count, data_set, data_set_ans, weight_array, learning_rate, epoaches=10):
    data_set_m = len(data_set)
    y_pos_ans, y_negative_ans = answer_split(data_set_ans)
    #setting up the loss arrays for each of the catagories
    positive_loss = []
    negative_loss = []
    #data_set should be a list of sentences for each of the data sets, followed by their class (positive, etc)
    for x in range(0, epoaches):
        print("epoch: " + str(x))
        X, P_y1, P_y2 = make_LR_for_sentences(template, data_set, weight_array)

 
        y_pred = numpy.array([P_y2, P_y1])
        y_pred = numpy.argmax(y_pred, axis=0)

        #just subtracting to match the range (argmax outputs from 0, 1, 2, but answers come in -1, 0, 1 due to
        #the implementation choice
        predict_catagory(data_set_ans, y_pred)


        avg_pos_grad = numpy.dot(X.transpose(), (P_y1 - y_pos_ans)) / data_set_m
        avg_negative_grad1 = numpy.dot(X.transpose(), (P_y2 - y_negative_ans)) / data_set_m
        avg_gradients = numpy.array([avg_pos_grad, avg_negative_grad1])
        avg_gradients *= learning_rate
        weight_array -= avg_gradients


    return weight_array, positive_loss, negative_loss


def save_loss(y1, y, updated_loss: list):
    #using the same loss function as in the 4/5 class
    len_of_dataset = len(y)
    loss = -(1/len_of_dataset)*numpy.sum(y*numpy.log(y1) + (1-y)*numpy.log(1-y1))
    updated_loss.append(loss)
    print("loss: ", str(loss))



def make_graph(loss, type):
    if type == 1:
        str_var = "positive"
    elif type == 0:
        str_var = "neutral"
    else:
        str_var = "negative"
    #the loss is a loss array
    # Data for plotting
    e = numpy.arange(len(loss))
    f = loss
    e += 1
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(e, f)

    ax.set(xlabel='epoch (e)', ylabel='loss',
           title='loss over epoch graph for logistic regression of class ' + str_var)
    ax.grid()

    fig.savefig("test_" + str_var + ".png")
    matplotlib.pyplot.show()

def predict_catagory(y, y_pre):
    #this will process corect and incorrect answers
    #y_pre is a vector from the function to predict the guess, by taking arg_max
    y = numpy.array(y, dtype=numpy.int)
    correct = numpy.equal(y, y_pre)
    accuracy = numpy.sum(correct)/len(y)
    print("accuracy:" + str(accuracy))


def predict_test_category(feature_vectors, weight_values):

    #first, we get all the test data and load them into variables
    

    test_data = open("SemEval2018-Task3/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt", encoding="utf8")

    all_sentences = []
    task_a_test_lines = test_data.readlines()
    total_sentences = len(task_a_test_lines)
    task_a_test_lines.remove(task_a_test_lines[0])
    # in this first line of code, we just initialize the bag of words vector
    th_dict = dict_num_count()
    th_answers = []
    # we create a vector of the answers as well, with
    # 1 as positive, 0 as neutral, and -1 as negative

    for words2 in task_a_test_lines:
        wordl2 = words2.split()
        th_answers.append(int(wordl2[1]))
        wordl2.remove(wordl2[1])
        for w in wordl2:
            th_dict.append(w)


        # just making an array of the numbers to make the returning parameters easier
    th_dict.reset()

    X, P_y1, P_y2 = make_LR_for_sentences(feature_vectors, task_a_test_lines, weight_values, False)
    y_pred = numpy.array([P_y2, P_y1])
    y_pred = numpy.argmax(y_pred, axis=0)
    print("Test Data:\n")
    predict_catagory(th_answers, y_pred)





#here is the main code entry point 


#task a regression
sentences, feature_vectors, answers = create_word_feature_vector()
epoachs = 5
weight_vectors = initialize_weight_vectors(feature_vectors, 2)
completed_weight_vectors, positive_loss, negative_loss = train(feature_vectors, sentences, answers, weight_vectors, .1, epoachs)
predict_test_category(feature_vectors, completed_weight_vectors)

#task_b feature vector





