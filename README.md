**Name:** MD SAMEER ANSARI

**Company:** CODTECH IT SOLUTIONS

**ID:** CT12WDS94

**Domain:** Python Programming

**Duration:** December to March 2024  

# CODTECH - TASK 3 - AI-chatbot
# Introduction
In today’s digital age, where communication is increasingly driven by artificial intelligence (AI) technologies, building your own chatbot has never been more accessible. With the rise of platforms like ChatGPT from OpenAI and powerful libraries such as NLTK (Natural Language Toolkit) in Python, creating a basic Python chatbot has become a straightforward endeavor for aspiring data scientists and developers.


# What are Chatbots?
Chatbots are AI-powered software applications designed to simulate human-like conversations with users through text or speech interfaces. They leverage natural language processing (NLP) and machine learning algorithms to understand and respond to user queries or commands in a conversational manner.

Chatbots can be deployed across various platforms, including websites, messaging apps, and virtual assistants, to provide a wide range of services such as customer support, information retrieval, task automation, and entertainment. They play a crucial role in improving efficiency, enhancing user experience, and scaling customer service operations for businesses across different industries.

# Why Do We Need Chatbots?
Enhanced Customer Service: Chatbots provide instant responses to customer queries, ensuring round-the-clock support without the need for human intervention. This results in faster resolution times and improved customer satisfaction.
Scalability: With chatbots, businesses can handle multiple customer interactions simultaneously, scaling their support operations to accommodate growing demand without significantly increasing costs.
Cost Efficiency: Implementing chatbots reduces the need for hiring and training additional customer service representatives, resulting in cost savings for businesses over time.
24/7 Availability: Chatbots operate continuously, offering support to users regardless of the time of day or geographical location. This ensures that customers can receive assistance whenever they need it, leading to higher engagement and retention rates.
Data Collection and Analysis: Chatbots can gather valuable customer data during interactions, such as preferences, frequently asked questions, and pain points. This data can be analyzed to identify trends, improve products or services, and tailor marketing strategies, driving business growth and innovation.
Build a simple Chatbot using NLTK Library in Python
# Types of Chatbots
There are mainly 2 types of AI chatbots.

1) Rule-based Chatbots: As the name suggests, there are certain rules by which chatbot operates. Like a machine learning model, we train the chatbots on user intents and relevant responses, and based on these intents chatbot identifies the new user’s intent and response to him.

2) Self-learning chatbots: Self-learning bots are highly efficient because they are capable to grab and identify the user’s intent on their own. they are built using advanced tools and techniques of machine learning, deep learning, and NLP.

Self-learning bots are further divided into 2 subcategories.

Retrieval-based chatbots: Retrieval-based is somewhat the same as rule-based where predefined input patterns and responses are embedded.
Generative-based chatbots: It is based on the same phenomenon as Machine Translation built using sequence 2 sequences neural network.
Most organizations use self-learning chatbots along with embedding some rules like the hybrid version of both methods which makes chatbots powerful enough to handle each situation during a conversation with a customer.

# How does Chatbot Works?
Chatbots are computer programs that simulate conversation with humans. They’re used in a variety of applications, from providing customer service to answering questions on a website.

# Here’s a general breakdown of how a chatbot works:

User Input: The user starts a conversation with the chatbot by typing in a message or speaking to it through a voice interface.
Understanding the User: The chatbot analyzes the user’s input using NLP. For rule-based chatbots, this involves matching keywords and phrases. For AI-powered chatbots, it’s more complex and involves understanding the intent behind the user’s words.
Generating a Response: Based on its understanding of the user’s input, the chatbot retrieves a response from its database. This response could be a simple answer, a more complex explanation, or even a question to clarify the user’s intent.
Conversation Flow: The chatbot delivers the response to the user, and the conversation continues. The user can provide additional information or ask follow-up questions, and the chatbot will respond accordingly.
Building A Chatbot Using Python
Now we have an immense understanding of the theory of chatbots and their advancement in the future. Let’s make our hands dirty by building one simple rule-based chatbot using Python for ourselves.


# let’s start building logic for the NLTK chatbot.

After importing the libraries, First, we have to create rules. The lines of code given below create a simple set of rules. the first line describes the user input which we have taken as raw string input and the next line is our chatbot response.

We have created an amazing Rule-based chatbot just by using Python and NLTK library. The nltk.chat works on various regex patterns present in user Intent and corresponding to it, presents the output to a user. Let’s run the application and chat with your created chatbot

![Screenshot 2025-01-11 162135](https://github.com/user-attachments/assets/0c499307-5eb2-42f9-977a-51219fc0f2ed)



Step 1: Install Required Modules
Begin by installing the necessary modules using the pip command:

pip install tensorflow keras pickle nltk
Step 2: Import and Load Data File
Import the required packages and load the data file (`intents.json` in this case) containing intents for the chatbot.

import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import json

import pickle

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import SGD

import random

# Load data from intents.json

data_file = open('intents.json').read()

intents = json.loads(data_file)
Step 3: Preprocess Data
The “preprocess data” step involves tokenizing, lemmatizing, removing stop words, and removing duplicate words to prepare the text data for further analysis or modeling.

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Sample data
data = [
    "The quick brown fox jumps over the lazy dog",
    "A bird in the hand is worth two in the bush",
    "Actions speak louder than words"
]

# Tokenize, lemmatize, and remove stopwords
tokenized_data = []
for sentence in data:
    tokens = nltk.word_tokenize(sentence.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]  # Remove stop words
    tokenized_data.append(filtered_tokens)

# Remove duplicate words
for i in range(len(tokenized_data)):
    tokenized_data[i] = list(set(tokenized_data[i]))

print(tokenized_data)
Step 4: Create Training and Testing Data
Prepare the training data by converting text into numerical form.

# Create training data

training = []

output_empty = [0] * len(classes)

for doc in documents:

    bag = []

    pattern_words = doc[0]

    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    

    for w in words:

        bag.append(1) if w in pattern_words else bag.append(0)

    

    output_row = list(output_empty)

    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert to numpy array

random.shuffle(training)

training = np.array(training)

# Separate features and labels

train_x = list(training[:,0])

train_y = list(training[:,1])
Step 5: Build the Model
Create a neural network model using Keras.

model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model

model.save('chatbot_model.h5')
Step 6: Predict the Response
Implement a function to predict responses based on user input.

def predict_class(sentence, model):

    p = bow(sentence, words, show_details=False)

    res = model.predict(np.array([p]))[0]

    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:

        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list
Step 7: Run the Chatbot
Develop a graphical user interface to interact with the chatbot.

# GUI with Tkinter

import tkinter

from tkinter import *

# Function to send message and get response

def send():

    msg = EntryBox.get("1.0",'end-1c').strip()

    EntryBox.delete("0.0",END)

    if msg != '':

        ChatLog.config(state=NORMAL)

        ChatLog.insert(END, "You: " + msg + '\n\n')

        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)

        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)

        ChatLog.yview(END)

# GUI setup

base = Tk()

base.title("Chatbot")

base.geometry("400x500")

base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")

ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")

ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff', command= send )

EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

scrollbar.place(x=376,y=6, height=386)

ChatLog.place(x=6,y=6, height=386, width=370)

EntryBox.place(x=128, y=401, height=90, width=265)

SendButton.place(x=6, y=401, height=90)

base.mainloop()
By following these steps and running the appropriate files, you can create a self-learning chatbot using the NLTK library in Python.


