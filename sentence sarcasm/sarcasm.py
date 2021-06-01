import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import json
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

srcsm_json = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)

sentences = srcsm_json['headline']
labels = srcsm_json['is_sarcastic']

training_size = round(len(sentences) * .75)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


# Setting tokenizer properties
vocab_size = 10000
oov_tok = "<oov>"

# Fit the tokenizer on Training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Setting the padding properties
max_length = 100
trunc_type='post'
padding_type='post'

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)


# Training the model
num_epochs = 50

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels)
                    , verbose=2)


def toplevel1(top):
    '''This class configures and populates the toplevel window.
        top is the toplevel containing window.'''
    _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
    _fgcolor = '#000000'  # X11 color: 'black'
    _compcolor = '#d9d9d9' # X11 color: 'gray85'
    _ana1color = '#d9d9d9' # X11 color: 'gray85'
    _ana2color = '#ececec' # Closest X11 color: 'gray92'
    font11 = "-family {Times New Roman} -size 35 -weight bold"
    font12 = "-family {Times New Roman} -size 25 -weight bold"
    font9 = "-family {Times New Roman} -size 22 -weight bold"

    x, y = top.winfo_screenwidth(), top.winfo_screenheight()
    top.geometry("%dx%d+0+0" % (x, y))
    top.minsize(148, 1)
    top.maxsize(1924, 1055)
    top.resizable(1, 1)
    top.title("New Toplevel")
    top.configure(background="#000066")
    top.configure(highlightbackground="#f0f0f0f0f0f0")
    top.configure(highlightcolor="#646464646464")
    top.protocol('WM_DELETE_WINDOW', sam)

    global sentence, sarcastic
    sentence = tk.StringVar()
    sarcastic = tk.StringVar()


    Label1 = tk.Label(top)
    Label1.place(relx=0.239, rely=0.048, height=87, width=808)
    Label1.configure(background="#000066")
    Label1.configure(disabledforeground="#a3a3a3")
    Label1.configure(font=font11)
    Label1.configure(foreground="white")
    Label1.configure(highlightcolor="#000000")
    Label1.configure(text='''Identify Sarcasm in a Sentence''')

    Label2 = tk.Label(top)
    Label2.place(relx=0.3, rely=0.191, height=67, width=610)
    Label2.configure(background="#000066")
    Label2.configure(disabledforeground="#a3a3a3")
    Label2.configure(font=font12)
    Label2.configure(foreground="white")
    Label2.configure(text='''Enter Sentence you want to Check Sarcasm''')

    Entry1 = tk.Entry(top, textvariable=sentence)
    Entry1.place(relx=0.104, rely=0.298,height=104, relwidth=0.795)
    Entry1.configure(background="white")
    Entry1.configure(disabledforeground="#a3a3a3")
    Entry1.configure(font=font12)
    Entry1.configure(foreground="#000000")
    Entry1.configure(insertbackground="black")

    Label3 = tk.Label(top)
    Label3.place(relx=0.32, rely=0.656, height=88, width=578)
    Label3.configure(background="#000066")
    Label3.configure(disabledforeground="#a3a3a3")
    Label3.configure(font=font12)
    Label3.configure(foreground="white")
    Label3.configure(text='''SARCASTIC STATUS''')

    Entry2 = tk.Entry(top, textvariable=sarcastic, state='readonly')
    Entry2.place(relx=0.148, rely=0.776,height=104, relwidth=0.725)
    Entry2.configure(background="white")
    Entry2.configure(disabledforeground="#a3a3a3")
    Entry2.configure(font=font12)
    Entry2.configure(foreground="#000000")
    Entry2.configure(insertbackground="black")

    Button1 = tk.Button(top, command=check)
    Button1.place(relx=0.54, rely=0.477, height=103, width=360)
    Button1.configure(activebackground="#efdfff")
    Button1.configure(activeforeground="#000000")
    Button1.configure(background="#8080ff")
    Button1.configure(disabledforeground="#a3a3a3")
    Button1.configure(font=font9)
    Button1.configure(foreground="#000000")
    Button1.configure(highlightbackground="#d9d9d9")
    Button1.configure(highlightcolor="black")
    Button1.configure(pady="0")
    Button1.configure(text='''IS  SARCASTIC''')

    Button2 = tk.Button(top, command=sam)
    Button2.place(relx=0.24, rely=0.477, height=103, width=360)
    Button2.configure(activebackground="#efdfff")
    Button2.configure(activeforeground="#000000")
    Button2.configure(background="#8080ff")
    Button2.configure(disabledforeground="#a3a3a3")
    Button2.configure(font=font9)
    Button2.configure(foreground="#000000")
    Button2.configure(highlightbackground="#d9d9d9")
    Button2.configure(highlightcolor="black")
    Button2.configure(pady="0")
    Button2.configure(text='''EXIT''')

def sam():
    root.destroy()

def check():
    sentences = []
    sentences.append(sentence.get()) 
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    if(model.predict(padded)[0][0]>=0.3):
        sarcastic.set("Given Sentence is Sarcastic")
    else:
        sarcastic.set("Given Sentence is not Sarcastic")


if __name__ == '__main__':
    sentence, sarcastic = None, None
    root = tk.Tk()
    toplevel1(root)
    root.mainloop()





