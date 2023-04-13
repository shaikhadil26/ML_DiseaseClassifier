import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel
import sys

cbot = ChatBot('Baymax')

# Training the chat bot using yml files
trainer = ChatterBotCorpusTrainer(cbot)

trainer.train(r'C:\Users\Adil\Desktop\ML_Proj\cb_training\conversations.yml',
              r'C:\Users\Adil\Desktop\ML_Proj\cb_training\greetings.yml',
              r'C:\Users\Adil\Desktop\ML_Proj\cb_training\medical.yml',              
            )

df = pd.read_csv(r'C:\Users\Adil\Desktop\ML_Proj\ML_db.csv')

# Label encoding the categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for x in df.columns:
  df[x] = label_encoder.fit_transform(df[x])

x = df.iloc[:, :37].values
y = df['prognosis'].values

# Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating models for diff classifiers
model_dt = DecisionTreeClassifier()
model_nb = GaussianNB()
model_knn = KNeighborsClassifier()

# Creating voting based ensemble classifier
ensemble = VotingClassifier(estimators=[('dt', model_dt), ('nb', model_nb), ('knn', model_knn)], voting='hard')
ensemble.fit(x_train, y_train)

# Gui for the chat screen
class ChatScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Baymax")
        self.setGeometry(100, 100, 400, 400)

        # create chat history text box
        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)

        # create message input box and send button
        self.message_input = QLineEdit(self)
        self.send_button = QPushButton("SEND", self)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addWidget(self.chat_history)
        layout.addLayout(input_layout)

        self.setLayout(layout)

        # connect send button to method
        self.send_button.clicked.connect(self.send_message)

    def send_message(self):
        message = self.message_input.text()

        if message:
            x = message.lower()

            if x == 'begin' or x == 'i am sick' or x == 'self diagnosis':
                self.chat_history.append(f'You-> {message}')
                self.message_input.setText("")

                ds.show()
                cs.hide()
                
                self.chat_history.append(f'Baymax-> Diagnosis Complete')
                self.message_input.setText("")

            elif x == "goodbye" or x == "quit" or x == "bye":
                self.chat_history.append(f'Baymax-> Thank You have a great day!')
                self.message_input.setText("")
                quit()

            else:
                bot_input = cbot.get_response(x)
                # print(bot_input)

                self.chat_history.append(f'You-> {message}')
                self.chat_history.append(f'Baymax-> {bot_input}')
                self.message_input.setText("") 
                   

symptoms = df.columns[:37]

# GUI for Diagnosis Screen
class DiagnosisScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Diagnosis Screen")
        self.setGeometry(100, 100, 400, 200)

        self.i = 0
        self.test_user = []
        
        self.label = QLabel(f"Do you have {symptoms[self.i]}?", self)

        self.yes_button = QPushButton("Yes", self)
        self.no_button = QPushButton("No", self)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.yes_button)
        button_layout.addWidget(self.no_button)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # connect buttons to method
        self.yes_button.clicked.connect(lambda: self.button_clicked("Yes"))
        self.no_button.clicked.connect(lambda: self.button_clicked("No"))

    def button_clicked(self, answer):
        if answer == "Yes":
            self.test_user.append(1)
        else:
            self.test_user.append(0)
        self.i += 1
        

        if self.i==len(symptoms):
            test = np.array(self.test_user).reshape(1, -1)
            y_pred = ensemble.predict(test)

            y_predL = label_encoder.inverse_transform(y_pred)

            self.label.setText(f"You most likely have {y_predL[0]}")
            # print(y_predL)

        elif self.i > len(symptoms):
            # print(test_user)
            self.i = 0
            self.test_user = []
            ds.hide()
            cs.show()
            self.label.setText(f"Do you have {symptoms[0]}?")
            
        elif self.i < len(symptoms):
            self.label.setText(f"Do you have {symptoms[self.i]}?")


app = QApplication(sys.argv)
cs = ChatScreen()
ds = DiagnosisScreen()
cs.show()
sys.exit(app.exec_())


