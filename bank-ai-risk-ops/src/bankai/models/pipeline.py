import os

os.system("python main.py generate-data")
os.system("python main.py train-fraud")
os.system("python main.py train-credit")
os.system("python main.py train-complaints")
os.system("python main.py score-fraud")
