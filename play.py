import torch

import csv

from models.lstm_lang_class import LSTMLanguageClassifier

device = torch.device("cuda")
# device = torch.device("cpu")

def clean_string(st):
    return [x.lower() for x in st if x.isalpha() or x==" "]

def read_data():
    raw_text_lines=[]
    counter=0
    all_text_lines=[]
    sentences=[]
    labels=[]
    with open("lang_data.csv", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                if(len(row[0])>1):
                    cleaner_string=clean_string(row[0])
                    #for x in cleaner_string.split(" "):
                    sentences.append("".join(cleaner_string))
                    labels.append(row[1])

    print(len(sentences), len(labels))
    return (sentences, labels)

def get_num_characters(sentences):
    # print(pairs)
    unique_chars = set()
    for sentence in sentences:
        unique_chars.update(set(list(sentence)))
    print("unique chars:", unique_chars)
    num_chars = len(unique_chars)
    return num_chars

def count_class_occurrence(labels):
    label_names=["English", "Afrikaans", "Nederlands"]
    label_counts=[]
    for x in label_names:
        label_counts.append(sum([1 for y in labels if y==x]))
        #print(f"label {x} has {sum([1 for y in labels if y==x])} sentences")
    return label_counts
def main_train_loop():

    pairs=read_data() # [[sentence,label],[,]...
    count_class_occurrence(pairs[1])
    num_chars=get_num_characters(pairs[0])
    classifier_object=LSTMLanguageClassifier(num_chars).to(device)
    num_params = sum(p.numel() for p in classifier_object.parameters() if p.requires_grad)
    print("num params:", num_params, "in", classifier_object)
    epoch=20
    classes={0:"English", 1:"Afrikaans", 2:"Nederlands", "English":0, "Afrikaans":1, "Nederlands":1}
    label_counts=count_class_occurrence(pairs[1])
    label_ratios=[x/sum(label_counts) for x in label_counts]
    print(label_ratios)

    loss_function= torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier_object.parameters(),lr=0.01)
    for n in range(epoch):
        correct_calls = 0
        incorrect_calls=0
        epoch_loss=0
        english_guesses=0
        for sentence, label in zip(pairs[0], pairs[1]):
            #print(sentence, label)
            optimizer.zero_grad()
            probabilities=classifier_object(sentence)
            predicted_id=torch.max(probabilities, dim=1)[1].item()
            #print(probabilities)
            label_id = torch.tensor([classes[label]]).to(device)
            if(predicted_id==0):
                english_guesses+=1
            loss=loss_function(probabilities, label_id)
            epoch_loss+=loss.item()
            loss/= label_ratios[classes[label]]
            loss.backward()
            optimizer.step()
            if(predicted_id==classes[label]):
                correct_calls+=1
            else:
                incorrect_calls+=1
        print(f"English guesses {english_guesses}")
        english_guesses=0
        print(f"epoch {n} with total loss {epoch_loss}")
        print(f"Correct calls {correct_calls} Incorrect calls {incorrect_calls} ratio correct: {correct_calls/(correct_calls+incorrect_calls)}")
main_train_loop()