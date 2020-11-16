import torch
import time
import csv

from models.gru_lang_class import GRULanguageClassifier

device = torch.device("cuda")
#Epoch took 39.84827542304993 seconds.
#device = torch.device("cpu")
# Epoch took 113.60103845596313 seconds with cpu

def clean_string(st):
    return "".join([x.lower() for x in st if x.isalpha() or x==" "])



def read_data():

    counter=0
    sentences=[]
    labels=[]
    training_percentage=0.2
    english_samples_size=round(0.7445730824891461*2764*training_percentage) #we use 20% of enlish sentences as a test sample
    afrikaans_samples_size=round(0.23118668596237338*2764*training_percentage) #we use 20% of afrikaans sentences as a test sample
    nederlands_samples_size=round(0.024240231548480463*2764*training_percentage) #we use 20% of dutch sentences as a test sample
    english_test_samples=[]
    afrikaans_test_samples=[]
    nederlands_test_samples=[]
    with open("lang_data.csv", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                if(len(row[0])>1):
                    cleaner_string=clean_string(row[0])
                    if(row[1]=="English" and len(english_test_samples)<english_samples_size):
                        english_test_samples.append(cleaner_string)
                    elif (row[1] == "Afrikaans" and len(afrikaans_test_samples) < afrikaans_samples_size):
                        afrikaans_test_samples.append(cleaner_string)
                    elif(row[1]=="Nederlands" and len(nederlands_test_samples)<nederlands_samples_size):
                        nederlands_test_samples.append(cleaner_string)
                    else:
                        sentences.append(cleaner_string)
                        labels.append(row[1])

    #print(len(sentences), len(labels))
    return (sentences, labels, english_test_samples,afrikaans_test_samples,nederlands_test_samples)

def get_num_characters(sentences):
    # print(pairs)
    unique_chars = set()
    for sentence in sentences:
        unique_chars.update(set(list(sentence)))
    #print("unique chars:", unique_chars)
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
    data=read_data()
    pairs=data[0],data[1]
    #print(len(data[2]),len(data[3]),len(data[4]))
    #test_sets=pairs[2],pairs[3]
    # [[sentence,label],[,]...
    count_class_occurrence(pairs[1])
    num_chars=get_num_characters(pairs[0])
    classifier_object=GRULanguageClassifier(num_chars).to(device)
    num_params = sum(p.numel() for p in classifier_object.parameters() if p.requires_grad)
    #print("num params:", num_params, "in", classifier_object)
    epoch=40
    classes={0:"English", 1:"Afrikaans", 2:"Nederlands", "English":0, "Afrikaans":1, "Nederlands":2}
    label_counts=count_class_occurrence(pairs[1])
    label_ratios=[x/sum(label_counts) for x in label_counts]
    print(label_ratios)

    loss_function= torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_object.parameters(),lr=0.0001) #low learning rate because each sentence does a backprop weight update
    for n in range(epoch):
        print(f"Epoch {n} beginning:...")
        epoch_start = time.time()
        correct_calls = 0
        incorrect_calls=0
        epoch_loss=0
        english_guesses=0
        for sentence, label in zip(pairs[0], pairs[1]):
            #print(sentence, label)
            optimizer.zero_grad()
            probabilities=classifier_object(sentence) #ask the classifier for the probability of the input belonging to each class
            predicted_id=torch.max(probabilities, dim=1)[1].item()#find the class it thinks that the input most likely belongs to
            #print(probabilities)
            label_id = torch.tensor([classes[label]]).to(device)
            loss=loss_function(probabilities, label_id)
            epoch_loss+=loss.item()
            loss/= label_ratios[classes[label]] #divide the loss by the labels occurence ratio, this is a way to overcome the dataset class imbalance
            loss.backward() #this does backprop wrt loss for each trainable parameter and embeddings
            optimizer.step() #updates weights according to their fault and learning rate
            if(predicted_id==classes[label]):
                correct_calls+=1
            else:
                incorrect_calls+=1
        #print(f"English guesses {english_guesses}")
        english_guesses=0
        print(f"epoch {n} with total loss {epoch_loss}")

        print(f"Correct calls {correct_calls} Incorrect calls {incorrect_calls} ratio correct: {correct_calls/(correct_calls+incorrect_calls)}")
        if(correct_calls/(correct_calls+incorrect_calls)) >0.85:
            pass
            #torch.save(classifier_object, "my_model")
        print(f"Epoch took {time.time()-epoch_start} seconds.")
        #test_model(classifier_object)
    #torch.save(classifier_object, "my_model")


def test_model(model):
    classes = {0: "English", 1: "Afrikaans", 2: "Nederlands", "English": 0, "Afrikaans": 1, "Nederlands": 2}
    data=read_data()
    english_samples, afrikaans_samples, nederlands_samples = data[2], data[3], data[4]
    english_right, english_wrong, afrikaans_right,afrikaans_wrong, nederlands_right,nederlands_wrong=0,0,0,0,0,0
    label="English"
    for sentence in english_samples:
        probabilities = model(sentence)
        predicted_id = torch.max(probabilities, dim=1)[1].item()
        # print(probabilities)
        label_id = torch.tensor([classes[label]]).to(device)
        if (predicted_id == classes[label]):
            english_right += 1
        else:
            english_wrong += 1
    label="Afrikaans"
    for sentence in afrikaans_samples:
        probabilities = model(sentence)
        predicted_id = torch.max(probabilities, dim=1)[1].item()
        # print(probabilities)
        label_id = torch.tensor([classes[label]]).to(device)
        if (predicted_id == classes[label]):
            afrikaans_right += 1
        else:
            afrikaans_wrong += 1
    label="Nederlands"
    for sentence in nederlands_samples:
        probabilities = model(sentence)
        predicted_id = torch.max(probabilities, dim=1)[1].item()
        # print(probabilities)
        label_id = torch.tensor([classes[label]]).to(device)
        if (predicted_id == classes[label]):
            nederlands_right += 1
        else:
            nederlands_wrong += 1
    print(f"English Right {english_right} English Wrong {english_wrong} English Accuracy {english_right/(english_wrong+english_right)}")
    print(f"Afrikaans Right {afrikaans_right} Afrikaans Wrong {afrikaans_wrong} Afrikaans accuracy {afrikaans_right/(afrikaans_wrong+afrikaans_right)}")
    print(f"Nederlands Right {nederlands_right} Nederlands Wrong {nederlands_wrong} Nederlands accuracy {nederlands_right/(nederlands_right+nederlands_wrong)}")
    print(f"Total Accuracy {(english_right+afrikaans_right+nederlands_right)/(len(afrikaans_samples) + len(english_samples) + len(nederlands_samples))}")
    inp="w"
    while input!="q" and input!="Q":
        inp=clean_string(input("Please enter a sentence in English, Afrikaans, or Dutch, or 'q' to exit : "))
        if(inp=="q"):
            return
        probabilities = model(inp)
        predicted_id = torch.max(probabilities, dim=1)[1].item()
        print(classes[predicted_id])
def main():
    model=torch.load("my_model")
    test_model(model)
    #main_train_loop()
main()
#print(clean_string("Saam met ons, rondom ons, in ons en by ons is die onsienlike. Hy sal altyd daar wees."))