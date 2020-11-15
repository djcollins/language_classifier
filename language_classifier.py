import csv


def read_data():
    raw_text_lines=[]
    counter=0
    all_text_lines=[]
    sentences=[]
    labels=[]
    with open("lang_data.csv", encoding="mbcs") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                if(len(row[0])>1):
                    sentences.append(row[0])
                    labels.append(row[1])
    print(len(sentences), len(labels))
    return sentences, labels






def main():
    read_data()


main()