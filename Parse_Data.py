import numpy as np

_num_items = None

def parse():
    _step1()
    _step2()

def _step1():
    global _num_items
    count = 0
    filename = 'Data/yelp.rating'
    test_rating = open('Data/yelp.test.rating', 'w')
    train_rating = open('Data/yelp.train.rating', 'w')
    # test_negative = open('Data/yelp.negtive','w')
    with open(filename, 'r') as f:
        line = f.readline()
        arr = line.split("\t")
        maxline = line
        user = int(arr[0])
        max = int(arr[3])
        while 1:
            line = f.readline()
            if line == None or line == "":
                break
            arr = line.split("\t")
            u = int(arr[0])
            num = int(arr[3])
            item = int(arr[1])
            if count < item:
                count = item
            if u == user:
                if num > max:
                    print >> train_rating, maxline,
                    maxline = line
                    max = num
                else:
                    print >> train_rating, line,
            else:
                user = u
                print >> test_rating, maxline,
                maxline = line
        print >> test_rating, maxline,
    print count
    _num_items = count
    test_rating.close()
    train_rating.close()

def _step2():
    ratingList = load_rating_file_as_list('Data/yelp.test.rating')
    trainingList = load_training_file_as_list('Data/yelp.train.rating')
    test_negative = open('Data/yelp.test.negative', 'w')
    print _num_items
    for i in range(len(trainingList)):
        print >> test_negative, "(" + str(ratingList[i][0]) + "," + str(ratingList[i][1]) + ")",
        for j in range(100):
            item = np.random.randint(_num_items)
            while item in trainingList[j]:
                item = np.random.randint(_num_items)
            print >> test_negative, "\t" + str(item),
        print >> test_negative, "\n",
    test_negative.close()

def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList

def load_training_file_as_list(filename):
    # Get number of users and items
    u_ = 0
    lists, items = [], []
    with open(filename, "r") as f:
        line = f.readline()
        index = 0
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            if u_ < u:
                index = 0
                lists.append(items)
                items = []
                u_ += 1
            index += 1
            if index<300:
                items.append(i)
            line = f.readline()
    lists.append(items)
    print "already load the trainList..."
    return lists

if __name__ == "__main__":
    parse()