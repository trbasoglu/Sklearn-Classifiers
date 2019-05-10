from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier,\
    VotingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
import argparse

# ------Creating numpy files to gain time. No need to read and traverse these files.--------


def create_numpy_files(folder):
    try:
        f = open("dict.txt", "r")  # dict.txt dosya konumlarını ve dosyaların sınıflarını içerir.
        file_dict = eval(f.read())
        f.close()
    except IOError:
        file_dict = dict()
        '''
        Dosya formatı sınıf isminde klasörler ve içinde dosyalar şeklinde olmalıdır.
        '''
        for root, dirs, files in os.walk(folder):  # ilgili klasörünün içindeki dosyaları dolaşmak için
            if len(os.path.basename(root)) > 1:
                continue
            root_int = ord(os.path.basename(root))
            print(chr(root_int))
            for filename in files:  # Klasor içindeki bütün dosyaları dolaşır.
                file_path = os.path.join(root, filename)
                file_dict[file_path] = chr(root_int)
                print(file_path, ':', root_int)
        f = open("dict.txt", "w")
        f.write(str(file_dict))
        f.close()

    print(file_dict)
    all_features = []
    image_labels = []

    for _key in file_dict.keys():  # key=path
        image_labels.append(file_dict[_key])
        print(_key)
        _image = cv2.imread(_key, cv2.IMREAD_GRAYSCALE)
        all_features.append(_image)
    #  resimleri ve etiketleri 'ndarray' olarak kaydedilir.
    np.save('labels.npy', image_labels)
    np.save('images.npy', all_features)
    print('Files are created.')
    return image_labels, all_features


# ------- Dosyaların sınıflara göre dağılımını gösterir.----------


def plot_bar(y, loc='left', relative=True):
    plt.suptitle('relative amount of photos per type')
    width = 0.35
    if loc == 'left':
        n = -0.5
    else:
        n = 0.5

    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]

    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'

    xtemp = np.arange(len(unique))

    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)

#  ----------------Main function--------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required = True, help="path of training folder")
    args = vars(ap.parse_args())
    try:  # dosya varsa okur,yoksa oluşturur.
        labels = np.load('labels.npy')
        images = np.load('images.npy')
    except IOError:
        print('DOSYA YOK')
        labels, images = create_numpy_files(args["folder"])

    print(labels)
    # Veri kümesini  test ve train olmak üzere ikiye böler. test_size=0.2 ifadesi
    # verinin %20si test verisi olacağını belirtir
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    plot_bar(y_train, loc='left')
    plot_bar(y_test, loc='right')
    plt.legend([
        'train ({0} photos)'.format(len(y_train)),
        'test ({0} photos)'.format(len(y_test))
    ])
    plt.show()

    # Clasifier dictionary, optimal değerleri bulmaya çalıştım.
    trainers_dictionary = {
        'Svm Poly': svm.SVC(gamma=0.001, C=100, kernel='poly'),
        'Random Forest': RandomForestClassifier(max_depth=15, n_estimators=10, max_features=100),
        'Decison Tree': DecisionTreeClassifier(max_depth=15),
        'AdaBoost': AdaBoostClassifier(learning_rate=0.025),
        'KNN': KNeighborsClassifier(3),
        'Neural Network': MLPClassifier(alpha=1),
        'Gausian Naive Bayes': GaussianNB(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Bernouli Naive Bayes': BernoulliNB(fit_prior=False),
        'complement Naive Bayes': ComplementNB(),
        'ExtraTreeClassifier()': ExtraTreeClassifier(),
        'Bagging Classifier': BaggingClassifier(),
        'Gradient Boosting Clasifier': GradientBoostingClassifier(),
        'Voting Clasifiers': VotingClassifier(estimators=[('rf',
                                                           RandomForestClassifier(n_estimators=50, random_state=1)),
                                                          ('gnb', GaussianNB())], voting='hard')
    }

    x_train = x_train.reshape(len(x_train), -1)
    y_train = y_train.reshape(len(y_train), -1)
    y_train = np.ravel(y_train)
    x_test = x_test.reshape(len(x_test), -1)
    y_test = y_test.reshape(len(y_test), -1)
    y_test = np.ravel(y_test)

    # verilen isimde eğitilmiş model varsa onu yükler ve kullanır.
    try:
        clf = load('model.joplib')
    except IOError:
        clf = svm.SVC(gamma=0.001, C=100, kernel='poly')
        clf.fit(x_train, y_train)
        dump(clf, 'model.joplib')

    #  ----- Accuracy hesaplama-------
    y_predicted = clf.predict(x_test)
    i = 0
    t = 0
    for y_p in y_predicted:
        # print(y_p, '|', y_test[i])
        if y_p == y_test[i]:
            t += 1
        plt.show()
        i += 1
    print('SVM polynomial Accuracy:'+str((t*100)/i))
    # -----------Herbir classifier için accuracy hesaplanır.------------
    print('------Accuracy Results----------')
    for key in trainers_dictionary.keys():
        clf = trainers_dictionary[key]
        clf.fit(x_train, y_train)
        y_predicted = clf.predict(x_test)
        i = 0
        t = 0
        for y_p in y_predicted:
            if y_p == y_test[i]:
                t += 1
            plt.show()
            i += 1
        print(str(key)+' Accuracy : ' + str((t * 100) / i))
        dump(clf, key+'.joplib')
    print('-----***-----')


if __name__ == '__main__':
    main()
