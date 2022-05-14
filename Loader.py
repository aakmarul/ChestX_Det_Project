import numpy
import numpy as np
import PIL.Image


import pathlib
import json
from skimage.transform import resize

def load_test_data(test_data_path):
    """
    *Asagidaki konumdan dosyalar okundu ve her dosyanin isim
    *bilgisi im_filename listesinde tutuldu.
    """
    # test data path is the 'C:/Users/ataka/OneDrive/Masaüstü/test_data/test'
    data_dir = str(test_data_path)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*.PNG')))
    test_images = list(data_dir.glob('*.PNG')) # test klasörü içindeki görüntülerin dosya yolunu tutan liste
    test_labels = []                           #json dosyasından elde edilecek labeller için oluşturulan liste.

    """
    * .json file ile resimlerin isimleri ve multi-label olarak
    * sınıfları alınıp label_data değişkeni içinde liste veri tipinde 
    * tutuluyor.  
    """
    json_file = open(test_data_path + '/ChestX_Det_train.json')
    label_data = json.load(json_file)
    label_names = [] # it holds image names from the .json file
    labels = []      # it holds labels corresponding its name
    for v in range (0, image_count):
        label_names.append(label_data[v]['file_name'])
        labels.append(label_data[v]['syms'])
    """
    ** Bu for loopdan sonra label_data listesine ihtiyac kalmiyor cünkü icinde tutulan file_name ve syms 
    ** verileri ayrı listelere çekiliyor.
    """
    del label_data

    i = 0
    j = 0
    im_filename = [] #görüntülerin sırayla okunan isimleri. Bu  isim sırasına göre .jsondan ilgili labeller bulunacak.
    image_array = [] #first define as a list then convert it to the numpy array
    wrong_indexes = []
    while i < image_count:
        test_images[i] = str(test_images[i])
        im_filename.append(test_images[i].split("\\")[-1]) # her görüntünün dosya yolunun son kısımları alınarak görüntünün
                                                           # ismi string olarak listede tutuluyor.
        im = PIL.Image.open(str(test_images[i]))           # Dosya yolundaki görüntüleri aç.
        resized_image= resize(np.array(im), (512,512))     # (1024, 1024) olan görüntüleri (512, 512) ye düşür.
        if ((np.shape(resized_image) != (512, 512))):

            wrong_indexes.append(i)                       # Veri yapısı bozuk olan resimler verilerin içerisinden
                                                          # çıkartılacak.

        image_array.append(resized_image)                  # Görüntüleri image_array içerisinde liste şeklinde tut.

        indexOfJson = label_names.index(im_filename[i])   #sırayla okunan görüntülerin json dosyasındaki indexleri bulunur.
        test_labels.append(tuple(labels[indexOfJson]))    #Bulunan indexten multi-label bilgilerine ulaşılarak tuple veri
                                                          #tipinde liste içerisinde tutulur.
        i = i+1

    for v in range(0, len(wrong_indexes)):
        del image_array[wrong_indexes[v]-j]
        del test_labels[wrong_indexes[v]-j]
        j = j+1

    #print("Length of image array: ", len(image_array), "type of ", type(image_array[3]))
    #print("Length of test labels: ", len(test_labels), "type of ", type(test_labels[3]))
    #okunan görüntüler numpy array olarak kaydet.
    image_array = numpy.asarray(image_array)
    print("Type of test data: ", type(image_array))
    print("Shape of test data: ", np.shape(image_array))

    #tuple olarak tutulan labelları numpy array formatına çevir.
    #print(test_labels)

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    test_labels = mlb.fit_transform(test_labels)
    print("Type of test labels: ", type(test_labels))
    print("Shape of test labels: ", np.shape(test_labels))
    #print(test_labels)

    """
    
    im = PIL.Image.open(str(test_images[552]))
    image_array = np.array(im)
    print(image_array.shape)
    
    #im.show()
    #print(test_images)
    """
    return (image_array, test_labels)


def load_train_data(train_data_path):
    """
    *Asagidaki konumdan dosyalar okundu ve her dosyanin isim
    *bilgisi im_filename listesinde tutuldu.
    """
    # test data path is the 'C:/Users/ataka/OneDrive/Masaüstü/train_data/train'
    data_dir = str(train_data_path)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*.PNG')))
    test_images = list(data_dir.glob('*.PNG'))  # test klasörü içindeki görüntülerin dosya yolunu tutan liste
    test_labels = []  # json dosyasından elde edilecek labeller için oluşturulan liste.

    """
    * .json file ile resimlerin isimleri ve multi-label olarak
    * sınıfları alınıp label_data değişkeni içinde liste veri tipinde 
    * tutuluyor.  
    """
    json_file = open('C:/Users/ataka/OneDrive/Masaüstü/train_data/train/ChestX_Det_train.json')
    label_data = json.load(json_file)
    label_names = []  # it holds image names from the .json file
    labels = []  # it holds labels corresponding its name
    for v in range(0, image_count):
        label_names.append(label_data[v]['file_name'])
        labels.append(label_data[v]['syms'])
    """
    ** Bu for loopdan sonra label_data listesine ihtiyac kalmiyor cünkü icinde tutulan file_name ve syms 
    ** verileri ayrı listelere çekiliyor.
    """
    del label_data

    i = 0
    j = 0
    im_filename = []  # görüntülerin sırayla okunan isimleri. Bu  isim sırasına göre .jsondan ilgili labeller bulunacak.
    image_array = []  # first define as a list then convert it to the numpy array
    wrong_indexes = []
    while i < image_count:
        test_images[i] = str(test_images[i])
        im_filename.append(
            test_images[i].split("\\")[-1])  # her görüntünün dosya yolunun son kısımları alınarak görüntünün
        # ismi string olarak listede tutuluyor.
        im = PIL.Image.open(str(test_images[i]))  # Dosya yolundaki görüntüleri aç.
        resized_image = resize(np.array(im), (512, 512))  # (1024, 1024) olan görüntüleri (512, 512) ye düşür.
        if ((np.shape(resized_image) != (512, 512))):
            wrong_indexes.append(i)  # Veri yapısı bozuk olan resimler verilerin içerisinden
            # çıkartılacak.

        image_array.append(resized_image)  # Görüntüleri image_array içerisinde liste şeklinde tut.

        indexOfJson = label_names.index(
            im_filename[i])  # sırayla okunan görüntülerin json dosyasındaki indexleri bulunur.
        test_labels.append(tuple(labels[indexOfJson]))  # Bulunan indexten multi-label bilgilerine ulaşılarak tuple veri
        # tipinde liste içerisinde tutulur.
        i = i + 1

    for v in range(0, len(wrong_indexes)):
        del image_array[wrong_indexes[v] - j]
        del test_labels[wrong_indexes[v] - j]
        j = j + 1

    # print("Length of image array: ", len(image_array), "type of ", type(image_array[3]))
    # print("Length of test labels: ", len(test_labels), "type of ", type(test_labels[3]))
    # okunan görüntüler numpy array olarak kaydet.
    image_array = numpy.asarray(image_array)
    print("Type of test data: ", type(image_array))
    print("Shape of test data: ", np.shape(image_array))

    # tuple olarak tutulan labelları numpy array formatına çevir.
    # print(test_labels)

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    test_labels = mlb.fit_transform(test_labels)
    print("Type of test labels: ", type(test_labels))
    print("Shape of test labels: ", np.shape(test_labels))
    # print(test_labels)

    """

    im = PIL.Image.open(str(test_images[552]))
    image_array = np.array(im)
    print(image_array.shape)

    #im.show()
    #print(test_images)
    """
    return (image_array, test_labels)