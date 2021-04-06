import os, random
import cv2, argparse
import numpy as np

entoko = {'ek':'다','sj':'너','an':'무','aj':'머','rk':'가','sh':'노','fj':'러','dn':'우','th':'소','tk':'사','qj':'버','fn':'루','dk':'아','en':'두','dh':'오','eh':'도',
'qk':'바','rh':'고','sk':'나','wn':'주','fh':'로','dj':'어','fk':'라','tn':'수','gj':'허','wk':'자','qn':'부','wh':'조','rj':'거','ah':'모','sn':'누','tj':'서','ak':'마','qh':'보','ej':'더','rn':'구','wj':'저'}

def image_augmentation(img, type2=False):
    # perspective
    w, h, c = img.shape
    pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])
    # 좌표의 이동점
    begin, end = 30, 90
    pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                       [random.randint(begin, end), w - random.randint(begin, end)],
                       [h - random.randint(begin, end), random.randint(begin, end)],
                       [h - random.randint(begin, end), w - random.randint(begin, end)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img = cv2.warpPerspective(img, M, (h, w))

    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0,4) * 2 + 1
    img = cv2.blur(img,(blur_value, blur_value))
    if type2:
        return img[130:280, 180:600, :]
    return img[130:280, 120:660, :]
    
    # Noise
    # img=np.clip((img/255+np.random.normal(scale=0.1, size=img.shape))*255,0,255).astype('uint8')


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate.jpg")
        self.plate2 = cv2.imread("plate_y.jpg")
        self.plate3 = cv2.imread("plate_g.jpg")

        # loading Number ====================  white-one-line  ==========================
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "./char1/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[0:-4])

        # loading Number ====================  yellow-two-line  ==========================
        file_path = "./num_y/"
        file_list = os.listdir(file_path)
        self.Number_y = list()
        self.number_list_y = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number_y.append(img)
            self.number_list_y.append(file[0:-4])

        # loading Char
        file_path = "./char1_y/"
        file_list = os.listdir(file_path)
        self.char_list_y = list()
        self.Char1_y = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1_y.append(img)
            self.char_list_y.append(file[0:-4])

        # loading Resion
        file_path = "./region_y/"
        file_list = os.listdir(file_path)
        self.Resion_y = list()
        self.resion_list_y = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Resion_y.append(img)
            self.resion_list_y.append(file[0:-4])
        #=========================================================================

        # loading Number ====================  green-two-line  ==========================
        file_path = "./num_g/"
        file_list = os.listdir(file_path)
        self.Number_g = list()
        self.number_list_g = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number_g.append(img)
            self.number_list_g.append(file[0:-4])

        # loading Char
        file_path = "./char1_g/"
        file_list = os.listdir(file_path)
        self.char_list_g = list()
        self.Char1_g = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1_g.append(img)
            self.char_list_g.append(file[0:-4])

        # loading Resion
        file_path = "./region_g/"
        file_list = os.listdir(file_path)
        self.Resion_g = list()
        self.resion_list_g = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Resion_g.append(img)
            self.resion_list_g.append(file[0:-4])
        #=========================================================================

    def Type_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        f_1 = open("/home/oms/sung/Korean-license-plate-Generator/text/type_1.txt", "w")
        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

            label = ""
            space = ""
            result = ""
            eng = ""
            # row -> y , col -> x
            row, col = 13, 35  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            eng += self.char_list[i%37]
            result+=entoko[eng]
            label += result
            space += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-110)/2), int((800-520)/2)
            background[s_width:110 + s_width, s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)
            

            if save:
                f_1.write(self.save_path + "T1" +space + ".jpg\t" + label+"\n")
                background = cv2.resize(background, (67,19))
                cv2.imwrite(self.save_path + "T1" + space +".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        f_1.close()

    def Type_2(self, num, save=False):
        number = [cv2.resize(number, (45, 83)) for number in self.Number]
        char = [cv2.resize(char1, (49, 70)) for char1 in self.Char1]
        f_2 = open("/home/oms/sung/Korean-license-plate-Generator/text/type_2.txt", "w")
        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (360, 160))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

            label = ""
            space = ""
            result = ""
            eng = ""
            row, col = 46, 10  # row + 83, col + 56

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 3
            eng += self.char_list[i%37]
            result+=entoko[eng]
            label += result
            space += self.char_list[i%37]
            Plate[row + 12:row + 82, col + 2:col + 49 + 2, :] = char[i%37]
            col += 49 + 2

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col + 2:col + 45 + 2, :] = number[rand_int]
            col += 45 + 2

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            s_width, s_height = int((400 - 160) / 2), int((800 - 360) / 2)
            background[s_width:160 + s_width, s_height:360 + s_height, :] = Plate
            background = image_augmentation(background, type2=True)

            if save:
                f_2.write(self.save_path + "T2" +space + ".jpg\t" + label+"\n")
                background = cv2.resize(background, (70,25))
                cv2.imwrite(self.save_path + "T2" + space + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        f_2.close()


    # def Type_3(self, num, save=False):
    #     number1 = [cv2.resize(number, (44, 60)) for number in self.Number_y]
    #     number2 = [cv2.resize(number, (64, 90)) for number in self.Number_y]
    #     resion = [cv2.resize(resion, (88, 60)) for resion in self.Resion_y]
    #     char = [cv2.resize(char1, (64, 62)) for char1 in self.Char1_y]

    #     for i, Iter in enumerate(range(num)):
    #         Plate = cv2.resize(self.plate2, (336, 170))
    #         b_width, b_height = 400, 800
    #         random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    #         background = np.zeros((b_width, b_height, 3), np.uint8)
    #         cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

    #         label = str()
    #         # row -> y , col -> x
    #         row, col = 8, 76

    #         # resion
    #         label += self.resion_list_y[i % 16]
    #         Plate[row:row + 60, col:col + 88, :] = resion[i % 16]
    #         col += 88 + 8

    #         # number 1
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_y[rand_int]
    #         Plate[row:row + 60, col:col + 44, :] = number1[rand_int]
    #         col += 44

    #         # number 2
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_y[rand_int]
    #         Plate[row:row + 60, col:col + 44, :] = number1[rand_int]

    #         row, col = 72, 8

    #         # character 3
    #         label += self.char_list_y[i % 37]
    #         Plate[row:row + 62, col:col + 64, :] = char[i % 37]
    #         col += 64

    #         # number 4
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_y[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
    #         col += 64

    #         # number 5
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_y[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
    #         col += 64

    #         # number 6
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_y[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
    #         col += 64

    #         # number 7
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_y[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]

    #         s_width, s_height = int((400 - 170) / 2), int((800 - 336) / 2)
    #         background[s_width:170 + s_width, s_height:336 + s_height, :] = Plate
    #         background = image_augmentation(background, type2=True)

    #         if save:
    #             cv2.imwrite(self.save_path + label + ".jpg", background)
    #         else:
    #             cv2.imshow(label, background)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    # def Type_4(self, num, save=False):
    #     number1 = [cv2.resize(number, (44, 60)) for number in self.Number_g]
    #     number2 = [cv2.resize(number, (64, 90)) for number in self.Number_g]
    #     resion = [cv2.resize(resion, (88, 60)) for resion in self.Resion_g]
    #     char = [cv2.resize(char1, (64, 62)) for char1 in self.Char1_g]

    #     for i, Iter in enumerate(range(num)):
    #         Plate = cv2.resize(self.plate3, (336, 170))
    #         b_width, b_height = 400, 800
    #         random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    #         background = np.zeros((b_width, b_height, 3), np.uint8)
    #         cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

    #         label = str()
    #         # row -> y , col -> x
    #         row, col = 8, 76

    #         # resion
    #         label += self.resion_list_g[i % 16]
    #         Plate[row:row + 60, col:col + 88, :] = resion[i % 16]
    #         col += 88 + 8

    #         # number 1
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_g[rand_int]
    #         Plate[row:row + 60, col:col + 44, :] = number1[rand_int]
    #         col += 44

    #         # number 2
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_g[rand_int]
    #         Plate[row:row + 60, col:col + 44, :] = number1[rand_int]

    #         row, col = 72, 8

    #         # character 3
    #         label += self.char_list_g[i % 37]
    #         Plate[row:row + 62, col:col + 64, :] = char[i % 37]
    #         col += 64

    #         # number 4
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_g[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
    #         col += 64

    #         # number 5
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_g[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
    #         col += 64

    #         # number 6
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_g[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
    #         col += 64

    #         # number 7
    #         rand_int = random.randint(0, 9)
    #         label += self.number_list_g[rand_int]
    #         Plate[row:row + 90, col:col + 64, :] = number2[rand_int]

    #         s_width, s_height = int((400 - 170) / 2), int((800 - 336) / 2)
    #         background[s_width:170 + s_width, s_height:336 + s_height, :] = Plate
    #         background = image_augmentation(background, type2=True)

    #         if save:
    #             cv2.imwrite(self.save_path + label + ".jpg", background)
    #         else:
    #             cv2.imshow(label, background)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    def Type_5(self, num, save=False):
        number1 = [cv2.resize(number, (60, 65)) for number in self.Number_g]
        number2 = [cv2.resize(number, (80, 90)) for number in self.Number_g]
        char = [cv2.resize(char1, (60, 65)) for char1 in self.Char1_g]
        f_5 = open("/home/oms/sung/Korean-license-plate-Generator/text/type_5.txt", "w")
        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate3, (336, 170))
            random_width, random_height =  400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((random_width, random_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (random_height, random_width), (random_R, random_G, random_B), -1)
            label = ""
            space = ""
            result = ""
            eng = ""

            # row -> y , col -> x
            row, col = 8, 78

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            space += self.number_list_g[rand_int]
            Plate[row:row + 65, col:col + 60, :] = number1[rand_int]
            col += 60

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            space += self.number_list_g[rand_int]
            Plate[row:row + 65, col:col + 60, :] = number1[rand_int]
            col += 60

            # character 3
            eng += self.char_list[i%37]
            result+=entoko[eng]
            label += result
            space += self.char_list[i%37]
            Plate[row:row + 65, col:col + 60, :] = char[i%37]
            row, col = 75, 8

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            space += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80


            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            space += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            space += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            space += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]

            s_width, s_height = int((400 - 170) / 2), int((800 - 336) / 2)
            background[s_width:170 + s_width, s_height:336 + s_height, :] = Plate

            background = image_augmentation(background, type2=True)

            if save:
                f_5.write(self.save_path+ "T3" + space + ".jpg\t" + label+"\n")
                background = cv2.resize(background, (70,25))
                cv2.imwrite(self.save_path + "T3" +space + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        f_5.close()


    def Type_new(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        f_0 = open("/home/oms/sung/Korean-license-plate-Generator/text/type_0.txt", "w")
        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

            label = ""
            space = ""
            result = ""
            eng = ""
            # row -> y , col -> x
            row, col = 13, 20  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 51

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 4
            label += self.char_list[i%37]
            space += self.char_list[i%37]
            space = space.upper()
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            space += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-110)/2), int((800-520)/2)
            background[s_width:110 + s_width, s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                background = cv2.resize(background, (68,19))
                f_0.write(self.save_path + "T0" + space + ".jpg"+"\t" + label+"\n")
                cv2.imwrite(self.save_path+ "T0" + space +".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        f_0.close()
        filenames = ['text/type_1.txt', 'text/type_2.txt','text/type_5.txt', 'text/type_0.txt']
        with open('/home/oms/sung/Korean-license-plate-Generator/text/result.txt', 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="/home/oms/sung/Korean-license-plate-Generator/output/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

A.Type_1(num_img, save=Save)
print("Type 1 finish")
A.Type_2(num_img, save=Save)
print("Type 2 finish")
# A.Type_3(num_img, save=Save)
# print("Type 3 finish")
# A.Type_4(num_img, save=Save)
# print("Type 4 finish")
A.Type_5(num_img, save=Save)
print("Type 3 finish")
A.Type_new(num_img, save=Save)
print("Type new finish")
