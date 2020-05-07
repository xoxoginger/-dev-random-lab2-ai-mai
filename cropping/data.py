import cv2


def cropping(image, x0, x1, cr_name):
    y0 = 0
    y1 = 28

    cr_width = 28  # р-ры вырезания
    cr_height = 28

    cnt_w = 0
    cnt_h = 0

    while cnt_h < 20:  # кол-во по вертикали
        cnt_w = 0
        if x0 == 3:
            x0 = 3  # для 1
            x1 = 31
        else:
            x0 = 0 # 2
            x1 = 28
        while cnt_w < 10:  # кол-во по горизонтали
            cr_im = image[y0:y1, x0:x1]
            filename = '{0}.png'.format(cr_name)
            cv2.imwrite(filename, cr_im)
            cv2.imshow("Cropped image", cr_im)  # чтобы отсмотреть полученное
            cv2.waitKey(0)
            cnt_w += 1
            x0 += cr_width * 2
            x1 += cr_width * 2
            cr_name += 1
        cnt_h += 1
        y0 += cr_height * 2
        y1 += cr_height * 2
        if 7 < cnt_h < 11:  # чтобы лучше вырезалось ^^ (угол фото)
            y0 -= 2
            y1 -= 2
        if cnt_h > 10:
            x0 += 6
            x1 += 6


image = cv2.imread("1_final.jpg") # загружаем изображение и отображаем его
x0 = 3  # начальная точка 1
x1 = 31  # конечная точка вырезания 1
cr_name = 100
cropping(image, x0, x1, cr_name)

image = cv2.imread("2_final.jpg")
x0 = 0 # нач точка 2
x1 = 28 # 2
cr_name += 200
cropping(image, x0, x1, cr_name)

image = cv2.imread("3_final.jpg")
cr_name += 200
cropping(image, x0, x1, cr_name)
