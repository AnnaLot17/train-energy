# Импорт библиотек
from math import log, pi, atan, exp, sin
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.colors as colors
import matplotlib.patches as ptch
from shapely.geometry import Polygon, LineString, Point

# цветоваяя схема графиков
#plt.style.use('seaborn-white')
cmap = 'YlOrRd'

'''
Ось х - ось вдоль рельс
Ось y - ось поперёк рельс
Ось z - ось вверх к проводам

'''

# РЕЖИМ РАБОТЫ СЕТИ

# первый путь
I1 = 300  # cуммарная сила тока, А
U1 = 30000  # cуммарное напряжение, В

# второй путь
I2 = 300  # cуммарная сила тока, А
U2 = 30000  # cуммарное напряжение, В

# третий путь
I3 = 300  # cуммарная сила тока, А
U3 = 30000  # cуммарное напряжение, В

# распределение тока по проводам
part_kp = 0.35
part_nt = 0.15
part_up = 0.50

part_kp = 0.35
part_nt = 0.15
part_up = 0.50
part_ep = -0.4  # минус так как ток запущен в обратном направлении

alpha = 5*pi/6  # угол между электрическим и магнитным поле

# СТАТИСТИЧЕСКИЕ ДАННЫЕ
x_chel = 0.9  # положение человека по оси х
y_chel = 0.9  # положение человека по оси y
floor = 2  # расстояние от земли до дна кабины
gr_floor = 1  # высота самого низа электровоза
z_chair = floor + 1.2  # сидушка стула
z_chel = floor + 1.5  # где находится человек по оси z
a = 1.75  # высота человека метры
b = 80  # масса человека килограммы
ti = 1  # длительность пребывания работника на рабочем месте, часов
z_graph = z_chel  # высота среза

# КОНСТАНТЫ

dis = 150  # дискретизация графиков (меньше - менее точно, но быстрее считает; больше - точнее, но дольше расчёт)
harm = {50: [1, 1],
        150: [0.3061, 0.400],
        250: [0.1469, 0.115],
        350: [0.0612, 0.050],
        450: [0.0429, 0.040],
        550: [0.0282, 0.036],
        650: [0.0196, 0.032],
        750: [0.0147, 0.022]}

# ДАННЫЕ О КОНТАКТНОЙ СЕТИ

xp = 0.760  # m - половина расстояния между рельсами
xp_kp = 0  # m - расстояние от центра между рельсами до КП (если левее центра - поставить минус)
xp_nt = 0  # m - расстояние от центра между рельсами до НТ (если левее центра - поставить минус)
xp_up = -3.7  # m - расстояние от центра между рельсами до УП
xp_ep = -2.7   # m - расстояние от центра между рельсами до ЭП
d_kp = 12.81 / 1000  # mm
d_nt = 12.5 / 1000  # mm
d_up = 17.5 / 1000  # mm
d_ep = 12.5 / 1000  # mm
h_kp = 6.0  # КП
h_nt = 7.8  # НТ
h_up = 8.0  # УП
h_ep = 8.4  # ЕП

# второй путь
xp_mid12 = 4.2  # расстояние между центрами перового и второго путей
xp_kp2 = 0  # m - расстояние от центра между рельсами до КП2 (если левее центра - поставить минус)
xp_nt2 = 0  # m - расстояние от центра между рельсами до НТ2 (если левее центра - поставить минус)
xp_up2 = 3.7  # m - расстояние от центра между рельсами до УП2 (если левее центра - поставить минус)
xp_ep2 = 2.7  # m - расстояние от центра между вторыми рельсами до ЭП2 (если левее центра - поставить минус)

# третий путь
# todo провода справа или слева?
xp_mid23 = 4.2  # расстояние между центрами второго и третьего путей
xp_mid13 = xp_mid12 + xp_mid23
xp_kp3 = 0  # m - расстояние от центра между рельсами до КП3 (если левее центра - поставить минус)
xp_nt3 = 0  # m - расстояние от центра между рельсами до НТ3 (если левее центра - поставить минус)
xp_up3 = 3.7  # m - расстояние от центра между рельсами до УП2
xp_ep3 = 2.7  # m - расстояние от центра между вторыми рельсами до ЭП2


# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
# min_x, max_x, min_y, max_y, min_z, max_z
bor = [0.2, 0.6, -1.2, 1.2, floor + 1.5, floor + 2.2]  # узлы окна
# min_x, max_x, min_z, max_z
sbor = [0.3, 1, floor + 1, floor + 2.2]  # узлы для бокового окна

# формируем передние окна методом Polygon: составляем список из координат точек по x, y, z каждого угла
frontWindleft = Polygon([(bor[0], bor[2], bor[4]),
                         (bor[1], bor[2], bor[5]),
                         (bor[1], -0.22, bor[5]),
                         (bor[0], -0.22, bor[4])])

frontWindright = Polygon([(bor[0], 0.22, bor[4]),
                          (bor[1], 0.22, bor[5]),
                          (bor[1], bor[3], bor[5]),
                          (bor[0], bor[3], bor[4])])

# расчёт границ теней боковых окон для кажого источника поля
min_nt = Point(0.5 * width, sbor[3]).distance(Point(xp_nt, h_nt))  # луч нижней границы тени от НТ
max_nt = Point(0.5 * width, sbor[2]).distance(Point(xp_nt, h_nt))  # луч верхней границы тени от НТ

min_kp = Point(0.5 * width, sbor[3]).distance(Point(xp_kp, h_kp))  # далее аналогично для остальных проводов
max_kp = Point(0.5 * width, sbor[2]).distance(Point(xp_kp, h_kp))

min_up_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_up, h_up))
max_up_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_up, h_up))
min_up_r = Point(0.5 * width, sbor[3]).distance(Point(xp_up, h_up))
max_up_r = Point(0.5 * width, sbor[2]).distance(Point(xp_up, h_up))

min_ep_l = Point(-0.5*width, sbor[3]).distance(Point(xp_ep, h_ep))
max_ep_l = Point(-0.5*width, sbor[2]).distance(Point(xp_ep, h_ep))
min_ep_r = Point(0.5*width, sbor[3]).distance(Point(xp_ep, h_ep))
max_ep_r = Point(0.5*width, sbor[2]).distance(Point(xp_ep, h_ep))

min_nt2_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_nt2 + xp_mid12, h_nt))
max_nt2_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_nt2 + xp_mid12, h_nt))
min_nt2_r = Point(0.5 * width, sbor[3]).distance(Point(xp_nt2 + xp_mid12, h_nt))
max_nt2_r = Point(0.5 * width, sbor[2]).distance(Point(xp_nt2 + xp_mid12, h_nt))

min_kp2_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_kp2 + xp_mid12, h_kp))
max_kp2_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_kp2 + xp_mid12, h_kp))
min_kp2_r = Point(0.5 * width, sbor[3]).distance(Point(xp_kp2 + xp_mid12, h_kp))
max_kp2_r = Point(0.5 * width, sbor[2]).distance(Point(xp_kp2 + xp_mid12, h_kp))

min_up2_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_up2 + xp_mid12, h_up))
max_up2_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_up2 + xp_mid12, h_up))
min_up2_r = Point(0.5 * width, sbor[3]).distance(Point(xp_up2 + xp_mid12, h_up))
max_up2_r = Point(0.5 * width, sbor[2]).distance(Point(xp_up2 + xp_mid12, h_up))

min_ep2_l = Point(-0.5*width, sbor[3]).distance(Point(xp_ep2+xp_mid12, h_ep))
max_ep2_l = Point(-0.5*width, sbor[2]).distance(Point(xp_ep2+xp_mid12, h_ep))
min_ep2_r = Point(0.5*width, sbor[3]).distance(Point(xp_ep2+xp_mid12, h_ep))
max_ep2_r = Point(0.5*width, sbor[2]).distance(Point(xp_ep2+xp_mid12, h_ep))

min_nt3_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_nt3 + xp_mid13, h_nt))
max_nt3_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_nt3 + xp_mid13, h_nt))
min_nt3_r = Point(0.5 * width, sbor[3]).distance(Point(xp_nt3 + xp_mid13, h_nt))
max_nt3_r = Point(0.5 * width, sbor[2]).distance(Point(xp_nt3 + xp_mid13, h_nt))

min_kp3_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_kp3 + xp_mid13, h_kp))
max_kp3_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_kp3 + xp_mid13, h_kp))
min_kp3_r = Point(0.5 * width, sbor[3]).distance(Point(xp_kp3 + xp_mid13, h_kp))
max_kp3_r = Point(0.5 * width, sbor[2]).distance(Point(xp_kp3 + xp_mid13, h_kp))

min_up3_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_up3 + xp_mid13, h_up))
max_up3_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_up3 + xp_mid13, h_up))
min_up3_r = Point(0.5 * width, sbor[3]).distance(Point(xp_up3 + xp_mid13, h_up))
max_up3_r = Point(0.5 * width, sbor[2]).distance(Point(xp_up3 + xp_mid13, h_up))

min_ep3_l = Point(-0.5*width, sbor[3]).distance(Point(xp_ep3+xp_mid13, h_ep))
max_ep3_l = Point(-0.5*width, sbor[2]).distance(Point(xp_ep3+xp_mid13, h_ep))
min_ep3_r = Point(0.5*width, sbor[3]).distance(Point(xp_ep3+xp_mid13, h_ep))
max_ep3_r = Point(0.5*width, sbor[2]).distance(Point(xp_ep3+xp_mid13, h_ep))

# ЭКРАН
# стекло - высчитываем d для подсчёта энергии преломлённой волны
e1 = 1
e2 = 4
mu1 = 1
mu2 = 0.99

n1 = (e1 * mu1) ** 0.5
n2 = (e2 * mu2) ** 0.5
k_glass = ((n1 - n2) / (n1 + n2)) ** 2
d_glass = 1 - k_glass


# РАСЧЁТЫ


# по теореме Пифагора расчёт значения вектора из составляющих х и y
def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5


# магнитное поле гармоники f для заданной координаты x и z
def magnetic_calc(x_m, z_m, f_m):
    # общая сила тока гармоники
    I_h = I1 * harm.get(f_m)[0]

    # сила тока по проводам
    Ikp = part_kp * I_h
    Int = part_nt * I_h
    Iup = part_up * I_h
    Iep = part_ep * I_h

    # расчёт x и z составляющих магнитного поля от правого рельса для КП
    x = x_m - xp_kp
    h1xkp = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / (x ** 2 + (h_kp - z_m) ** 2))
    h1zkp = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_kp - z_m) ** 2))
    # расчёт x и z составляющих магнитного поля от левого рельса для КП
    x = x_m - 2 * xp - xp_kp
    h2xkp = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))

    # далее аналогично для остальных проводов:
    # НТ
    x = x_m - xp_nt
    h1xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
    x = x_m - 2 * xp - xp_nt
    h2xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))

    # УП
    x = x_m - xp_up
    x2 = -xp + xp_up
    h1xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup = Iup / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_up - z_m) ** 2))
    x = x_m - xp_up - 2 * xp
    x2 = -xp + xp_up
    h2xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2zup = Iup / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
                (x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
 
    # ЭП
    x = x_m - xp_ep
    x2 = -xp + xp_ep
    h1xep = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / (x ** 2 + (h_ep - z_m) ** 2))
    h1zep = Iep / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_ep - z_m) ** 2))
    x = x_m - xp_ep - 2 * xp
    x2 = -xp + xp_ep
    h2xep = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / ((x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2zep = Iep / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))          

 
    # второй путь
    
    I_h = I2 * harm.get(f_m)[0] 

    Ikp = part_kp * I_h
    Int = part_nt * I_h
    Iup = part_up * I_h
    Iep = part_ep * I_h 

    # КП2
    x = x_m - (xp_kp2 + xp_mid12)
    h1xkp_2 = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / (x ** 2 + (h_kp - z_m) ** 2))
    h1zkp_2 = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_kp - z_m) ** 2))
    x = x_m - 2 * xp - (xp_kp2 + xp_mid12)
    h2xkp_2 = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp_2 = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))


    # НТ2
    x = x_m - (xp_nt2 + xp_mid12)
    h1xnt_2 = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt_2 = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
 
    x = x_m - 2 * xp - (xp_nt2 + xp_mid12)
    h2xnt_2 = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt_2 = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
 

    # УП2
    x = x_m - (xp_up2 + xp_mid12)
    x2 = -xp + xp_up2
    h1xup_2 = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup_2 = Iup / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_up - z_m) ** 2))
    x = x_m - (xp_up2 + xp_mid12) - 2 * xp
    x2 = -xp + xp_up2
    h2xup_2 = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2zup_2 = Iup / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_up - z_m) ** 2))

    # ЭП2
    x = x_m - (xp_ep2 + xp_mid12)
    x2 = -xp + xp_ep2
    h1xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / (x ** 2 + (h_ep - z_m) ** 2))
    h1zep_2 = Iep / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_ep - z_m) ** 2))
    x = x_m - (xp_ep2 + xp_mid12) - 2 * xp
    x2 = -xp + xp_ep2
    h2xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / ((x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2zep_2 = Iep / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))

    # третий путь
    
    I_h = I3 * harm.get(f_m)[0]

    Ikp = part_kp * I_h
    Int = part_nt * I_h
    Iup = part_up * I_h
    Iep = part_ep * I_h
    
    # КП3
    x = x_m - (xp_kp2 + xp_mid13)
    h1xkp_3 = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m**2) + (z_m - h_kp)/(x ** 2 + (h_kp - z_m)**2))
    h1zkp_3 = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1/(x ** 2 + (h_kp - z_m) ** 2))
    x = x_m - 2*xp - (xp_kp2 + xp_mid13)
    h2xkp_3 = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp_3 = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))

    # НТ3
    x = x_m - (xp_nt2 + xp_mid13)
    h1xnt_3 = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt_3 = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
    x = x_m - 2 * xp - (xp_nt2 + xp_mid13)
    h2xnt_3 = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt_3 = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))

    # УП3
    x = x_m - (xp_up2 + xp_mid13)
    x2 = -xp + xp_up2
    h1xup_3 = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup_3 = Iup / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_up - z_m) ** 2))
    x = x_m - (xp_up2 + xp_mid13) - 2 * xp
    x2 = -xp + xp_up2
    h2xup_3 = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2zup_3 = Iup / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
                (x + 2 * xp) ** 2 + (h_up - z_m) ** 2))

    # ЭП3
    x = x_m - (xp_ep2 + xp_mid13)
    x2 = -xp + xp_ep2
    h1xep_3 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / (x ** 2 + (h_ep - z_m) ** 2))
    h1zep_3 = Iep / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_ep - z_m) ** 2))
    x = x_m - (xp_ep2 + xp_mid13) - 2 * xp
    x2 = -xp + xp_ep2
    h2xep_3 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / ((x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2zep_3 = Iep / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))


    # Сумма всех магнитных полей по оси x        
    hx1 = sum([h1xkp, h2xkp, h1xnt, h2xnt, h1xup, h2xup, h1xep, h2xep])
    hx2 = sum([h1xkp_2, h2xkp_2, h1xnt_2, h2xnt_2, h1xup_2, h2xup_2, h1xep_2, h2xep_2])
    hx3 = sum([h1xkp_3, h2xkp_3, h1xnt_3, h2xnt_3, h1xup_3, h2xup_3, h1xep_3, h2xep_3])
    # Сумма всех магнитных полей по оси z
    hz1 = sum([h1zkp, h2zkp, h1znt, h2znt, h1zup, h2zup, h1zep, h2zep])
    hz2 = sum([h1zkp_2, h2zkp_2, h1znt_2, h2znt_2, h1zup_2, h2zup_2, h1zep_2, h2zep_2])
    hz3 = sum([h1zkp_3, h2zkp_3, h1znt_3, h2znt_3, h1zup_3, h2zup_3, h1zep_3, h2zep_3])
    # Итоговое магнитное поле по теореме Пифагора:
    h1 = mix(hx1, hz1)
    h2 = mix(hx2, hz2)
    h3 = mix(hx3, hz3)
    
    # результат - значение магнитного поля в этой точке для выбранной гармоники
    return [h1, h2, h3]


# расчёт электрического поля для гармоники f в точке x, z
def electric_calc(x_e, z_e, f_e):
    U_h = U1 * harm.get(f_e)[1]
    
    a = x_e - xp_kp 
    ekpx = U_h * a / log(2 * h_kp / d_kp) * (1 / ((h_kp - z_e) ** 2 + a ** 2) - 1 / ((h_kp + z_e) ** 2 + a ** 2)) 
    ekpz = U_h / log(2 * h_kp / d_kp) * ((h_kp - z_e) / ((h_kp - z_e) ** 2 + a ** 2) + ((h_kp + z_e)) / ((h_kp + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_nt
    entx = U_h * a / log(2 * h_nt / d_nt) * (1 / ((h_nt - z_e) ** 2 + a ** 2) - 1 / ((h_nt + z_e) ** 2 + a ** 2)) 
    entz = U_h / log(2 * h_nt / d_nt) * ((h_nt - z_e) / ((h_nt - z_e) ** 2 + a ** 2) + ((h_nt + z_e)) / ((h_nt + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_up
    eupx = U_h * a / log(2 * h_up / d_up) * (1 / ((h_up - z_e) ** 2 + a ** 2) - 1 / ((h_up + z_e) ** 2 + a ** 2))
    eupz = U_h / log(2 * h_up / d_up) * ((h_up - z_e) / ((h_up - z_e) ** 2 + a ** 2) + ((h_up + z_e)) / ((h_up + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_ep
    eepx = -U_h * a / log(2 * h_ep / d_ep) * (1 / ((h_ep - z_e) ** 2 + a ** 2) - 1 / ((h_ep + z_e) ** 2 + a ** 2)) 
    eepz = -U_h / log(2 * h_ep / d_ep) * ((h_ep - z_e) / ((h_ep - z_e) ** 2 + a ** 2) + ((h_ep + z_e)) / ((h_ep + z_e) ** 2 + a ** 2)) 


    U_h = U2 * harm.get(f_e)[1]
    
    a = x_e - xp_kp2 - xp_mid12
    ekpx2 = U_h * a / log(2 * h_kp / d_kp) * (1 / ((h_kp - z_e) ** 2 + a ** 2) - 1 / ((h_kp + z_e) ** 2 + a ** 2)) 
    ekpz2 = U_h / log(2 * h_kp / d_kp) * ((h_kp - z_e) / ((h_kp - z_e) ** 2 + a ** 2) + ((h_kp + z_e)) / ((h_kp + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_nt2 - xp_mid12
    entx2 = U_h * a / log(2 * h_nt / d_nt) * (1 / ((h_nt - z_e) ** 2 + a ** 2) - 1 / ((h_nt + z_e) ** 2 + a ** 2)) 
    entz2 = U_h / log(2 * h_nt / d_nt) * ((h_nt - z_e) / ((h_nt - z_e) ** 2 + a ** 2) + ((h_nt + z_e)) / ((h_nt + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_up2 - xp_mid12
    eupx2 = U_h * a / log(2 * h_up / d_up) * (1 / ((h_up - z_e) ** 2 + a ** 2) - 1 / ((h_up + z_e) ** 2 + a ** 2)) 
    eupz2 = U_h / log(2 * h_up / d_up) * ((h_up - z_e) / ((h_up - z_e) ** 2 + a ** 2) + ((h_up + z_e)) / ((h_up + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_ep2 - xp_mid12
    eepx2 = -U_h * a / log(2 * h_ep / d_ep) * (1 / ((h_ep - z_e) ** 2 + a ** 2) - 1 / ((h_ep + z_e) ** 2 + a ** 2)) 
    eepz2 = -U_h / log(2 * h_ep / d_ep) * ((h_ep - z_e) / ((h_ep - z_e) ** 2 + a ** 2) + ((h_ep + z_e)) / ((h_ep + z_e) ** 2 + a ** 2)) 


    U_h = U3 * harm.get(f_e)[1]
    
    a = x_e - xp_kp2 - xp_mid13
    ekpx3 = U_h * a / log(2 * h_kp / d_kp) * (1 / ((h_kp - z_e) ** 2 + a ** 2) - 1 / ((h_kp + z_e) ** 2 + a ** 2)) 
    ekpz3 = U_h / log(2 * h_kp / d_kp) * ((h_kp - z_e) / ((h_kp - z_e) ** 2 + a ** 2) + ((h_kp + z_e)) / ((h_kp + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_nt3 - xp_mid13
    entx3 = U_h * a / log(2 * h_nt / d_nt) * (1 / ((h_nt - z_e) ** 2 + a ** 2) - 1 / ((h_nt + z_e) ** 2 + a ** 2)) 
    entz3 = U_h / log(2 * h_nt / d_nt) * ((h_nt - z_e) / ((h_nt - z_e) ** 2 + a ** 2) + ((h_nt + z_e)) / ((h_nt + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_up2 - xp_mid13
    eupx3 = U_h * a / log(2 * h_up / d_up) * (1 / ((h_up - z_e) ** 2 + a ** 2) - 1 / ((h_up + z_e) ** 2 + a ** 2)) 
    eupz3 = U_h / log(2 * h_up / d_up) * ((h_up - z_e) / ((h_up - z_e) ** 2 + a ** 2) + ((h_up + z_e)) / ((h_up + z_e) ** 2 + a ** 2)) 

    a = x_e - xp_ep2 - xp_mid13
    eepx3 = -U_h * a / log(2 * h_ep / d_ep) * (1 / ((h_ep - z_e) ** 2 + a ** 2) - 1 / ((h_ep + z_e) ** 2 + a ** 2)) 
    eepz3 = -U_h / log(2 * h_ep / d_ep) * ((h_ep - z_e) / ((h_ep - z_e) ** 2 + a ** 2) + ((h_ep + z_e)) / ((h_ep + z_e) ** 2 + a ** 2)) 
   
    # Сумма всех электрических полей по оси x        
    ex1 = sum([ekpx, entx, eupx, eepx,])
    ex2 = sum([ekpx2, entx2, eupx2, eepx2])
    ex3 = sum([ekpx3, entx3, eupx3, eepx3])
    # Сумма всех электрических полей по оси z
    ez1 = sum([ekpz, entz, eupz, eepz])
    ez2 = sum([ekpz2, entz2, eupz2, eepz2])
    ez3 = sum([ekpz3, entz3, eupz3, eepz3])
    # Итоговое электрических поле по теореме Пифагора:
    e1 = mix(ex1, ez1)
    e2 = mix(ex2, ez2)
    e3 = mix(ex3, ez3)
    
    # результат - значение электрических поля в этой точке для выбранной гармоники
    return [e1, e2, e3]


# суммироввание всех полей всех гармоник и подсчёт энергии для каждой точки:
def full_field(res_en):
    sum_h, sum_e, sum_eng = 0, 0, 0
    # cумма полей по гармоникам
    for en in res_en[0].values():
        for j in range(0, 3):
            sum_h += en[0][j]  # магнитная составляющая
            sum_e += en[1][j]  # электрическая составляющая
            sum_eng += en[0][j] * en[1][j] * sin(alpha)
    return sum_h, sum_e, sum_eng  # энергия - произведение магнитного и электрического поля
	

#  расчёт экрана переменного поля
def ekran(en):
    x, y, z = en[1]  # координаты точки

    # расстояние от текущей точки до проводов - для расчёта лобовых окон
    kppth = LineString([(x, y, z), (x, xp_kp, h_kp)])
    ntpth = LineString([(x, y, z), (x, xp_nt, h_nt)])
    uppth = LineString([(x, y, z), (x, xp_up, h_up)])
    eppth = LineString([(x, y, z), (x, xp_ep, h_ep)])
    # проверяем, попадает ли лобовое окно по направлению от текущей точки до проводов
    kp_pass = kppth.intersects(frontWindleft) or kppth.intersects(frontWindright)
    nt_pass = ntpth.intersects(frontWindleft) or ntpth.intersects(frontWindright)
    up_pass = ntpth.intersects(frontWindleft) or uppth.intersects(frontWindright)
    ep_pass = eppth.intersects(frontWindleft) or eppth.intersects(frontWindright)


    # для каждого провода проверяем, попадает ли текущая точка в тень от бокового окна или нет
    kp_dist = Point(y, z).distance(Point(xp_kp, h_kp))  # направление от точки до провода
    # есть ли на пути этого направления окно
    kp_pass |= (kp_dist >= min_kp) and (kp_dist <= max_kp) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
 
    nt_dist = Point(y, z).distance(Point(xp_nt, h_nt))
    nt_pass |= (nt_dist >= min_nt) and (nt_dist <= max_nt) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
 
    up_dist = Point(y, z).distance(Point(xp_up, h_up))
    up_pass |= (up_dist >= min_up_l) and (up_dist <= max_up_l) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
    up_pass |= (up_dist >= min_up_r) and (up_dist <= max_up_r) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
               
    ep_dist = Point(y, z).distance(Point(xp_ep, h_ep))
    ep_pass |= (ep_dist >= min_ep_l) and (ep_dist <= max_ep_l) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
    ep_pass |= (ep_dist >= min_ep_r) and (ep_dist <= max_ep_r) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])

    kp_sec_dist = Point(y, z).distance(Point(xp_kp2 + xp_mid12, h_kp))
    kp_sec_pass = (kp_sec_dist >= min_kp2_l) and (kp_sec_dist <= max_kp2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    kp_sec_pass |= (kp_sec_dist >= min_kp2_r) and (kp_sec_dist <= max_kp2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    nt_sec_dist = Point(y, z).distance(Point(xp_nt2 + xp_mid12, h_nt))
    nt_sec_pass = (nt_sec_dist >= min_nt2_l) and (nt_sec_dist <= max_nt2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    nt_sec_pass |= (nt_sec_dist >= min_nt2_r) and (nt_sec_dist <= max_nt2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    up_sec_dist = Point(y, z).distance(Point(xp_up2 + xp_mid12, h_up))
    up_sec_pass = (up_sec_dist >= min_up2_l) and (up_sec_dist <= max_up2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    up_sec_pass |= (up_sec_dist >= min_up2_r) and (up_sec_dist <= max_up2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])
                   
    ep_sec_dist = Point(y, z).distance(Point(xp_ep2 + xp_mid12, h_ep))
    ep_sec_pass = (ep_sec_dist >= min_ep2_l) and (ep_sec_dist <= max_ep2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    ep_sec_pass = (ep_sec_dist >= min_ep2_r) and (ep_sec_dist <= max_ep2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])

    kp_thd_dist = Point(y, z).distance(Point(xp_kp3 + xp_mid13, h_kp))
    kp_thd_pass = (kp_thd_dist >= min_kp3_l) and (kp_thd_dist <= max_kp3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    kp_thd_pass |= (kp_thd_dist >= min_kp3_r) and (kp_thd_dist <= max_kp3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    nt_thd_dist = Point(y, z).distance(Point(xp_nt3 + xp_mid13, h_nt))
    nt_thd_pass = (nt_thd_dist >= min_nt3_l) and (nt_thd_dist <= max_nt3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    nt_thd_pass |= (nt_thd_dist >= min_nt3_r) and (nt_thd_dist <= max_nt3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    up_thd_dist = Point(y, z).distance(Point(xp_up3 + xp_mid13, h_up))
    up_thd_pass = (up_thd_dist >= min_up3_l) and (up_thd_dist <= max_up3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    up_thd_pass |= (up_thd_dist >= min_up3_r) and (up_thd_dist <= max_up3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    ep_thd_dist = Point(y, z).distance(Point(xp_ep3 + xp_mid13, h_ep))
    ep_thd_pass = (ep_thd_dist >= min_ep3_l) and (ep_thd_dist <= max_ep3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                    and (z >= sbor[2]) and (z <= sbor[3])
    ep_thd_pass |= (ep_thd_dist >= min_ep3_r) and (ep_thd_dist <= max_ep3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                    and (z >= sbor[2]) and (z <= sbor[3])

 

    # для каждой точки внутри кабины проверяем, проходит ли для неё какое-либо поле через стекло
    # сталь: электрическое поле полностью отражается, магнитное полностью затухает
    # стекло: и электрическое, и магнитное домножаются на d_glass по формуле:
    # Эпрел = Эпад*d = (ExH)*d = E*d x H*d
    if (abs(y) <= 0.5 * width) and (z >= gr_floor) and (z <= floor + height) and (x > 0) and (x < length):
        # внутри кабины
        pass_1 = kp_pass or nt_pass or up_pass or ep_pass
        pass_2 = kp_sec_pass or nt_sec_pass or up_sec_pass or ep_sec_pass
        pass_3 = kp_thd_pass or nt_thd_pass or up_thd_pass or ep_thd_pass
        # поле КП через стекло
        if pass_1:
            for f in en[0].keys():
                en[0][f][0][0] *= d_glass
                en[0][f][1][0] *= d_glass
        if pass_2:
            for f in en[0].keys():
                en[0][f][0][1] *= d_glass
                en[0][f][1][1] *= d_glass
        if pass_3:
            for f in en[0].keys():
                en[0][f][0][2] *= d_glass
                en[0][f][1][2] *= d_glass
        if not (pass_1 or pass_2 or pass_3):
            # если ни через одно стекло не проходит, значит тут сталь, т.е. поле равно нулю
            for f in en[0].keys():
                en[0][f][0] = [0, 0, 0]
                en[0][f][1] = [0, 0, 0]
    return en 
     

# ГРАФИКА И ВЫВОД

# сохранение картинки в файл
def show(name):
    mng = plt.get_current_fig_manager()  # захват изображения 
    # mng.window.state('zoomed')  # вывод изображения на весь экран если граф.оболочка это поддерживает
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")
    # сохранение картинки в файл дата_время_название.png в папку со скриптом


# рисование линий кабины вид спереди
def fr_kab_lines(star=False):
    ln_ = '--'  # стиль линии

    cl_ = 'royalblue'  # окна
    plt.hlines(bor[4], bor[2], bor[3], colors=cl_, linestyles=ln_)
    plt.hlines(bor[5], bor[2], bor[3], colors=cl_, linestyles=ln_)
    plt.vlines(bor[2], bor[4], bor[5], colors=cl_, linestyles=ln_)
    plt.vlines(bor[3], bor[4], bor[5], colors=cl_, linestyles=ln_)
    plt.vlines(0, bor[4], bor[5], colors=cl_, linestyles=ln_)
    # дворники
    plt.plot(np.array([bor[2] + .1, bor[2] + 0.6]),
             np.array([bor[4] + .1, bor[5] + .1]), c=cl_, linestyle=ln_)
    plt.plot(np.array([bor[3] - .1, bor[3] - 0.6]),
             np.array([bor[4] + .1, bor[5] + .1]), c=cl_, linestyle=ln_)

    if star:
        cl_ = 'red'  # полосы и звезда
        plt.hlines(bor[4] - .3, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=3)
        plt.hlines(floor + .3, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=3)
        plt.hlines(1.4, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=4)
        plt.scatter(0, 2.7, s=200, marker='*', color=cl_)

        # пантограф
        cl_ = 'forestgreen'
        delta = (h_kp - (height + floor + .3) - .2)/12
        for i in range(1, 12):
            plt.hlines(height + floor + .3 + i * delta, -.4, -.7, colors=cl_, linestyles='solid')
            plt.hlines(height + floor + .3 + i * delta, .4, .7, colors=cl_, linestyles='solid')
        plt.plot(np.array([-.5*width, -1, 1, .5*width]),
                 np.array([h_kp-.4, h_kp-.1, h_kp-.1, h_kp-.4]),
                 c=cl_, linestyle='solid', lw=3)

    cl_ = 'forestgreen'  # очертания кабины
    plt.hlines(height + floor, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.hlines(floor+.1, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.hlines(gr_floor, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.vlines(-0.5 * width, gr_floor, height + floor, colors=cl_, linestyles=ln_)
    plt.vlines(0.5 * width, gr_floor, height + floor, colors=cl_, linestyles=ln_)

    # низ
    plt.plot(np.array([-.5 * width + .1, -.5 * width + .4, .5 * width - .4, .5 * width - .1]),
             np.array([gr_floor, .4, .4, gr_floor]), c=cl_, linestyle=ln_)
    delta = (width - .2) / 6
    for i in range(1, 6):
        plt.vlines(-.5 * width + .1 + delta * i, gr_floor, .6, colors=cl_, linestyles=ln_)

    # головная фара и крыша
    bj1 = ptch.Arc((0, floor + height), width, .8, theta1=0, theta2=180, color=cl_, linestyle=ln_)
    bj2 = ptch.Circle((0, floor + height + 0.3), 0.2, color=cl_, linestyle=ln_, fill=None)

    for bj in [bj1, bj2]:
        plt.gca().add_artist(bj)


# рисование линий внутри кабины
def kab_lines_front():
    d = 0.13
    cl = 'blue'
    plt.hlines(z_chair, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair - 0.05, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair - 0.05, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.vlines(y_chel - d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')
    plt.vlines(y_chel + d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')
    plt.vlines(-y_chel - d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')
    plt.vlines(-y_chel + d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')

    d = 0.12
    plt.hlines(z_chair + 0.05, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair + 0.05 + 2 * d, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair + 0.05, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair + 0.05 + 2 * d, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.vlines(y_chel - d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')
    plt.vlines(y_chel + d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')
    plt.vlines(-y_chel - d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')
    plt.vlines(-y_chel + d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')


# рисование линий кабины вид сверху
def kab_lines_up():
    d = 0.12
    cl = 'blue'
    plt.hlines(y_chel - d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.hlines(y_chel + d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.hlines(-y_chel - d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.hlines(-y_chel + d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel - d, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel - d, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.hlines(y_chel - d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.hlines(y_chel + d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.hlines(-y_chel - d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.hlines(-y_chel + d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.05, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.10, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.05, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.10, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.vlines(bor[0], bor[2], -0.22, colors='white', linestyles='--')
    plt.vlines(bor[1], bor[2], -0.22, colors='white', linestyles='--')
    plt.hlines(bor[2], bor[0], bor[1], colors='white', linestyles='--')
    plt.hlines(-0.22, bor[0], bor[1], colors='white', linestyles='--')

    plt.vlines(bor[0], 0.22, bor[3], colors='white', linestyles='--')
    plt.vlines(bor[1], 0.22, bor[3], colors='white', linestyles='--')
    plt.hlines(0.22, bor[0], bor[1], colors='white', linestyles='--')
    plt.hlines(bor[3], bor[0], bor[1], colors='white', linestyles='--')

    cl = 'black'
    plt.plot(np.array([0.01, bor[0]]), np.array([0, bor[2]]), c=cl, linestyle='--')
    plt.plot(np.array([0.01, bor[0]]), np.array([0, bor[3]]), c=cl, linestyle='--')

    plt.hlines(0.5 * width - 0.01, 0, length, colors=cl, linestyles='--')
    plt.hlines(-0.5 * width + 0.01, 0, length, colors=cl, linestyles='--')
    plt.vlines(0.01, 0.5 * width, -0.5 * width, colors=cl, linestyles='--')
    plt.vlines(length - 0.01, 0.5 * width, -0.5 * width, colors=cl, linestyles='--')


# построение вида сверху без электровоза
def visual_up():
    print('График строится..................')

    # границы графика
    Xmin = -0.5
    Xmax = length + 0.5
    Ymax = 1 * 0.5 * width * 1.3
    Ymin = xp_up * 1.15

    # разбиение по точкам
    x = np.linspace(Xmin, Xmax, dis)
    y = np.linspace(Ymin, Ymax, dis)

    # расчёт значений полей для каждой точки графика
    every_f = [[[{fr: [magnetic_calc(y_, z_graph, fr), electric_calc(y_, z_graph, fr)] for fr in harm.keys()},
                 [x_, y_, z_graph]] for x_ in x] for y_ in y]

    # применяем экран и считаем итоговое значение для каждой точки
    summar = [[full_field(x_el) for x_el in y_list] for y_list in every_f]

    # формируем массив значений на магнитную, электрическую составляющую и энергию
    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    # общая функция отрисовки графика
    def do_graph(content, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):
        # создаём объект точек графика
        ct = plt.contour(x, y, content, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        # создаём линии уровней из объекта точек        
        plt.clabel(ct, fontsize=10)
        # отрисовка
        plt.imshow(content, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)
        # раскраска        
        plt.colorbar()

        # рисование и подпись проводов
        for delta_y in [xp_kp, xp_up, xp_nt, xp_ep]:
            plt.hlines(delta_y, Xmin, Xmax, color='black', linewidth=2)
        plt.text(-.5, xp_kp - 0.1, 'КП', color='black')
        plt.text(-.5, xp_up - 0.1, 'УП', color='black')
        plt.text(-.5, xp_nt + 0.4, 'НТ', color='black')
        plt.text(-.5, xp_ep + 0.1, 'ЕП', color='black')


        # рисование очертания поезда
        plt.hlines(0.5 * width, 0, length, colors='red', linestyles='--')
        plt.hlines(-0.5 * width, 0, length, colors='red', linestyles='--')
        plt.vlines(0, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.vlines(length, -0.5 * width, 0.5 * width, colors='red', linestyles='--')

        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)

    # отрисовка по очереди магнитного, электрического и энергии
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Контактная сеть вид сверху (без электровоза)'
    plt.subplot(1, 3, 1)
    do_graph(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 2)
    do_graph(electric, 'Электрическое', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 3)
    do_graph(energy, 'Энергия', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.suptitle(name)
    show(name)

    print('График построен.')

    return every_f  # возвращаем поле для перерасчёта с локомотивом


# вывод вида спереди без электровоза
def visual_front():
    print('График строится..................')

    # границы графика
    Ymax = 1 * max(xp, width) * 1.15
    Ymin = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    # разбиение на точки
    y = np.linspace(Ymin, Ymax, dis)
    z = np.linspace(Zmin, Zmax, dis)

    # расчёт значений полей для каждой точки графика
    every_f = [[[{fr: [magnetic_calc(y_, z_, fr), electric_calc(y_, z_, fr)] for fr in harm.keys()},
                 [x_chel, y_, z_]] for y_ in y] for z_ in z]

    # считаем итоговое значение для каждой точки
    summar = [[full_field(x_el)[2] for x_el in y_list] for y_list in every_f]

    def graph_do(znach, name_):
        # задаём уровни
        b = len(str(round(np.amax(znach))))  # высчитываем диапазон графика для правильного отображения линий уровня
        levels = [i * (10 ** j) for j in range(0, b) for i in [1, 2, 5, 7]]
        # создаём объект точек графика
        ct = plt.contour(y, z, znach, alpha=0.75, colors='black', linestyles='dotted',
                         levels=levels)
        # создаём линии уровней из объекта точек
        plt.clabel(ct, fontsize=10)
        # отрисовка
        summar[0][0] = 2000  # несущественной для построения точке даём минимальное выводное зеначение чтобы график был соразмерен по цвету отражённому графику
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
        # раскраска
        plt.colorbar()

        # названия проводов
        plt.text(xp_kp, h_kp, 'КП', color='black', fontsize=14)
        plt.text(xp_up, h_up, 'УП', color='black', fontsize=14)
        plt.text(xp_nt, h_nt, 'НТ', color='black', fontsize=14)
        plt.text(xp_ep, h_ep, 'ЕП', color='black', fontsize=14)

        # очертания кабины
        fr_kab_lines(star=True)

        # название осей
        plt.xlabel('Ось y, метры')
        plt.ylabel('Ось z, метры')

        plt.title(name_)  # подпись названия
        show(name_)  # вывести и сохранить

    # вывод общей КС
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    graph_do(summar, 'Контактная сеть вид спереди (без электровоза) - Энергия')

    print('График построен.')
    return every_f  # возвращаем поле для перерасчёта с локомотивом


# вывод вида сверху с электровозом
def visual_up_locomotive(ext_f):
    print('График строится..................')
    # границы графика
    Xmin = 0
    Xmax = length
    Ymax = -0.5 * width
    Ymin = -Ymax

    # выборка области для отрисовки из уже посчитанного поля
    inside = [[full_field(ekran(el)) for el in y_list if (el[1][0] >= Xmin) and (el[1][0] <= Xmax)]
              for y_list in ext_f if abs(y_list[0][1][1]) <= 0.5 * width]

    # формируем массивы значений на магнитную, электрическую составляющую и энергию
    energy = [[x_el[2] for x_el in y_list] for y_list in inside]

    # общая функция отрисовки графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # отрисовка
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        # раскраска          
        plt.colorbar()

        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)  # подпись названия

    # отрисовка энергии
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Кабина вид сверху (c экраном) - энергия'
    graph_do(energy, name, x_lb='Ось x, метры', )
    kab_lines_up()
    show(name)
    print('График построен.')

def visual_front_locomotive(ext_f):
    print('График строится..................')

    # границы графика
    Ymin, Ymax = -0.6 * width, 0.6 * width
    Zmin, Zmax = floor + height + 1, 0.1

    # применяем экран
    ekran_ = [[ekran(y_el) for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ext_f
              if z_list[0][1][2] < Zmin]

    # перевод значений посчитанных для каждой гармоники каждого провода в одно значение
    summar = [[full_field(x_el) for x_el in y_list] for y_list in ekran_]
    # выбор значений только энергии
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    # разбиение по точкам
    y_ln = np.linspace(Ymin, Ymax, len(ekran_[0]))
    z_ln = np.linspace(Zmin, Zmax, len(ekran_))
    # находим координаты человека в массиве точек
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    chel_z = np.where(z_ln == max([z_ for z_ in z_ln if z_ <= z_chel]))[0][0]

    # общая функция отрисовки графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # отрисовка        
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
        # раскраска    
        plt.colorbar()

        # очертания кабины
        fr_kab_lines()
        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)  # подпись названия

    # отрисовка энергии поля
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Кабина вид спереди (c экраном) - энергия'
    graph_do(energy, name, x_lb='Ось y, метры', )
    kab_lines_front()
    show(name)

    # отрисовка отрисовка энергии по гармоникам
    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники вид спереди (экран) - энергия'
    plt.title(name)
    i = 0
    chel_harm = {}
    # для каждой гармоники формируем массив точек на отрисовку + считаем воздействие в положении человека
    for fr in harm.keys():
        i += 1
        plt.subplot(3, 3, i)
        # считаем энергию для конкретной гармоники
        data = [[
                 el[0][fr][0][0]*el[0][fr][1][0] + el[0][fr][0][1]*el[0][fr][1][1] + el[0][fr][0][2]*el[0][fr][1][2]
                 for el in lst] for lst in ekran_]
        chel_harm[fr] = data[chel_z][chel_y]
        graph_do(data, '', y_lb=str(fr))
        kab_lines_front()
    plt.subplot(3, 3, 9)
    plt.bar(range(0, len(harm.keys())), chel_harm.values())
    #plt.suptitle(name)
    show(name)

    print('График построен.')
    # возвращаем значения для гармоник в координатах человека чтобы вывести в блоке статистики
    return chel_harm



# ВЫВОД ПАРАМЕТРОВ
print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print('Первый путь')
print(f'Напряжение: {U1} Вольт')
print(f'Суммарный ток: {I1} Ампер')
print('Второй путь')
print(f'Напряжение: {U2} Вольт')
print(f'Суммарный ток: {I2} Ампер')
print('Третий путь')
print(f'Напряжение: {U3} Вольт')
print(f'Суммарный ток: {I3} Ампер')
print(f'Высота среза: {z_graph} метров')

# ПОСТРОЕНИЕ ГРАФИКА

gph_num = 0
print('\nБез электровоза:')
print('\nВид сверху')
cont_f_up = visual_up()

print('\nВид спереди')
cont_f_front = visual_front()

print('\nКабина электровоза:')
print('\nВид сверху')
visual_up_locomotive(cont_f_up)

print('\nВид спереди')
chel_harm = visual_front_locomotive(cont_f_front)


# РАСЧЁТ СТАТИСТИКИ

print('СТАТИСТИКА\n')

print('Гармоники энергии поля для человека:')
for f, znach in chel_harm.items():
    print(f, ': %.4f' % znach)

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

chel_f_per = [{fr: [magnetic_calc(y_chel, z_chel, fr), electric_calc(y_chel, z_chel, fr)] for fr in harm.keys()},
              (x_chel, y_chel, z_chel)]
no_ekran_per = full_field(chel_f_per)[2]
print('\nПеременное поле без экрана: %.4f' % no_ekran_per)

ekran_per = full_field(ekran(chel_f_per))[2]
print('Переменное поле с экраном %.4f' % ekran_per)
Dco = ekran_per * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)
print('Удельная суточная доза облученной энергии: %.4f' % Dco)

plt.show()
