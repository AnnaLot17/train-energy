# Импорт библиотек
from math import log, exp, pi, atan
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

I = 300  # cуммарная сила тока, А
part_kp = 0.35
part_nt = 0.15
part_up = 0.50
part_ep = -0.4  # минус так как ток запущен в обратном направлении

U = 27000  # cуммарное напряжение, В

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
dis = 100  # дискретизация графиков (меньше - менее точно, но быстрее считает; больше - точнее, но дольше расчёт)
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
d_ep = 17.5 / 1000  # mm
h_kp = 6.0  # КП
h_nt = 7.8  # НТ
h_up = 8.8  # УП
h_ep = 8.4  # ЕП

# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
# min_x, max_x, min_y, max_y, min_z, max_z
bor = [0.2, 0.6, -1.2, 1.2, floor + 1.5, floor + 2.2]  # узлы лобового окна
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
    I_h = I * harm.get(f_m)[0]

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
            
    # Сумма всех магнитных полей по оси x        
    hx = sum([h1xkp, h2xkp, h1xnt, h2xnt, h1xup, h2xup, h1xep, h2xep])
    # Сумма всех магнитных полей по оси z
    hz = sum([h1zkp, h2zkp, h1znt, h2znt, h1zup, h2zup, h1zep, h1zep])
    # Итоговое магнитное поле по теореме Пифагора:
    h = mix(hx, hz)
    
    # результат - значение магнитного поля в этой точке для выбранной гармоники
    return h


# расчёт электрического поля для гармоники f в точке x, z
def electric_calc(x_e, z_e, f_e, reflect=False):
    # напряжение гармоники
    U_h = U * harm.get(f_e)[1]

    # электрическое поле от каждого провода
    ekp = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt))
    ent = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp))
    eup = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))
    eep = -U_h * log(1 + 4 * h_ep * z_e / ((x_e - xp_ep) ** 2 + (h_ep - z_e) ** 2)) / (2 * z_e * log(4 * h_ep / d_ep))

    # результат - сумма значений полей от каждого провода
    return sum([ekp, ent, eup, eep])


# суммироввание всех полей всех гармоник и подсчёт энергии для каждой точки:
def full_field(res_en):
    sum_h, sum_e = 0, 0
    # cумма полей по гармоникам
    for en in res_en[0].values():
        sum_h += en[0]  # магнитная составляющая
        sum_e += en[1]  # электрическая составляющая
    return sum_h, sum_e, sum_h*sum_e  # энергия - произведение магнитного и электрического поля


#  расчёт экрана переменного поля
def ekran(en):
    x, y, z = en[1]  # координаты точки

    # направление от текущей точки до проводов - для расчёта лобовых окон
    kppth = LineString([(x, y, z), (x, xp_kp, h_kp)])
    ntpth = LineString([(x, y, z), (x, xp_nt, h_nt)])
    uppth = LineString([(x, y, z), (x, xp_up, h_up)])
    eppth = LineString([(x, y, z), (x, xp_ep, h_ep)])
    # проверяем, попадает ли лобовое окно по направлению от текущей точки до проводов
    kp_pass = kppth.intersects(frontWindleft) or kppth.intersects(frontWindright)
    nt_pass = ntpth.intersects(frontWindleft) or ntpth.intersects(frontWindright)
    up_pass = uppth.intersects(frontWindleft) or ntpth.intersects(frontWindright)
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


    # сталь: электрическое поле полностью отражается, магнитное полностью затухает
    # стекло: и электрическое, и магнитное домножаются на d_glass по формуле:
    # Эпрел = Эпад*d = (ExH)*d = E*d x H*d
    
    if (abs(y) <= 0.5 * width) and (z >= gr_floor) and (z <= floor + height) and (x > 0) and (x < length):
        # внутри кабины
        if kp_pass or nt_pass or up_pass or ep_pass:
            # поле КП через стекло
            for f in en[0].keys():
                en[0][f][0] *= d_glass
                en[0][f][1] *= d_glass
        else:
            # если ни через одно стекло не проходит, значит тут сталь, т.е. поле равно нулю
            for f in en[0].keys():
                en[0][f][0] = 0
                en[0][f][1] = 0
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
    plt.hlines(floor + .1, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
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
    all_field = [[full_field(x_el) for x_el in y_list] for y_list in every_f]
    summar = [[x_el[2] for x_el in y_list] for y_list in all_field]


    # создаём новое окно
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    # задаём уровни
    b = 10 ** (len(str(round(np.amin(summar)))) - 1)  # для правильного отображения линий
    # создаём объект точек графика
    summar[0][0] = 100  # несущественной для построения точке даём минимальное выводное зеначение чтобы график был соразмерен по цвету отражённому графику
    ct = plt.contour(y, z, summar, alpha=0.75, colors='black', linestyles='dotted',
                     levels=[b, 2 * b, 5 * b, 7 * b, 10 * b, 20 * b, 50 * b, 100 * b, 200 * b, 500 * b, 700 * b])
    # создаём линии уровней из объекта точек
    plt.clabel(ct, fontsize=10)
    # отрисовка
    plt.imshow(summar, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
    # раскраска
    plt.colorbar()

    # названия проводов
    plt.text(xp_kp, h_kp, 'КП', color='white', fontsize=14)
    plt.text(xp_up, h_up, 'УП', color='white', fontsize=14)
    plt.text(xp_nt, h_nt, 'НТ', color='white', fontsize=14)
    plt.text(xp_ep, h_ep, 'ЕП', color='black', fontsize=14)

    # очертания кабины
    fr_kab_lines(star=True)

    # название осей
    plt.xlabel('Ось y, метры')
    plt.ylabel('Ось z, метры')

    plt.title('Контактная сеть вид спереди (без электровоза)')  # подпись названия

    show('вид сбоку')  # вывести и сохранить

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
    name = 'Гармоники вид спереди - энергия'
    plt.title(name)
    i = 0
    chel_harm = {}
    # для каждой гармоники формируем массив точек на отрисовку + считаем воздействие в положении человека
    for fr in harm.keys():
        i += 1
        plt.subplot(3, 3, i)
        # считаем энергию для конкретной гармоники
        data = [[el[0][fr][0] * el[0][fr][1]
                 for el in lst] for lst in ekran_]
        chel_harm[fr] = data[chel_z][chel_y]
        graph_do(data, '', y_lb=str(fr))
        kab_lines_front()
    plt.subplot(3, 3, 9)
    plt.bar(range(0, len(harm.keys())), chel_harm.values())
    plt.suptitle(name)
    show(name)
    print('График построен.')

    # возвращаем значения для гармоник в координатах человека чтобы вывести в блоке статистики
    return chel_harm


def glass_reflect(x, y, z):
    # для полей, отражённых окнами, строим "мнимые" провода, генерирующие зеркальные отражения их полей
    
    h_p = [(sbor[3]+sbor[2]) - h_kp,
           (sbor[3]+sbor[2]) - h_nt,
           (sbor[3]+sbor[2]) - h_up,
           (sbor[3]+sbor[2]) - h_ep]
    x_p = [xp_kp, xp_nt, xp_up, xp_ep]
    I_p = [part_kp, part_nt, part_up, part_ep]
    d_p = [d_kp, d_nt, d_up, d_ep]


    def energy():
        e = 0
        h = 0
               
        for f in harm.keys():
            hx, hz = 0, 0
            for i in range(0,4):
                I_h = I_p[i] * harm[f][0]

                x_m = y - x_p[i]
                hx += I_h / (4 * pi) * (z - h_p[i]) / (x_m ** 2 + (h_p[i] - z) ** 2)
                hz += I_h / (4 * pi) * x_m / (x_m ** 2 + (h_p[i] - z) ** 2)
                hx += I_h / (4 * pi) * (z - h_p[i]) / (x_m ** 2 + (h_p[i] - z) ** 2)
                hz += I_h / (4 * pi) * x_m / (x_m ** 2 + (h_p[i] - z) ** 2)
                
                U_h = U * harm[f][1]
                if i < 3:
                    e += abs(U_h * log(1 + 4 * h_p[i] * z / (x_m ** 2 + (h_p[i] - z) ** 2)) / (2 * z * log(2 * abs(h_p[i]) / d_p[i])))
                else:
                    e -= abs(U_h * log(1 + 4 * h_p[i] * z / (x_m ** 2 + (h_p[i] - z) ** 2)) / (2 * z * log(2 * abs(h_p[i]) / d_p[i])))
            h += mix(hx, hz)
        return e*h
 
    kp_dist = Point(y, z).distance(Point(xp_kp, h_p[0]))  # расстояние от точки до провода
    kp_pass = (kp_dist >= min_kp) and (kp_dist <= max_kp) \
              and (x >= sbor[0]) and (x <= sbor[1]) and abs(y) > .5*width  # попадает ли в траекторию окна

    nt_dist = Point(y, z).distance(Point(xp_nt, h_p[1]))
    nt_pass = (nt_dist >= min_nt) and (nt_dist <= max_nt) and (x >= sbor[0]) and (x <= sbor[1]) and abs(y) > .5*width

    up_dist = Point(y, z).distance(Point(xp_up, h_p[2]))
    up_pass = (up_dist >= min_up_l) and (up_dist <= max_up_l) and (x >= sbor[0]) and (x <= sbor[1]) and abs(y) > .5*width
    up_pass |= (up_dist >= min_up_r) and (up_dist <= max_up_r) and (x >= sbor[0]) and (x <= sbor[1]) and abs(y) > .5*width

    ep_dist = Point(y, z).distance(Point(xp_ep, h_p[3]))
    ep_pass = (ep_dist >= min_ep_l) and (ep_dist <= max_ep_l) and (x >= sbor[0]) and (x <= sbor[1]) and abs(y) > .5*width
    ep_pass |= (ep_dist >= min_ep_r) and (ep_dist <= max_ep_r) and (x >= sbor[0]) and (x <= sbor[1]) and abs(y) > .5*width

    if kp_pass or nt_pass or up_pass or ep_pass:
        return energy()
    else:
        return 0


def steel_reflect(y, z, x=None):
    if x:  # вид сверху
        if abs(y) < 0.5 * width and x > 0:  # если внутри кабны
            return 0
    else:  # вид спереди
        if abs(y) < 0.5 * width and z < height + floor and z > gr_floor:  # внутри кузова напряжённость равна 0
            return 0
    if z > height + floor:  # отражение вверх
        z = 2 * (height + floor) - z
    elif z < gr_floor:  # отражение вниз
        z = 2 * gr_floor - z
    if y < -0.5 * width:  # отражение влево
        y = -width - y
    elif y > 0.5 * width:  # отражение влево
        y = width - y

    E = 0
    for f in harm.keys():
        U_h = U * harm[f][1]

        # электрическое поле от каждого провода
        E += U_h * log(1 + 4 * h_kp * z / ((y - xp_kp) ** 2 + (h_kp - z) ** 2)) / (2 * z * log(2 * h_kp / d_kp))
        E += U_h * log(1 + 4 * h_nt * z / ((y - xp_nt) ** 2 + (h_nt - z) ** 2)) / (2 * z * log(2 * h_nt / d_nt))
        E += U_h * log(1 + 4 * h_up * z / ((y - xp_up) ** 2 + (h_up - z) ** 2)) / (2 * z * log(2 * h_up / d_up))

    return E


# визуализируем вид сверху с учётом отражённой от экрана
def visual_up_reflect(ext_f):
    print('График строится..................')

    # чтобы не возникло проблем при вычитании поля КС и поля отражений,
    # получаем список точек графика из уже рассчитанного ранее поля КС
    x_ln = [el[1][0] for el in ext_f[0]]
    y_ln = [el[0][1][1] for el in ext_f]
    Xmin, Xmax = x_ln[0], x_ln[-1]
    Ymin, Ymax = y_ln[0], y_ln[-1]

    # посчёт отражённого поля
    # посдчёт поля, отражённого от стекла
    refl_glass = np.array([[glass_reflect(x_, y_, z_graph) for x_ in x_ln] for y_ in y_ln])
    # подсчёт поля,. отражённого от стали
    refl_steel = np.array([[steel_reflect(y_, z_graph, x=x_) for x_ in x_ln] for y_ in y_ln])
    # суммирование отражённых полей
    summar_reflect = refl_glass+refl_steel
    # перевод в конечные значения внешнего поля с экраном
    summar_ext = np.array([[full_field(ekran(el))[2] for el in x_list] for x_list in ext_f])
    # # вычитаем из поля внешнего поле отражённое
    summar = np.absolute(summar_ext - summar_reflect)

    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name_ = 'Вид сверху (экран и отражённое поле) - Энергия'

    # # создаём объект точек графика
    # ct = plt.contour(x_ln, y_ln, summar, alpha=0.75, colors='black', linestyles='dotted')
    # # создаём линии уровней из объекта точек
    # plt.clabel(ct, fontsize=10)
    # отрисовка
    plt.imshow(summar, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)
    # раскраска
    plt.colorbar()

    # названия проводов
    for delta_y in [xp_kp, xp_up, xp_nt, xp_ep]:
        plt.hlines(delta_y, Xmin, Xmax, color='black', linewidth=2)
    plt.text(-.5, xp_kp - 0.1, 'КП', color='black')
    plt.text(-.5, xp_up - 0.1, 'УП', color='black')
    plt.text(-.5, xp_nt + 0.4, 'НТ', color='black')
    plt.text(-.5, xp_ep + 0.1, 'ЕП', color='black')

    # очертания кабины
    kab_lines_up()

    # название осей
    plt.xlabel('Ось x, метры')
    plt.ylabel('Ось y, метры')

    plt.title(name_)  # подпись названия
    show(name_)  # вывести и сохранить
    print('График построен.')


# визуализируем вид спереди с учётом отражённой от экрана
def visual_front_reflect(ext_f):
    print('График строится..................')

    # чтобы не возникло проблем при вычитании поля КС и поля отражений,
    # получаем список точек графика из уже рассчитанного ранее поля КС
    y_ln = [el[1][1] for el in ext_f[0]]
    z_ln = [el[0][1][2] for el in ext_f]
    Ymin, Ymax = y_ln[0], y_ln[-1]
    Zmin, Zmax = z_ln[0], z_ln[-1]

    # посчёт отражённого поля
    # посдчёт поля, отражённого от стекла
    refl_glass = np.array([[glass_reflect(x_chel, y_, z_) for y_ in y_ln] for z_ in z_ln])
    # подсчёт поля,. отражённого от стали
    refl_steel = np.array([[steel_reflect(y_, z_) for y_ in y_ln] for z_ in z_ln])
    # суммирование отражённых полей
    summar_reflect = refl_steel + refl_glass

    # перевод в конечные значения внешнего поля с экраном
    summar_ext = np.array([[full_field(ekran(x_el))[2] for x_el in y_list] for y_list in ext_f])
    # вычитаем из поля внешнего поле отражённое
    summar = np.absolute(summar_ext - summar_reflect)

    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name_ = 'Вид спереди (экран и отражённое поле) - Энергия'

    # задаём уровни
    b = len(str(round(np.amax(summar))))  # ручной подсчёт порядка диапазона для отображения линий уровня
    levels = [i * (10 ** j) for j in range(4, b) for i in [1, 2, 5, 7]]  # ограничиваем 4-ой степенью чтобы не было
    # артефактов на границе с экраном
    # создаём объект точек графика
    ct = plt.contour(y_ln, z_ln, summar, alpha=0.75, colors='black', linestyles='dotted',
                     levels=levels)
    # создаём линии уровней из объекта точек
    plt.clabel(ct, fontsize=10)
    # отрисовка
    plt.imshow(summar, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
    # раскраска
    plt.colorbar()

    # названия проводов
    plt.text(xp_kp, h_kp, 'КП', color='black', fontsize=14)
    plt.text(xp_up, h_up, 'УП', color='black', fontsize=14)
    plt.text(xp_nt, h_nt, 'НТ', color='black', fontsize=14)

    # очертания кабины
    fr_kab_lines(star=True)

    # название осей
    plt.xlabel('Ось y, метры')
    plt.ylabel('Ось z, метры')

    plt.title(name_)  # подпись названия
    show(name_)  # вывести и сохранить
    print('График построен.')


# ВЫВОД ПАРАМЕТРОВ
print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print(f'Напряжение: {U} Вольт')
print(f'Суммарный ток: {I} Ампер')
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

print('\nВид сверху для отражённого поля')
visual_up_reflect(cont_f_up)

print('\nВид спереди для отражённого поля')
visual_front_reflect(cont_f_front)

# РАСЧЁТ СТАТИСТИКИ

print('\nСТАТИСТИКА\n')

print('Гармоники энергии поля для человека:')
for f, znach in chel_harm.items():
    print(f, ': %.4f' % znach)

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

chel_f_per = [{fr: [magnetic_calc(y_chel, z_chel, fr), electric_calc(y_chel, z_chel, fr)] for fr in harm.keys()},
              (x_chel, y_chel, z_chel)]
print(chel_f_per)
no_ekran_per = full_field(chel_f_per)[2]
print('\nПеременное поле без экрана: %.4f' % no_ekran_per)

ekran_per = full_field(ekran(chel_f_per))[2]
print('Переменное поле с экраном %.4f' % ekran_per)
Dco = ekran_per * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)
print('Удельная суточная доза облученной энергии: %.4f' % Dco)

plt.show()
