import packaging
import customtkinter as CTk
from tkinter import *
from tkinter import messagebox, filedialog
import tkinter as Tk
from matplotlib import style
from matplotlib import pyplot as plt
import re
from cat_array import neko, neko_am
import random
from PIL import Image, ImageTk, ImageDraw

CTk.set_appearance_mode('dark')
CTk.set_default_color_theme('dark-blue')

# window
root = CTk.CTk()


class Stick:
    def __init__(self, props: list):
        self.l = props[0]
        self.a = props[1]
        self.e = props[2]
        self.sigma = props[3]

    def get_prop(self):
        return self.l, self.a, self.e, self.sigma


class LoadImage:
    def __init__(self, root, image: str, ld_f: list, ld_q: list, sticks: list):
        frame = Frame(root)
        frame.pack()
        self.canvas = Canvas(frame, width=1920, height=1080, scrollregion=(0, 0, 1920, 1080))

        File = image
        self.orig_img = Image.open(File)
        self.img = ImageTk.PhotoImage(self.orig_img)
        self.canvas.create_image(0, 0, image=self.img, anchor="nw")

        frame.pack(expand=True, fill=BOTH)  # .grid(row=0,column=0)

        self.hbar = Scrollbar(frame, orient=HORIZONTAL, command=self.canvas.xview)
        self.hbar.pack(side=BOTTOM, fill=X)

        self.vbar = Scrollbar(frame, orient=VERTICAL, command=self.canvas.yview)
        self.vbar.pack(side=RIGHT, fill=Y)

        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)

        self.zoomcycle = 0
        self.zimg_id = None

        root.bind("<MouseWheel>", self.zoomer)
        self.canvas.bind("<Motion>", self.crop)

    def zoomer(self, event):
        if event.delta > 0:
            if self.zoomcycle != 4: self.zoomcycle += 1
        elif event.delta < 0:
            if self.zoomcycle != 0: self.zoomcycle -= 1
        self.crop(event)

    def crop(self, event):
        if self.zimg_id: self.canvas.delete(self.zimg_id)
        if self.zoomcycle != 0:
            x, y = event.x, event.y
            if self.zoomcycle == 1:
                tmp = self.orig_img.crop((x - 45, y - 30, x + 45, y + 30))
            elif self.zoomcycle == 2:
                tmp = self.orig_img.crop((x - 30, y - 20, x + 30, y + 20))
            elif self.zoomcycle == 3:
                tmp = self.orig_img.crop((x - 15, y - 10, x + 15, y + 10))
            elif self.zoomcycle == 4:
                tmp = self.orig_img.crop((x - 6, y - 4, x + 6, y + 4))
            size = 300, 200
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            self.zimg_id = self.canvas.create_image(event.x, event.y, image=self.zimg)


# variables
sup_l = CTk.BooleanVar(value=True)
sup_r = CTk.BooleanVar(value=False)
stick_am = 1
sticks = [Stick([0, 0, 0, 0])]
ld_f = []
ld_q = []
filled = []
entries = []  # 0 - L, 1 - A, 2- E, 3 - sigma , 4 - L... == x*4 - L, x*4+1 - A, x*4+2 - E, x*4+3 - sigma
entries_ld_f = []
entries_ld_q = []
tasks = [[3, 2, 1, 1, 2, 3, 1, 1, 0, 26, 0, 0, 0], [1, 3, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 3, 1, 0, 0, 0, 0]]
tasks.append([2, 1, 1, 1, 1, 2, 1, 1, 0, -4, 0, 1, 0])
tasks.append([2, 2, 1, 1, 1, 1, 1, 1, 0, -2, 0, 0, 1])
tasks.append([1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1])
tasks.append([2, 2, 1, 1, 4, 1, 1, 1, 0, 0, 0, 9, -2])
tasks.append([2, 1, 1, 1, 3, 3, 1, 1, 3, 2, 1, 1, 0, 3, -3, 0, -3, 0, 3])
enter_number_ent = CTk.CTkEntry(None)
enter_point_ent = CTk.CTkEntry(None)
tab_step_ent = CTk.CTkEntry(None)
tab_step_index_ent = CTk.CTkEntry(None)
delta, a, e, l, sigma, N, u = [], [], [], [], [], [], []
processed = False
plot_index_ent = CTk.CTkEntry(None)
created = False


def right_s_d(x, y, y_l, draw: ImageDraw):
    correct = 100
    x += correct
    draw.line([x, y, x, y_l], (0, 0, 0))
    draw.line([x + 1, y, x + 1, y_l], (0, 0, 0))
    i = y
    while i < y_l:
        draw.line([x, i, x + 5, i + 5], (0, 0, 0))
        i += y_l / 50
    draw.line([x, y_l, x + 5, y_l + 5], (0, 0, 0))
    return draw


def left_s_d(x, y, y_l, draw: ImageDraw):
    draw.line([x, y, x, y_l], (0, 0, 0))
    draw.line([x + 1, y, x + 1, y_l], (0, 0, 0))
    i = y
    while i < y_l:
        draw.line([x, i, x - 5, i - 5], (0, 0, 0))
        i += y_l / 50
    draw.line([x, y_l, x - 5, y_l - 5], (0, 0, 0))
    return draw


def create_im():
    global ld_f, ld_q, sticks

    image = 'test.jpg'
    image1 = Image.new("RGB", (1920, 1080), (255, 255, 255))
    draw = ImageDraw.Draw(image1)
    # left_s_d(100, 100, 500, draw)
    green = (0, 128, 0)
    red = (255, 0, 0)
    black = (0, 0, 0)

    arr_a = []
    arr_l = []
    for k in range(len(sticks)):
        arr_l.append(sticks[k].l)
        arr_a.append(sticks[k].a)
    l_std = 20000
    a_std = 2000000
    max_w = 1700 / len(sticks)
    max_h = 200
    centre = 400

    max_a = max(arr_l)
    max_b = max(arr_a)
    coef = max_a * l_std / max_w
    coef_2 = max_b * a_std / max_h

    res = [lambda x=x: x / coef for x in arr_l]
    result = []
    for r in res:
        result.append(r())

    res = [lambda x=x: x / coef_2 for x in arr_a]
    result_a = []
    for r in res:
        result_a.append(r())

    min = 110
    min_a = 20
    for r in range(len(arr_l)):
        if result[r] * l_std < min:
            result[r] = min / l_std
    arr_l = result
    for r in range(len(arr_a)):
        if result_a[r] * a_std < min_a:
            result_a[r] = min_a / a_std
    arr_a = result_a

    if sup_l.get():
        left_s_d(100, centre - arr_a[0] / 2 * a_std, centre + arr_a[0] / 2 * a_std, draw)
    if sup_r.get():
        x_s = 0
        for i in range(0, len(sticks)):
            x_s += arr_l[i] * l_std
        right_s_d(x_s, centre - arr_a[len(arr_a) - 1] / 2 * a_std, centre + arr_a[len(arr_a) - 1] / 2 * a_std, draw)
    for i in range(0, len(arr_l)):
        shift = 0
        for k in range(1, i + 1):
            shift += arr_l[i - k] * l_std
        draw.rectangle((100 + shift, centre - (arr_a[i] / 2 * a_std), 100 + shift + (l_std * arr_l[i]),
                        centre + (arr_a[i] / 2 * a_std)), fill=None, outline=black)

    for i in range(len(ld_f)):
        if ld_f[i] > 0:
            shift = 0
            for k in range(1, i + 1):
                shift += arr_l[i - k] * l_std
            draw.line((100 + shift, centre + 1, 100 + shift + 50, centre + 1), green)
            draw.line((100 + shift, centre, 100 + shift + 50, centre), green)
            draw.line((100 + shift + 50, centre, 100 + shift + 45, centre - 5), green)
            draw.line((100 + shift + 50, centre + 1, 100 + shift + 45, centre + 6), green)
        elif ld_f[i] < 0:
            shift = 0
            for k in range(1, i + 1):
                shift += arr_l[i - k] * l_std
            draw.line((100 + shift, centre + 1, 100 + shift - 50, centre + 1), red)
            draw.line((100 + shift, centre, 100 + shift - 50, centre), red)
            draw.line((100 + shift - 50, centre, 100 + shift - 45, centre - 5), red)
            draw.line((100 + shift - 50, centre + 1, 100 + shift - 45, centre + 6), red)

    for i in range(len(ld_q)):
        if ld_q[i] > 0:
            shift = 0
            for k in range(1, i + 1):
                shift += arr_l[i - k] * l_std
            iter = shift + 100
            while iter + 20 < shift + 100 + arr_l[i] * l_std:
                draw.line((iter, centre + 1, iter + 20, centre + 1), green)
                draw.line((iter, centre, iter + 20, centre), green)
                draw.line((iter + 20, centre, iter + 15, centre - 5), green)
                draw.line((iter + 20, centre + 1, iter + 15, centre + 6), green)
                iter += 30
        elif ld_q[i] < 0:
            shift = 0
            for k in range(0, i + 1):
                shift += arr_l[i - k] * l_std
            iter = shift + 100
            if i > 0:
                while iter - 20 > shift + 100 - arr_l[i] * l_std:
                    draw.line((iter, centre + 1, iter - 20, centre + 1), red)
                    draw.line((iter - 20, centre, iter - 20, centre), red)
                    draw.line((iter - 20, centre, iter - 15, centre - 5), red)
                    draw.line((iter - 20, centre + 1, iter - 15, centre + 6), red)
                    iter -= 30
            elif i == 0:
                while iter - 20 > 100:
                    draw.line((iter, centre + 1, iter - 20, centre + 1), red)
                    draw.line((iter - 20, centre, iter - 20, centre), red)
                    draw.line((iter - 20, centre, iter - 15, centre - 5), red)
                    draw.line((iter - 20, centre + 1, iter - 15, centre + 6), red)
                    iter -= 30

    image1.save(image)
    return image


def get_all():
    global stick_am, ld_f, ld_q, entries_ld_q, entries_ld_f, entries, sticks, filled

    sticks = []
    filled = []
    for i in range(stick_am):
        for k in range(4):
            match = re.fullmatch(r'\d*.?\d*(e(-)?[1-9]\d*)?(e-[1-9])?\d*', entries[4 * i + k].get())
            if not match:
                if k == 0:
                    if chance(15):
                        messagebox.showerror(message=f'Некорректный параметр L стрежня {i+1}\n{neko[random.randint(0,neko_am-1)]}')
                    else:
                        messagebox.showerror(message=f'Некорректный параметр L стрежня {i+1}')
                elif k == 1:
                    if chance(15):
                        messagebox.showerror(message=f'Некорректный параметр A стрежня {i+1}\n{neko[random.randint(0,18)]}')
                    else:
                        messagebox.showerror(message=f'Некорректный параметр A стрежня {i+1}')
                elif k == 2:
                    if chance(15):
                        messagebox.showerror(message=f'Некорректный параметр E стрежня {i+1}\n{neko[random.randint(0,neko_am-1)]}')
                    else:
                        messagebox.showerror(message=f'Некорректный параметр E стрежня {i+1}')
                elif k == 3:
                    if chance(15):
                        messagebox.showerror(message=f'Некорректный параметр sigma стрежня {i+1}\n{neko[random.randint(0,neko_am-1)]}')
                    else:
                        messagebox.showerror(message=f'Некорректный параметр sigma стрежня {i+1}')
                return False
        arr_st = []
        for j in range(4):
            arr_st.append(float(entries[4 * i + j].get()))
        sticks.append(Stick(arr_st))
        filled.append(1)

    ld_q = []
    for i in range(stick_am):
        match = re.fullmatch(r'-?\d*.?\d+(e(-)?[1-9]\d*)?(e-[1-9])?\d*', entries_ld_q[i].get())
        if not match:
            if chance(15):
                messagebox.showerror(message=f'Некорректная величина q в стержне {i + 1}\n{neko[random.randint(0,neko_am-1)]}')
            else:
                messagebox.showerror(message=f'Некорректная величина q в стержне {i+1}')
            return False
        ld_q.append(float(entries_ld_q[i].get()))

        ld_f = []
    for i in range(stick_am+1):
        match = re.fullmatch(r'-?\d*.?\d+(e(-)?[1-9]\d*)?(e-[1-9])?\d*', entries_ld_f[i].get())
        if not match:
            if chance(15):
                messagebox.showerror(message=f'Некорректная величина F в узле {i + 1}\n{neko[random.randint(0,neko_am-1)]}')
            else:
                messagebox.showerror(message=f'Некорректная величина F в узле {i+1}')
            return False
        ld_f.append(float(entries_ld_f[i].get()))

    return True


def build_constr():
    global filled, ld_f, ld_q, sticks, sup_l, sup_r, created

    if not created:
        if chance(15):
            messagebox.showerror(title='Ошибка!', message=f'Конструкция не определена!\nПожалуйста, задайте конструкцию\n{neko[random.randint(0,neko_am-1)]}')
        else:
            messagebox.showerror(title='Ошибка!', message='Конструкция не определена!\nПожалуйста, задайте конструкцию')
        return

    if not get_all():
        return

    if sum(filled) != len(filled):
        if chance(15):
            messagebox.showerror(title='Ошибка!', message=f'Не все стержни заданы!\nПожалуйста, задайте все стержни\n{neko[random.randint(0,neko_am-1)]}')
        else:
            messagebox.showerror(title='Ошибка!', message='Не все стержни заданы!\nПожалуйста, задайте все стержни')
        return
    elif sup_l.get() == 0 and sup_r.get() == 0:
        if chance(15):
            messagebox.showerror(title='Ошибка!', message=f'Должна быть хотя бы 1 опора\n{neko[random.randint(0,neko_am-1)]}')
        else:
            messagebox.showerror(title='Ошибка!', message='Должна быть хотя бы 1 опора')
        return

    constr_w = Toplevel()
    constr_w.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

    img = create_im()

    App = LoadImage(constr_w, img, ld_f, ld_q, sticks)
    constr_w.mainloop()


def proc():
    global stick_am, ld_f, ld_q, sticks, sup_l, sup_r, delta, a, e, l, sigma, N, u, processed

    get_all()

    if sup_l.get() == 0 and sup_r.get() == 0:
        if chance(15):
            messagebox.showerror(title='Ошибка!', message=f'Должна быть хотя бы 1 опора\n{neko[random.randint(0,neko_am-1)]}')
        else:
            messagebox.showerror(title='Ошибка!', message='Должна быть хотя бы 1 опора')
        return

    l = []
    a = []
    e = []
    sigma = []
    delta = []
    for i in range(len(sticks)):
        l.append(sticks[i].l)
        a.append(sticks[i].a)
        e.append(sticks[i].e)
        sigma.append(sticks[i].sigma)
    l.append(1)
    a.append(0)
    e.append(0)
    A = []
    for i in range(stick_am+1):
        A.append([0]*(stick_am+1))
    for i in range(0, stick_am):
        A[i][i] += e[i]*a[i]/l[i]
        A[i][i+1] = -e[i]*a[i]/l[i]
        A[i+1][i] = -e[i]*a[i]/l[i]
        A[i+1][i+1] = e[i]*a[i]/l[i]
    if sup_l.get():
        A[0][0] = 1
        A[0][1] = 0
        A[1][0] = 0
    if sup_r.get():
        A[stick_am][stick_am] = 1
        A[stick_am-1][stick_am] = 0
        A[stick_am][stick_am-1] = 0

    l.pop(-1)
    a.pop(-1)
    e.pop(-1)
    b = []
    b.append(ld_f[0] + ld_q[0]*l[0]/2)
    for i in range(1, stick_am):
        b.append(ld_f[i]+ld_q[i]*l[i]/2+ld_q[i-1]*l[i-1]/2)
    b.append(ld_f[stick_am] + ld_q[stick_am-1]*l[stick_am-1]/2)
    if sup_l.get():
        b[0] = 0
    if sup_r.get():
        b[stick_am] = 0

    delta = gauss_method(A, b)
    print('delta: ', delta)
    u = []
    for i in range(stick_am):
        u.append([delta[i], delta[i+1]])
    N = []
    for i in range(stick_am):
        N.append([find_N(0, l[i], a[i], e[i], u[i], ld_q[i]), find_N(l[i], l[i], a[i], e[i], u[i], ld_q[i])])
    print('N: ', N)
    processed = True
    return


def find_N(l, L, a, e, u, q):
    return (e * a * (u[1] - u[0]) / L) + (q * L * (1 - 2 * l / L)) / 2


def find_u(l, L, a, e, u, q):
    return (1 - l/L)*u[0] + l*u[1]/L + (q * L**2 * l / (2 * e * a * L)) * (1 - l/L)


def gauss_method(matrix, vector):
    n = len(matrix)
    for i in range(n):
        max_el = abs(matrix[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > max_el:
                max_el = abs(matrix[k][i])
                max_row = k

        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
        vector[i], vector[max_row] = vector[max_row], vector[i]

        for k in range(i + 1, n):
            c = -matrix[k][i] / matrix[i][i]
            for j in range(i, n):
                if i == j:
                    matrix[k][j] = 0
                else:
                    matrix[k][j] += c * matrix[i][j]
            vector[k] += c * vector[i]

    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = vector[i] / matrix[i][i]
        for k in range(i - 1, -1, -1):
            vector[k] -= matrix[k][i] * x[i]
    return x


def am_acc():
    global stick_am, processed

    processed = False

    match = re.fullmatch(r'[1-9]{1}\d*', amount_st.get())
    if match:
        stick_am = int(amount_st.get())
        create_st(stick_am)
        create_lds(stick_am)
    else:
        if chance(15):
            messagebox.showerror(message=f'Некорректное количество стрежней\n{neko[random.randint(0,neko_am-1)]}')
        else:
            messagebox.showerror(message='Некорректное количество стрежней')
        amount_st.delete(0, END)
        return


def create_st(amount, task=0):
    global entries, tasks

    frame_st = CTk.CTkScrollableFrame(root)  # frame for sticks' properties
    frame_st.place(relx=0, rely=0.15, relwidth=1, relheight=0.45)

    l_label = CTk.CTkLabel(frame_st, text='L')
    l_label.grid(row=0, column=0)
    a_label = CTk.CTkLabel(frame_st, text='A')
    a_label.grid(row=0, column=1)
    e_label = CTk.CTkLabel(frame_st, text='E')
    e_label.grid(row=0, column=2)
    sigma_label = CTk.CTkLabel(frame_st, text='sigma')
    sigma_label.grid(row=0, column=3)

    entries = []
    if task:
        for i in range(stick_am):
            value_l = CTk.CTkEntry(frame_st)
            value_l.insert(END, tasks[task-1][0 + 4*i])
            entries.append(value_l)
            value_l.grid(row=i+1, column=0)
            value_a = CTk.CTkEntry(frame_st)
            value_a.insert(END, tasks[task-1][1 + 4*i])
            entries.append(value_a)
            value_a.grid(row=i+1, column=1)
            value_e = CTk.CTkEntry(frame_st)
            value_e.insert(END, tasks[task-1][2 + 4*i])
            entries.append(value_e)
            value_e.grid(row=i+1, column=2)
            value_sigma = CTk.CTkEntry(frame_st)
            value_sigma.insert(END, tasks[task-1][3 + 4*i])
            entries.append(value_sigma)
            value_sigma.grid(row=i+1, column=3)
    else:
        for i in range(1, amount + 1):
            value_l = CTk.CTkEntry(frame_st)
            entries.append(value_l)
            value_l.grid(row=i, column=0)
            value_a = CTk.CTkEntry(frame_st)
            entries.append(value_a)
            value_a.grid(row=i, column=1)
            value_e = CTk.CTkEntry(frame_st)
            entries.append(value_e)
            value_e.grid(row=i, column=2)
            value_sigma = CTk.CTkEntry(frame_st)
            entries.append(value_sigma)
            value_sigma.grid(row=i, column=3)


def create_lds(amount, task=0):
    global entries_ld_f, entries_ld_q, ld_f, ld_q, created

    frame_ld_f = CTk.CTkScrollableFrame(root)  # frame for sticks' properties
    frame_ld_f.place(relx=0, rely=0.6, relwidth=0.5, relheight=0.35)

    frame_ld_q = CTk.CTkScrollableFrame(root)  # frame for sticks' properties
    frame_ld_q.place(relx=0.5, rely=0.6, relwidth=0.5, relheight=0.35)

    s_label = CTk.CTkLabel(frame_ld_f, text='sticks')
    s_label.grid(row=0, column=0)
    f_label = CTk.CTkLabel(frame_ld_f, text='F')
    f_label.grid(row=0, column=1)
    n_label = CTk.CTkLabel(frame_ld_q, text='nodes')
    n_label.grid(row=0, column=0)
    q_label = CTk.CTkLabel(frame_ld_q, text='q')
    q_label.grid(row=0, column=1)

    entries_ld_f = []
    entries_ld_q = []
    created = True

    if task:
        for i in range(1, amount+1):
            q_st = CTk.CTkLabel(frame_ld_q, text=f"stick_{i}")
            q_st.grid(row=i, column=0)
            value_q = CTk.CTkEntry(frame_ld_q)
            value_q.insert(END, tasks[task-1][stick_am*4 + stick_am + i])
            entries_ld_q.append(value_q)
            value_q.grid(row=i, column=1)

        for j in range(1, amount + 2):
            f_st = CTk.CTkLabel(frame_ld_f, text=f"node_{j}")
            f_st.grid(row=j, column=0)
            value_f = CTk.CTkEntry(frame_ld_f)
            value_f.insert(END, tasks[task-1][4*stick_am + j-1])
            entries_ld_f.append(value_f)
            value_f.grid(row=j, column=1)
    else:
        for i in range(1, amount + 1):
            q_st = CTk.CTkLabel(frame_ld_q, text=f"stick_{i}")
            q_st.grid(row=i, column=0)
            value_q = CTk.CTkEntry(frame_ld_q)
            value_q.insert(END, 0)
            entries_ld_q.append(value_q)
            value_q.grid(row=i, column=1)

        for j in range(1, amount + 2):
            f_st = CTk.CTkLabel(frame_ld_f, text=f"node_{j}")
            f_st.grid(row=j, column=0)
            value_f = CTk.CTkEntry(frame_ld_f)
            value_f.insert(END, 0)
            entries_ld_f.append(value_f)
            value_f.grid(row=j, column=1)


def combo_task(task):
    global stick_am, sup_l, sup_r, processed
    processed = False
    if task == 'custom':
        pass
    elif task == 'task 1':
        stick_am = 2
        sup_l = CTk.BooleanVar(value=True)
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=True)
        check_right.configure(variable=sup_r)
        create_st(stick_am, int(re.findall(r'[1-9]', task)[0]))
        create_lds(stick_am, int(re.findall(r'[1-9]', task)[0]))
    elif task == 'task 2':
        stick_am = 3
        sup_l = CTk.BooleanVar(value=True)
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=True)
        check_right.configure(variable=sup_r)
        create_st(stick_am, int(re.findall(r'[1-9]', task)[0]))
        create_lds(stick_am, int(re.findall(r'[1-9]', task)[0]))
    elif task == 'task 3':
        stick_am = 2
        sup_l = CTk.BooleanVar(value=True)
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=True)
        check_right.configure(variable=sup_r)
        create_st(stick_am, int(re.findall(r'[1-9]', task)[0]))
        create_lds(stick_am, int(re.findall(r'[1-9]', task)[0]))
    elif task == 'task 4':
        sup_l = CTk.BooleanVar(value=True)
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=False)
        check_right.configure(variable=sup_r)
        stick_am = 2
        create_st(stick_am, int(re.findall(r'[1-9]', task)[0]))
        create_lds(stick_am, int(re.findall(r'[1-9]', task)[0]))
    elif task == 'task 5':
        sup_l = CTk.BooleanVar(value=True)
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=True)
        check_right.configure(variable=sup_r)
        stick_am = 4
        create_st(stick_am, int(re.findall(r'[1-9]', task)[0]))
        create_lds(stick_am, int(re.findall(r'[1-9]', task)[0]))
    elif task == 'task 6':
        sup_l = CTk.BooleanVar(value=True)
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=True)
        check_right.configure(variable=sup_r)
        stick_am = 2
        create_st(stick_am, int(re.findall(r'[1-9]', task)[0]))
        create_lds(stick_am, int(re.findall(r'[1-9]', task)[0]))
    elif task == 'own':
        sup_l = CTk.BooleanVar(value=True)
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=True)
        check_right.configure(variable=sup_r)
        stick_am = 3
        create_st(stick_am, 7)
        create_lds(stick_am, 7)


def find_point():
    global enter_number_ent, stick_am, enter_point_ent, delta, a, e, l, sigma, N, u, ld_f, ld_q

    match = re.fullmatch(r'[1-9]\d*', enter_number_ent.get())
    if not match:
        if chance(15):
            messagebox.showerror(message=f'Некорректный номер стрежня\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message='Некорректный номер стрежня')
        return

    if not int(enter_number_ent.get()) > 0 or int(enter_number_ent.get()) > stick_am:  # 0 < a <= amount
        if chance(15):
            messagebox.showerror(message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}')
        return

    match = re.fullmatch(r'\d+(\.\d+)?', enter_point_ent.get())
    if not match:
        if chance(15):
            messagebox.showerror(message=f'Некорректная точка стрежня\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message='Некорректная точка стрежня')
        return

    index = int(enter_number_ent.get())
    if not float(enter_point_ent.get()) >= 0 or float(enter_point_ent.get()) > float(entries[(index-1)*4].get()):
        if chance(15):
            messagebox.showerror(message=f'Некорректная точка стержня! Длина стержня: {entries[(index - 1) * 4].get()}\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message=f'Некорректная точка стержня! Длина стержня: {entries[(index-1)*4].get()}')
        return
    point = float(enter_point_ent.get())

    val_point_found = Toplevel()
    val_point_found.geometry("500x300")

    frame_val_found = CTk.CTkFrame(val_point_found)
    frame_val_found.place(relx=0, rely=0, relwidth=1, relheight=1)

    n_found = find_N(point, l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1])
    val_n_txt = CTk.CTkLabel(frame_val_found, text=f'N: {n_found}')
    val_n_txt.place(relx=0.3, rely=0.2)
    sigma_found = abs(n_found / a[index-1])
    val_sigma_txt = CTk.CTkLabel(frame_val_found, text=f'Sigma: {sigma_found}')
    val_sigma_txt.place(relx=0.3, rely=0.4)
    u_found = find_u(point, l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1])
    val_u_txt = CTk.CTkLabel(frame_val_found, text=f'u: {u_found}')
    val_u_txt.place(relx=0.3, rely=0.5)

def get_value_point():
    global enter_number_ent, enter_point_ent, amount_st, processed
    if not processed:
        if chance(15):
            messagebox.showerror(message=f'Конструкция еще не обработана в процессоре!\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message=f'Конструкция еще не обработана в процессоре!')
        return
    val_point = Toplevel()
    val_point.geometry("500x300")

    frame_val = CTk.CTkFrame(val_point)
    frame_val.place(relx=0, rely=0, relwidth=1, relheight=1)

    enter_txt_number = CTk.CTkLabel(frame_val, text='Введите номер стержня:')
    enter_txt_number.grid(row=0, column=0)
    enter_number_ent = CTk.CTkEntry(frame_val)
    enter_number_ent.grid(row=0, column=1)
    enter_txt_point = CTk.CTkLabel(frame_val, text='Введите точку:')
    enter_txt_point.grid(row=0, column=3)
    enter_point_ent = CTk.CTkEntry(frame_val)
    enter_point_ent.grid(row=0, column=4)
    find_btn = CTk.CTkButton(frame_val, text='Рассчитать', command=find_point)
    find_btn.grid(row=1, column=1)

    val_point.mainloop()


def chance(prob):
    return random.randint(0, 100) < prob+1


def tab_step():
    global processed, tab_step_index_ent, tab_step_ent

    if not processed:
        if chance(15):
            messagebox.showerror(message=f'Конструкция еще не обработана в процессоре!\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message=f'Конструкция еще не обработана в процессоре!')
        return

    tab_step_input = Toplevel()
    tab_step_input.geometry("500x300")

    frame_step = CTk.CTkFrame(tab_step_input)
    frame_step.place(relx=0, rely=0, relwidth=1, relheight=1)

    tab_step_index = CTk.CTkLabel(frame_step, text='Введите номер стержня:')
    tab_step_index.place(relx=0.18, rely=0.3)
    tab_step_index_ent = CTk.CTkEntry(frame_step)
    tab_step_index_ent.place(relx=0.5, rely=0.3)
    tab_step_label = CTk.CTkLabel(frame_step, text='Введите шаг таблицы:')
    tab_step_label.place(relx=0.2, rely=0.5)
    tab_step_ent = CTk.CTkEntry(frame_step)
    tab_step_ent.place(relx=0.5, rely=0.5)
    step_btn = CTk.CTkButton(frame_step, text='Рассчитать', command=tab_step_output)
    step_btn.place(relx=0.5, rely=0.7)

    tab_step_input.mainloop()


def tab_step_output():
    global tab_step_index_ent, tab_step_ent, entries, stick_am, delta, a, e, l, sigma, N, u, ld_f, ld_q

    match = re.fullmatch(r'[1-9]\d*', tab_step_index_ent.get())
    if not match:
        if chance(15):
            messagebox.showerror(message=f'Некорректный номер стрежня\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message='Некорректный номер стрежня')
        return

    if not int(tab_step_index_ent.get()) > 0 or int(tab_step_index_ent.get()) > stick_am:  # 0 < a <= amount
        if chance(15):
            messagebox.showerror(
                message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}')
        return

    match = re.fullmatch(r'\d+(\.\d+)?', tab_step_ent.get())
    if not match:
        if chance(15):
            messagebox.showerror(message=f'Некорректная точка стрежня\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message='Некорректная точка стрежня')
        return
    index = int(tab_step_index_ent.get())
    if not float(tab_step_ent.get()) >= 0 or float(tab_step_ent.get()) > float(entries[(index - 1) * 4].get()):
        if chance(15):
            messagebox.showerror(message=f'Некорректный шаг сечений стержня! Длина стержня: {entries[(index - 1) * 4].get()}\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message=f'Некорректный шаг сечений стержня! Длина стержня: {entries[(index - 1) * 4].get()}')
        return

    i = 0
    i_arr = []
    amount = 0
    n_tab = []
    u_tab = []
    sigma_tab = []
    while i < float(entries[(index - 1) * 4].get()):
        n_tab.append(find_N(i, l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1]))
        u_tab.append(find_u(i, l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1]))
        sigma_tab.append(abs(n_tab[amount] / a[index-1]))
        i_arr.append(i)
        i += float(tab_step_ent.get())
        amount += 1
    n_tab.append(find_N(i, l[index - 1], a[index - 1], e[index - 1], u[index - 1], ld_q[index - 1]))
    u_tab.append(find_u(i, l[index - 1], a[index - 1], e[index - 1], u[index - 1], ld_q[index - 1]))
    sigma_tab.append(abs(n_tab[amount] / a[index - 1]))
    i_arr.append(i)
    amount += 1

    tab_step_output = Toplevel()
    tab_step_output.geometry('1000x800')

    frame_tab = CTk.CTkScrollableFrame(tab_step_output)
    frame_tab.place(relx=0, rely=0, relwidth=1, relheight=1)

    output_str_label = ''
    output_str_label += ' x        N       u    sigma \n'
    output_str = ''
    output_sigma_max = ''
    for j in range(amount):
        output_str += f' {round(i_arr[j], 4)} '
        output_str += f'  {round(n_tab[j], 4)} '
        output_str += f'     {round(u_tab[j], 4)} '
        output_str += f'     {round(sigma_tab[j], 4)} \n'
        if sigma_tab[j] > sigma[index - 1]:
            output_sigma_max += '    Напряжение превышает максимальное допустимое\n'
        else:
            output_sigma_max += '\n'
    tab_label = CTk.CTkLabel(tab_step_output, anchor='nw', text=output_str_label, fg_color=('#212121', '#212121'))
    tab_label.place(relx=0, rely=0, relwidth=1, relheight=1)
    tab_label_output = CTk.CTkLabel(tab_step_output, anchor='nw', text=output_str, fg_color=('#212121', '#212121'))
    tab_label_output.place(relx=0, rely=0.02, relwidth=1, relheight=1)
    tab_label_output = CTk.CTkLabel(tab_step_output, anchor='nw', text=output_sigma_max, fg_color=('#212121', '#212121'))
    tab_label_output.place(relx=0.2, rely=0.02, relwidth=1, relheight=1)

    return


def plot_maker():
    global processed, plot_index_ent, plot_ent

    if not processed:
        if chance(15):
            messagebox.showerror(message=f'Конструкция еще не обработана в процессоре!\n{neko[random.randint(0, neko_am-1)]}')
        else:
            messagebox.showerror(message=f'Конструкция еще не обработана в процессоре!')
        return

    plot_input = Toplevel()
    plot_input.geometry("500x300")

    frame_plot = CTk.CTkFrame(plot_input)
    frame_plot.place(relx=0, rely=0, relwidth=1, relheight=1)

    plot_index = CTk.CTkLabel(frame_plot, text='Введите номер стержня:')
    plot_index.place(relx=0.18, rely=0.3)
    plot_index_ent = CTk.CTkEntry(frame_plot)
    plot_index_ent.place(relx=0.5, rely=0.3)
    plot_label = CTk.CTkLabel(frame_plot, text='Выберите компоненту:')
    plot_label.place(relx=0.2, rely=0.5)
    plot_comp_names = ['N', 'u', 'sigma с максимально допустимым']
    plot_comp_names = CTk.CTkComboBox(frame_plot, values=plot_comp_names, command=plot_output)
    plot_comp_names.place(relx=0.5, rely=0.5)

    plot_input.mainloop()


def plot_output(plot_type):
    global plot_index_ent, stick_am, entries, delta, a, e, l, sigma, N, u, ld_f, ld_q

    style.use('seaborn-v0_8')
    if plot_type == 'N':

        match = re.fullmatch(r'[1-9]\d*', plot_index_ent.get())
        if not match:
            if chance(15):
                messagebox.showerror(message=f'Некорректный номер стрежня\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(message='Некорректный номер стрежня')
            return

        index = int(plot_index_ent.get())
        if not index > 0 or index > stick_am:  # 0 < a <= amount
            if chance(15):
                messagebox.showerror(
                    message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(
                    message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}')
            return

        zeros = []
        n_plot = []
        i = 0
        i_arr = []
        while i < float(entries[(index - 1) * 4].get()):
            n_plot.append(find_N(i, l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1]))
            i_arr.append(i)
            zeros.append(0)
            i += float(entries[(index - 1) * 4].get()) / 20
        n_plot.append(find_N(float(entries[(index - 1) * 4].get()), l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1]))
        i_arr.append(float(entries[(index - 1) * 4].get()))
        zeros.append(0)
        fig, ax = plt.subplots()
        ax.plot(i_arr, n_plot, label='N')
        ax.plot(i_arr, zeros, label='zero')
        ax.grid(True)
        plt.legend()
        plt.show()
    elif plot_type == 'u':

        match = re.fullmatch(r'[1-9]\d*', plot_index_ent.get())
        if not match:
            if chance(15):
                messagebox.showerror(message=f'Некорректный номер стрежня\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(message='Некорректный номер стрежня')
            return

        index = int(plot_index_ent.get())
        if not index > 0 or index > stick_am:  # 0 < a <= amount
            if chance(15):
                messagebox.showerror(
                    message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(
                    message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}')
            return

        zeros = []
        u_plot = []
        i = 0
        i_arr = []
        while i < float(entries[(index - 1) * 4].get()):
            u_plot.append(find_u(i, l[index - 1], a[index - 1], e[index - 1], u[index - 1], ld_q[index - 1]))
            i_arr.append(i)
            zeros.append(0)
            i += float(entries[(index - 1) * 4].get()) / 20
        u_plot.append(
            find_u(float(entries[(index - 1) * 4].get()), l[index - 1], a[index - 1], e[index - 1], u[index - 1],
                   ld_q[index - 1]))
        i_arr.append(float(entries[(index - 1) * 4].get()))
        zeros.append(0)
        fig, ax = plt.subplots()
        ax.plot(i_arr, u_plot, label='u')
        ax.plot(i_arr, zeros, label='zero')
        ax.grid(True)
        plt.legend()
        plt.show()
    elif plot_type == 'sigma с максимально допустимым':

        match = re.fullmatch(r'[1-9]\d*', plot_index_ent.get())
        if not match:
            if chance(15):
                messagebox.showerror(message=f'Некорректный номер стрежня\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(message='Некорректный номер стрежня')
            return

        index = int(plot_index_ent.get())
        if not index > 0 or index > stick_am:  # 0 < a <= amount
            if chance(15):
                messagebox.showerror(
                    message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(
                    message=f'Некорректный номер стержня! Количество стержней в конструкции: {stick_am}')
            return

        zeros = []
        sigma_plot = []
        n_sigma_plot_plus = []
        n_sigma_plot_minus = []
        i = 0
        i_arr = []
        while i < float(entries[(index - 1) * 4].get()):
            sigma_plot.append(find_N(i, l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1]) / a[index-1])
            i_arr.append(i)
            zeros.append(0)
            n_sigma_plot_plus.append(sigma[index-1])
            n_sigma_plot_minus.append(-(sigma[index - 1]))
            i += float(entries[(index - 1) * 4].get()) / 20
        sigma_plot.append(find_N(float(entries[(index - 1) * 4].get()), l[index-1], a[index-1], e[index-1], u[index-1], ld_q[index-1]) / a[index-1])
        i_arr.append(float(entries[(index - 1) * 4].get()))
        zeros.append(0)
        n_sigma_plot_plus.append(sigma[index - 1])
        n_sigma_plot_minus.append(-(sigma[index - 1]))
        fig, ax = plt.subplots()
        ax.plot(i_arr, sigma_plot, label='sigma')
        ax.plot(i_arr, n_sigma_plot_plus, label='sigma_plus')
        ax.plot(i_arr, n_sigma_plot_minus, label='sigma_minus')
        ax.plot(i_arr, zeros, label='zero')
        ax.grid(True)
        plt.legend()
        plt.show()


def load_constr():
    global sup_l, sup_r, stick_am, tasks, processed

    load_file = filedialog.askopenfilename()
    if load_file == '':
        return
    match = re.findall(r'\.txt', load_file)
    if not match:
        if chance(15):
            messagebox.showerror(
                message=f'Некорректный формат файла! Корректный формат - .txt\n{neko[random.randint(0, neko_am - 1)]}')
        else:
            messagebox.showerror(
                message=f'Некорректный формат файла! Корректный формат - .txt')
        return
    with open(load_file, 'r', encoding='utf8') as f:
        temp_str = f.readline()
        temp_str = temp_str[:-1]
        match = re.findall(r'\d+\s|\d+\Z', temp_str)
        if len(match) != 2:
            if chance(15):
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 1!\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 1!')
            return
        sup_l = CTk.BooleanVar(value=bool(int(match[0])))
        check_left.configure(variable=sup_l)
        sup_r = CTk.BooleanVar(value=bool(int(match[1])))
        check_right.configure(variable=sup_r)
        temp_str = f.readline()
        match = re.findall(r'\d+\s|\d+\.\d+\s', temp_str)
        if len(match) % 4 != 0:
            if chance(15):
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 2!\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 2!')
            return
        file_task = []
        for i in range(len(match)):
            file_task.append(float(match[i]))
        stick_amount_file = int(len(file_task) / 4)
        temp_str = f.readline()
        match = re.findall(r'-?\d+\s|-?\d+\.\d+\s', temp_str)
        if len(match) != stick_amount_file + 1:
            if chance(15):
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 3!\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 3!')
            return
        for i in range(len(match)):
            file_task.append(float(match[i]))
        temp_str = f.readline()
        match = re.findall(r'-?\d+\s|-?\d+\.\d+\s', temp_str)
        if len(match) != stick_amount_file:
            if chance(15):
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 4!\n{neko[random.randint(0, neko_am - 1)]}')
            else:
                messagebox.showerror(
                    message=f'Некорректный формат записи в файле в строке 4!')
            return
        for i in range(len(match)):
            file_task.append(float(match[i]))

    stick_am = stick_amount_file
    task = len(tasks) + 1
    processed = False
    tasks.append(file_task)
    create_st(stick_am, task)
    create_lds(stick_am, task)


def save_constr():
    global created, sup_l, sup_r, entries, entries_ld_f, entries_ld_q

    if not created:
        if chance(15):
            messagebox.showerror(title='Ошибка!', message=f'Конструкция не определена, чтобы ее сохранить!\nПожалуйста, задайте конструкцию\n{neko[random.randint(0,neko_am-1)]}')
        else:
            messagebox.showerror(title='Ошибка!', message='Конструкция не определена, чтобы ее сохранить!\nПожалуйста, задайте конструкцию')
        return
    save_file = filedialog.asksaveasfilename()
    if save_file == '':
        return
    with open(save_file, 'w', encoding='utf8') as f:
        f.write(f'{int(sup_l.get())}  {int(sup_r.get())}\n')
        for i in range(len(entries)):
            f.write(f'{entries[i].get()} ')
        f.write('\n')
        for i in range(len(entries_ld_f)):
            f.write(f'{entries_ld_f[i].get()} ')
        f.write('\n')
        for i in range(len(entries_ld_q)):
            f.write(f'{entries_ld_q[i].get()} ')
        f.write('\n')


root.title('Settings of the construction')
root.geometry('1000x800')

canvas = Canvas(root, height=1000, width=800)
canvas.pack()

frame_files = CTk.CTkFrame(root)  # text for supports
frame_files.place(relx=0, rely=0.95, relwidth=1, relheight=0.05)

frame_btns = CTk.CTkFrame(root)
frame_btns.place(relx=0, rely=0, relwidth=1, relheight=0.05)  # 0.05

frame_task = CTk.CTkFrame(root)
frame_task.place(relx=0, rely=0.05, relwidth=1, relheight=0.05)  # 0.1

frame_txt_st = CTk.CTkFrame(root)
frame_txt_st.place(relx=0, rely=0.1, relwidth=1, relheight=0.05)  # 0.15

# in frame buttons
btn_build = CTk.CTkButton(frame_btns, text='Построить конструкцию', command=build_constr)
btn_build.place(relx=0.01, rely=0.3)
proc_btn = CTk.CTkButton(frame_btns, text='Процессор', command=proc)
proc_btn.place(relx=0.2, rely=0.3)
val_in_point_btn = CTk.CTkButton(frame_btns, text='Получить значение в точке', command=get_value_point)
val_in_point_btn.place(relx=0.37, rely=0.3)
tabs_btn = CTk.CTkButton(frame_btns, text='Таблица с шагом', command=tab_step)
tabs_btn.place(relx=0.6, rely=0.3)
tabs_btn = CTk.CTkButton(frame_btns, text='Построить графики', command=plot_maker)
tabs_btn.place(relx=0.8, rely=0.3)

#in frame_txt_st
title_st = CTk.CTkLabel(frame_txt_st, text='Введите количество стержней')
title_st.place(relx=0.3, rely=0.3)
amount_st = CTk.CTkEntry(frame_txt_st)
amount_st.place(relx=0.6, rely=0.3, relwidth=0.1)
amount_acc = CTk.CTkButton(frame_txt_st, text='Ввести', command=am_acc)
amount_acc.place(relx=0.8, rely=0.3, relwidth=0.1)

# in frame_task
title_st = CTk.CTkLabel(frame_task, text='Выберите задачу')
title_st.place(relx=0.5, rely=0.3)
tasks_names = ['custom', 'task 1', 'task 2', 'task 3', 'task 4', 'task 5', 'task 6', 'own']
combo_tasks = CTk.CTkComboBox(frame_task, values=tasks_names, command=combo_task)
combo_tasks.place(relx=0.7, rely=0.3)
check_left = CTk.CTkCheckBox(frame_task, text="Опора слева", hover_color='green', variable=sup_l)
check_left.place(relx=0.1, rely=0.3)
check_right = CTk.CTkCheckBox(frame_task, text="Опора справа", hover_color='green', variable=sup_r)
check_right.place(relx=0.3, rely=0.3)

# in frame_files
load_btn = CTk.CTkButton(frame_files, text='Загрузить конструкцию', command=load_constr)
load_btn.place(relx=0.25, rely=0.1)
save_btn = CTk.CTkButton(frame_files, text='Сохранить конструкцию', command=save_constr)
save_btn.place(relx=0.6, rely=0.1)

root.mainloop()
