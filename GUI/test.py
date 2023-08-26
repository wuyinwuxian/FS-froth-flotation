import tkinter as tk
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import function.Functions as f
from time import time

realPath = ""

class App:
    def __init__(self, root):
        self.count = 0
        self.entry = []
        self.sv = []
        self.root = root
        self.canvas = Canvas(self.root, background="#ffffff", borderwidth=0, height=300, width=400)
        self.frame = Frame(self.canvas, background="#ffffff")

        self.scrolly = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollx = Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrolly.set)    #, xscrollcommand=self.scrollx.set)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", tags="self.frame")
        self.scrolly.grid(row=0, column=1, sticky='NS')
        self.canvas.grid(row=0, column=0, sticky='NSEW')
        self.scrollx.grid(row=1, column=0, sticky='SWE')
        self.frame.bind("<Configure>", self.onFrameConfigure)
        for i in range(38):
            self.entry.append([])
            self.sv.append([])
            for c in range(20):
                self.sv[i].append(StringVar())
                # self.sv[i][c].trace("w", lambda name, index, mode,sv=self.sv[i][c], i=i, c=c: self.callback(sv, i, c)) # 变量追踪，在写“w”时调用callback
                self.entry[i].append(Entry(self.frame, width=14,justify='center',textvariable=self.sv[i][c]).grid(row=c, column=i))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        table_head = ['1灰度', '2泡沫平均大小', '3色调', '4大小标准差', '5红色均值', '6绿色均值', '7蓝色均值', '8长宽比', '9承载率',
        '10相对红色分量', '11速度', '12陡峭度', '13偏斜度', '14粗度', '15非均匀度', '16二阶矩', '17低频子图粗度',
        '18高频能量', '泡沫大小']
        for col in range(18):
            self.sv[col][0].set(table_head[col])
        self.count = self.count +1

    def onFrameConfigure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def callback(self, sv, column, row):
        print("Column: "+str(column)+", Row: "+str(row)+" = "+sv.get())
    def populate(self,data):
        '''Put in some fake data'''
        for col in range(38):
            self.sv[col][self.count].set(data[col])
        self.count = self.count + 1

def selectPicture():
    global realPath,img

    root = tk.Tk()
    root.withdraw()
    Fpath = filedialog.askopenfilename()  # 我们选择的图片的路径
    path.set(Fpath)                       # 把路径信息写到文本框中
    realPath = Fpath                      # 更新我们的路径

    image = Image.open(realPath)          # 加载图片

    # 图片缩放
    w, h = image.size      # 获取image的长和宽
    mlength = max(w, h)    # 取最大的一边作为缩放的基准
    mul = 400 / mlength    # 缩放倍数
    w1 = int(w * mul)
    h1 = int(h * mul)
    re_image = image.resize((w1, h1))   # 重设图片大小

    img = ImageTk.PhotoImage(re_image)                            # 在canvas中展示图片
    canvas.create_image(200, 150, anchor='center', image=img)     # 定义图片的位置,显示图片，以中心点为锚点（还有其他方式，如左上角为锚点）

def featureExtraction():
    '''
    参数设置
    '''
    # 纹理相关参数，获取灰度频次表的参数，在getgrayfrequency 、getgraymatrix中用到
    d = 1  # 邻域半径
    # 小波分解的参数，仅在 waveletDecomposition 中用到
    w = 'sym4'  # 小波基类型

    velocity = 0    #单张图片没有速度这一特征，置为0
    image = cv2.imread(realPath)
    # t0 = time()
    result = f.getFeatures(image, velocity, d,w)
    # print(time() - t0)
    st = [1,2,6,9,17]
    table.populate(result[st])



def closeWindow():
    window.destroy()  # destroy是注销
    sys.exit(0)       # 退出进程





window = Tk()  # Tk是一个类
window.title("图片特征提取")    # 窗口标题
window.geometry('1000x600')   # 窗口大小
window.geometry('+300+120')   # 窗口位置

#创建一个菜单
menubar = tk.Menu(window)                        # 创建一个大的菜单栏容器，用来放我们的菜单项
filemenu = tk.Menu(menubar, tearoff=0)           # 创建一个空主菜单项（设置为默认不下拉）
menubar.add_cascade(label='File', menu=filemenu) # 将上面定义的空主菜单命名为File，放在菜单栏中，就是装入菜单栏容器
# 在File主菜单下面中加入Open、Exit等小菜单，即我们平时看到的下拉菜单，每一个小菜单对应命令操作。
filemenu.add_command(label='Open', command=selectPicture)
filemenu.add_separator()              # 添加一条分隔线
filemenu.add_command(label='Exit', command=closeWindow)
#假设File主菜单添加完了，我们再添加一个help主菜单
helpmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='关于', command=closeWindow)
helpmenu.add_command(label='联系支持', command=closeWindow)
#创建菜单栏完成后，配置让菜单栏menubar显示出来
window.config(menu=menubar)

# 先分上下两大块
framemain = Frame(window)
framemain.grid(row=0, column=0,sticky='EWNS')
frameup = Frame(framemain)
frameup.grid(row=0, column=0)
Label(framemain, text="    ").grid(row=1, column=0)   #上下之间有点空隙比较好看
framedown = Frame(framemain)
framedown.grid(row=2, column=0)

# 创建一个从文件夹中选择一张图片
path = StringVar()  # 似乎可编辑文本框的必须是这个类型才可以用set,有待查找资料
Label(frameup, text="目标路径：").grid(row=0, column=0)
Entry(frameup, text=path, width=55).grid(row=0, column=1)  # 可编辑文本
btn = Button(frameup, text="选择图片", width=10, height=1, command=selectPicture)  # 第一个按钮
btn.grid(row=0, column=2)

# 下面一块再分左右两大块
frame = Frame(framedown)
frame.grid(row=0,column=0,columnspan=4,sticky='EWSN')
framel = Frame(frame)
framel.grid(row=0, column=0)
Label(frame, text="    ").grid(row=0, column=1)   #左右之间有点空隙比较好看
framer = Frame(frame)
framer.grid(row=0, column=2)

#左边创建一个画布用来显示图片
Label(framel, text="泡沫图像").grid(row=0, column=0)
canvas = tk.Canvas(framel, bg='white', height=300, width=400)  # 画布大小设置，白色背景
canvas.grid(row=0, column=1,columnspan=2)
btn1 = Button(framel, text="特征提取", width=10, height=1, command=featureExtraction)  # 第二个按钮
btn1.grid(row=1, column=1)

#右边创建一个表格显示特征提取的结果
table = App(framer)

# 按钮控制，command作为点击触发的事件

# btn2 = Button(window, text="退出", width=10, height=1, command=closeWindow)  # 第三个按钮
# btn2.grid(row=2, column=2, sticky=E, padx=7)

window.protocol("WM_DELETE_WINDOW", closeWindow)  # protocol()  用户关闭窗口触发的事件
window.mainloop()  # 显示窗口，也叫消息循环

