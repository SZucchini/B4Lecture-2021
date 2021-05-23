import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#2次元散布図
def pdscatter2(df, x, y, save, w_str):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    x:numpy.ndarray
      explanatory variable
    y:numpy.ndarray
      explained variable
    save:str
         save file name
    w_str:str
          regression equation
    """
    #計算値とデータをプロット
    plt.scatter(df["x1"], df["x2"],label="observed value")
    plt.plot(x, y, color="pink",label=w_str)
    
    #ラベル作成
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    plt.grid() 
    plt.legend() #凡例の追加
    plt.savefig(save)
    plt.show()

#3次元散布図
def pdscatter3(df, x, y, z, save, w_str):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    x:numpy.ndarray
      explanatory variable
    y:numpy.ndarray
      explanatory variable
    z:numpy.ndarray
      explained variable
    save:str
         save file name
    w_str:str
          regression equation
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    #計算値とデータをプロット
    ax.plot(df["x1"], df["x2"], df["x3"] , marker="o", linestyle='None', label="observed value")
    ax.plot_wireframe(x, y, z, color="pink", label=w_str)

    #ラベル作成
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    plt.grid()
    plt.legend()  #凡例の追加
    ax.view_init(elev=10, azim=50) #回転
    plt.savefig(save)
    plt.show()

#2次元の回帰式の計算値
def ideal2(df, K, lam):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K:int
      degree of regression equation
    lam:int
        regularization coefficient
      
    returns
    --
    ide_x:np.ndarray
          explanatory variable
    ide_y:np.ndarray
          explained variable
    w_str:str
          regression equation
    """
    #回帰式作成
    w, w_str = regression2(df, K, lam)
    
    #観測値の範囲の等差数列
    ide_x = np.arange(min(df["x1"]), max(df["x1"]), 0.01)
    ide_y = np.zeros(len(ide_x))
    
    #回帰係数をかける
    for i in range(len(ide_x)):       
        for j in range(K+1):
            ide_y[i] += w[j] * (ide_x[i] ** j)

    return ide_x, ide_y, w_str

#3次元の回帰式の計算値
def ideal3(df, K1, K2, lam):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K1 and K2:int
              degree of regression equation
    lam:int
        regularization coefficient
      
    returns
    --
    ide_z:np.ndarray
          explanatory variable
    ide_y:np.ndarray
          explanatory variable
    ide_z:np.ndarray
          explained variable
    w_str:str
          regression equation
          
    """
    #回帰式作成
    w, w_str = regression3(df, K1, K2, lam)
    
    #観測値の範囲の等差数列
    ide_x = np.arange(min(df["x1"]), max(df["x1"]), 0.01)    
    ide_y = np.linspace(min(df["x2"]), max(df["x2"]), len(ide_x))
    ide_z = np.zeros([len(ide_y),len(ide_x)])
    
    mesh_x, mesh_y = np.meshgrid(ide_x, ide_y)
    
    #回帰係数をかける
    for i in range(len(ide_y)):
        for m in range(len(ide_x)):
            for j in range(K1+K2+1):
                if j <= K1:
                    ide_z[i,m] += (w[j] * (ide_x[m] ** j))
                else:
                    ide_z[i,m] += (w[j] * (ide_y[i] ** (j-K1)))
                    
    return mesh_x, mesh_y, ide_z, w_str

#2次元の回帰係数
def regression2(df,K,lam):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K:int
      degree of regression equatio
      
    returns
    --
    w:numpy.ndarray
      regression coefficient
    w_str:str
          regression equation
    """
    #データを変換
    ar_x = np.array(df["x1"])
    ar_y = np.array(df["x2"])
    
    #基底関数行列Φ
    Len_x = len(ar_x)
    phi = np.zeros([Len_x, K+1])
    
    for i in range(Len_x):
        for k in range(K+1):
            phi[i, k] = ar_x[i] ** k
    
    #最小二乗法
    w = np.linalg.inv(phi.T@phi+lam*np.eye(K+1))@phi.T@ar_y   #linalg.inv:逆行列　@:内積 eye:単位行列
    
    #回帰式を文字列で表示
    w_str = ""
    for n in range(len(w)):
        if w[n] < 0:
            w_str = str(w[n])[:7] + "x^" + str(n) + w_str 
        elif w[n] > 0:
            w_str = "+" + str(w[n])[:6] + "x^" + str(n) + w_str
    if w_str[0] == "+":
        w_str = w_str[1:]
    w_str = w_str[:-3]
    
    return w, w_str

#3次元の回帰係数
def regression3(df, K1, K2, lam):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K1 and K2:int
              degree of regression equatio
      
    returns
    --
    w:numpy.ndarray
      regression coefficient
    w_str:str
          regression equation
    """
    #データを変換
    ar_x = np.array(df["x1"])
    ar_y = np.array(df["x2"])
    ar_z = np.array(df["x3"])
    
    #基底関数行列Φ
    Len_x = len(ar_x)
    phi = np.zeros([Len_x, K1 + K2 +1]) #定数項は一つ
    
    for i in range(Len_x):
        for j in range(K1+1):
            phi[i, j] = ar_x[i] ** j
    for l in range(Len_x):
        for m in range(K2):
            phi[l, K1 + m + 1] = ar_y[l] ** (m+1)
    
    
    #最小二乗法
    w = np.linalg.inv(phi.T@phi+lam*np.eye(K1+K2+1))@phi.T@ar_z   #linalg.inv:逆行列　@:内積
    
    #回帰式を文字列で表示
    w_str = ""
    for n in range(len(w)):
        if n <= K1:
            if w[n] < 0:
                w_str = str(w[n])[:7] + "x1^" + str(n) + w_str 
            elif w[n] > 0:
                w_str = "+" + str(w[n])[:6] + "x1^" + str(n) + w_str
        else:
            if w[n] < 0:
                w_str = str(w[n])[:7] + "x2^" + str(n-K1-1) + w_str 
            elif w[n] > 0:
                w_str = "+" + str(w[n])[:6] + "x2^" + str(n-K1) + w_str

    if w_str[0] == "+":
        w_str = w_str[1:]
    w_str = w_str[:-4]
    
    return w, w_str

def main():
    #csvファイルをデータフレームに読み込み
    df1 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data1.csv")
    df2 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data2.csv")
    df3 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data3.csv")
    
    #回帰式の計算値
    x1, y1, w1 = ideal2(df1, 1, 1)
    x2, y2, w2= ideal2(df2, 3, 1)
    x3, y3, z3, w3 = ideal3(df3, 3, 3, 1)
    
    #画像表示&保存
    pdscatter2(df1, x1, y1, "data1_scatter_lam.png", w1)
    pdscatter2(df2, x2, y2, "data2_scatter_lam.png", w2)
    pdscatter3(df3, x3, y3, z3, "data3_scatter_lam.png", w3)

if __name__ == "__main__":
    main()