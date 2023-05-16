import tkinter as tk
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import soundfile as sf

def run_train_script():
    output_text.delete(1.0, tk.END)  # 清空文本框
    result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
    output_text.insert(tk.END, result.stdout)  # 显示脚本的输出

def run_infer_script():
    output_text.delete(1.0, tk.END)  # 清空文本框
    result = subprocess.run(["python", "infer.py"], capture_output=True, text=True)
    output_text.insert(tk.END, result.stdout)  # 显示脚本的输出

    # 读取推理后的音频文件
    audio, sample_rate = sf.read("sp1.2.4.pr.wav")

    # 创建一个新的 matplotlib 图形
    fig, axs = plt.subplots(2)
    axs[0].plot(audio)  # 波形图
    axs[1].specgram(audio, NFFT=1024, Fs=sample_rate, noverlap=512)  # 频谱图

    # 在画布上显示图形
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

root = tk.Tk()

train_button = tk.Button(root, text="Run Train Script", command=run_train_script)
train_button.pack()

infer_button = tk.Button(root, text="Run Infer Script", command=run_infer_script)
infer_button.pack()

output_text = tk.Text(root)
output_text.pack()

root.mainloop()
