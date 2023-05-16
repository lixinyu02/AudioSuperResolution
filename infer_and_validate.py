import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from model import SpectralSuperResolution
from sklearn.metrics import mean_squared_error

def infer(lr_file, hr_file, model_file, pr_file):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralSuperResolution()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    # 读取低分辨率音频
    lr_audio, sample_rate = sf.read(lr_file)
    if lr_audio.ndim > 1:  # 如果音频是立体声的，将其转换为单声道
        lr_audio = np.mean(lr_audio, axis=1)
    lr_audio = torch.from_numpy(lr_audio).float().unsqueeze(0).unsqueeze(0)  # 添加一个批次维度和一个通道维度

    # 在音频上运行模型
    with torch.no_grad():
        lr_audio = lr_audio.to(device)
        pr_audio = model(lr_audio)
        pr_audio = pr_audio.cpu().numpy().squeeze()

    # 将预测的音频保存到文件
    sf.write(pr_file, pr_audio, sample_rate)

    # 读取高分辨率音频
    hr_audio, _ = sf.read(hr_file)
    if hr_audio.ndim > 1:  # 如果音频是立体声的，将其转换为单声道
        hr_audio = np.mean(hr_audio, axis=1)

    # 计算 MSE
    mse = mean_squared_error(hr_audio, pr_audio)

    # 绘制波形图
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(lr_audio.squeeze().numpy())
    plt.title('Original Audio')
    plt.subplot(1, 3, 2)
    plt.plot(pr_audio)
    plt.title('Predicted Audio')
    plt.subplot(1, 3, 3)
    plt.plot(hr_audio)
    plt.title('High Resolution Audio')
    plt.tight_layout()
    plt.savefig('waveform.png')

    # 绘制频谱图
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    f, t, Sxx = signal.spectrogram(lr_audio.squeeze().numpy(), sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.title('Original Audio')
    plt.subplot(1, 3, 2)
    f, t, Sxx = signal.spectrogram(pr_audio, sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.title('Predicted Audio')
    plt.subplot(1, 3, 3)
    f, t, Sxx = signal.spectrogram(hr_audio, sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.title('High Resolution Audio')
    plt.tight_layout()
    plt.savefig('spectrogram.png')

    print('MSE:', mse)


infer("piano.1.4.lr.wav","piano.1.4.hr.wav", "model.pth", "piano.1.4.pr.wav")
