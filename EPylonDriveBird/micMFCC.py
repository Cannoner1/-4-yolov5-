## mainMFCC.py
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load  # 使用 joblib 直接导入 load
import sounddevice as sd
import soundfile as sf
# 加载音频文件
# 加载音频文件
def load_audio_file(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return signal, sample_rate
def record_audio(duration, fs, channels):#44100 5秒
    
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32')
    
    sd.wait()  # 等待录音完成
    print("Recorded...")
    # 将录音转换为单通道（如果需要）
    if recording.ndim > 1:
        recording = np.mean(recording, axis=1)
    recording, _ = librosa.effects.trim(recording)  # 去除静音部分
    return recording
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"文件 {filename} 删除成功！")
    else:
        print(f"文件 {filename} 不存在。")
def extract_mfcc(audio_data):
    #signal, _ = librosa.load(audio_data, sr=None, mono=True)
    #mfccs = librosa.feature.mfcc(y=signal, fs=44100, n_mfcc=39)
    #mfcc_features = librosa.feature.mfcc()
    #mfcc_features=mfcc_features.astype(np.float32)
    #mfccs = mfcc_features.mfcc(y=audio_data, fs=44100, n_mfcc=39)
    signal, _ = load_audio_file(audio_data)
    mfccs = librosa.feature.mfcc(y=signal, sr=44100, n_mfcc=39)
    #mfccs = np.array(mfccs, np.float32)
    mfccs_scaled_features = np.mean(mfccs.T, axis=0)
    return mfccs_scaled_features
def predictMfcc():
    fs=44100
    filename="temp.wav"
    # 加载模型和标量
    # 假设你有一个名为 'model.pkl' 的训练好的模型文件
    model = load('model.pkl')  # 加载模型
    scaler = load('scaler.pkl')  # 加载标量
    """
    device_index=0
    # 获取麦克风列表
    devices = sd.query_devices()
    # 如果提供了设备索引并且有效，则使用指定的麦克风
    if device_index is not None and device_index < len(devices):
        print(f"Using microphone: {devices[device_index]['name']}")
    else:
        print("Using default microphone.")

    # 获取并设置麦克风支持的采样率
    supported_rates = devices[device_index]['default_samplerate']
    if supported_rates != fs:
        print(f"Adjusting sample rate to {int(supported_rates)} Hz (supported by the device).")
        fs = int(supported_rates)  # 确保采样率是整数
    """
    # 录制音频
    audio_data = record_audio(10, fs, 2)
    # 保存录音为WAV文件
    if(audio_data is not None):
        sf.write(filename, audio_data, fs)

        print("File saved ")
        try:
            #n_fft=1024
            # 提取特征
            features = extract_mfcc(filename)

            # 标准化特征
            features_scaled = scaler.transform([features])

            # 进行预测
            prediction = model.predict(features_scaled)

            # 输出预测结果
            
            print(f"文件名: {filename} --- 结果值: {prediction[0]}")
            return prediction[0]
        except Exception as e:
            print(f"Error processing : {e}")
            return 20
        #delete_file(filename)    
    else:
        print("No Sound Dectected ")
        return 10

if __name__ == "__main__":
    rt=predictMfcc()
