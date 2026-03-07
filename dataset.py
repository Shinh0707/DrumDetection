# 120bpm 4/4 1note/mesure C1,C#1,D1,E1,F1,F#1,G1,G#1,A1,A#1,B1,C2,C#2,D2,D#2
# Melspectrogram (PyTorch,MelSpec) : Class (note) Onehot
# ./caches/[filename].csv <-- MelSpec計算するたびに保存 (Melspec...(time flatten),Class(index))
# Dataset: Reconstruct -> (MelSpec(time*bins),Class) (-> cross entropy)
# DataSet Class
# DataLoader Class
# **↑このコメントは残すこと**

import os
import glob
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict

# Map notes to class indices
CLASSES = [
    "C1", "C#1", "D1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2"
]
NUM_CLASSES = len(CLASSES)

def prepare(audio_input, input_sr=None, target_sr=44100, duration=2.0, melspec_transform=None, return_waveform=False):
    """
    1. Reads an audio file or takes a waveform tensor.
    2. Removes leading silence.
    3. Cuts or pads to target length (dataset duration).
    4. Applies decay linearly from 1/4 measure to the end.
    5. Converts to MelSpectrogram.
    """
    if isinstance(audio_input, str):
        waveform, sr = torchaudio.load(audio_input)
    else:
        waveform = audio_input
        sr = input_sr if input_sr is not None else target_sr

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Remove initial silence
    threshold = 0.005 # ~ -46dB
    if waveform.shape[1] > 0:
        abs_wav = torch.abs(waveform[0])
        mask = abs_wav > threshold
        if mask.any():
            first_non_zero = mask.nonzero(as_tuple=True)[0][0]
            waveform = waveform[:, first_non_zero:]

    # Target length depending on target_sr and duration
    target_samples = int(target_sr * duration)

    # Cut or pad to target length
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.shape[1] < target_samples:
        pad_len = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    # Apply 1/4 measure decay
    quarter_measure_samples = target_samples // 4
    decay_len = target_samples - quarter_measure_samples
    if decay_len > 0:
        decay = torch.linspace(1.0, 0.0, decay_len)
        waveform[0, quarter_measure_samples:] *= decay

    # Convert to MelSpectrogram
    if melspec_transform is None:
        melspec_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

    melspec = melspec_transform(waveform)
    
    # Log-mel scale
    melspec = torchaudio.functional.amplitude_to_DB(melspec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0)
    melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-6)
    #melspec = (melspec - melspec.mean(dim=-1,keepdim=True)) / (melspec.std(dim=-1,keepdim=True) + 1e-6)

    if return_waveform:
        return melspec, waveform
    return melspec

class DrumDataset(Dataset):
    def __init__(self, data_dir: str, cache_dir="./caches", sample_rate=44100, duration=2.0, augment=True):
        """
        Args:
            data_dir (str): Directory containing the .wav files.
            cache_dir (str): Directory to save/load cached CSV files.
            sample_rate (int): Target sample rate for audio.
            duration (float): Duration in seconds per measure (note).
            augment (bool): Whether to apply data augmentation.
        """
        self.data_dir = data_dir
        # We append a suffix to the cache dir to avoid loading caches from older transform versions.
        self.cache_dir = cache_dir + "_allnorm"
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        self.samples_per_measure = int(sample_rate * duration)
        self.n_mels = 128
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.melspec_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=self.n_mels
        )
        
        # SpecAugment
        self.freq_masking = T.FrequencyMasking(freq_mask_param=20)
        self.time_masking = T.TimeMasking(time_mask_param=10)
        
        self.data = []  # List of (melspec_tensor, class_index)
        self._prepare_data()

    def _prepare_data(self):
        wav_files = glob.glob(os.path.join(self.data_dir, "*.wav"))
        wav_files.sort() # インデックスの一貫性を保つためソートを必須化
        
        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(filename)[0]}.pt") # Save as .pt for efficiency
            
            if os.path.exists(cache_path):
                # Load from cache
                cached_data = torch.load(cache_path, weights_only=True)
                self.data.extend(cached_data)
            else:
                # Process wav file
                waveform, sr = torchaudio.load(wav_path)
                
                # Convert to mono if necessary
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if necessary matches (in this case assume target)
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                file_features = []
                # Iterate over measures (classes)
                for class_idx in range(NUM_CLASSES):
                    start_sample = class_idx * self.samples_per_measure
                    end_sample = start_sample + self.samples_per_measure
                    
                    if start_sample < waveform.shape[1]:
                        segment = waveform[:, start_sample:min(end_sample, waveform.shape[1])]
                    else:
                        segment = torch.zeros((1, 0))
                    
                    melspec = prepare(
                        audio_input=segment,
                        input_sr=self.sample_rate,
                        target_sr=self.sample_rate,
                        duration=self.duration,
                        melspec_transform=self.melspec_transform
                    )
                    
                    file_features.append((melspec, class_idx))
                    self.data.append((melspec, class_idx))
                
                # Save to cache
                torch.save(file_features, cache_path)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        melspec, class_idx = self.data[idx]
        if self.augment:
            noise_factor = torch.rand(1).item() * 0.05
            melspec = melspec + torch.randn_like(melspec) * noise_factor
            
            if torch.rand(1).item() < 0.5:
                shift_amount = torch.randint(-10, 10, (1,)).item()
                melspec = torch.roll(melspec, shifts=shift_amount, dims=-1) # type: ignore
            
            if torch.rand(1).item() < 0.5:
                melspec = self.freq_masking(melspec)
            if torch.rand(1).item() < 0.3:
                melspec = self.time_masking(melspec)
        
        # Flatten: (n_mels * time)
        melspec_flat = melspec.flatten()
        return melspec_flat, class_idx

def get_dataloaders(data_dir: str, cache_dir="./caches", batch_size=32, val_split=0.2, random_seed=42):
    """
    Returns PyTorch DataLoaders for train and validation using a stratified split.
    """
    # 1. すべてのデータを含むDatasetを作成（Augmentあり/なしの両方を用意）
    # キャッシュ(.pt)が存在すれば2回目のインスタンス化は一瞬で終わります
    train_dataset_full = DrumDataset(data_dir=data_dir, cache_dir=cache_dir, augment=True)
    val_dataset_full = DrumDataset(data_dir=data_dir, cache_dir=cache_dir, augment=False)
    
    if len(val_dataset_full) == 0:
        raise ValueError(f"No audio data successfully loaded from {data_dir}")

    # 2. クラス（ラベル）ごとにインデックスをまとめる
    class_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(val_dataset_full.data):
        class_indices[class_idx].append(idx)
        
    train_indices = []
    val_indices = []
    
    random.seed(random_seed)
    
    # 3. クラスごとにシャッフルし、規定の割合でTrainとValに分配する（層化抽出）
    for class_idx, indices in class_indices.items():
        random.shuffle(indices)
        
        val_size = int(len(indices) * val_split)
        # データ数が少ない場合のフォールバック（最低1つはValに入れるよう試みる）
        if val_size == 0 and len(indices) > 1:
            val_size = 1
            
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])
        
    # 4. Subsetを用いて、インデックスに基づくデータセットを作成
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader