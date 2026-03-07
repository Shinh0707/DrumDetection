import os
from pathlib import Path
import torch
import torchaudio
from torch import nn
from dataset import prepare, CLASSES
from model import DrumResNet

class Evaluater:
    def __init__(self, model, device: str | torch.device="cpu"):
        self.model = model
        self.device = device
        
    def load_model(self, model_path):
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}...")
            # Check for weights_only support (Torch 2.4+)
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True)['model_state_dict'])
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        else:
            print(f"Warning: Model file {model_path} not found.")

    def predict_file(self, audio_path, return_waveform=False):
        self.model.eval()
        
        try:
            if return_waveform:
                melspec, waveform = prepare(audio_path, return_waveform=True)
            else:
                melspec: torch.Tensor = prepare(audio_path) # type: ignore
            
            melspec_flat = melspec.flatten().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(melspec_flat)
                _, predicted = outputs.max(1)
                predicted_idx = predicted.item()
                predicted_note = CLASSES[predicted_idx]
                
                if return_waveform:
                    return predicted_note, waveform # type: ignore
                return predicted_note
        except Exception as e:
            print(f"Error predicting {audio_path}: {e}")
            if return_waveform:
                return None, None
            return None

    def predict_dir(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        print("-" * 80)
        print(f"Detecting and saving sorted files to: {output_dir}")
        print("-" * 80)
        print(f"{'Filename':<40} | {'Prediction':<10} | Saved Path")
        print("-" * 80)

        input_path = Path(input_dir)
        
        # サブディレクトリを含めて .wav と .mp3 を再帰的に取得
        audio_files = list(input_path.rglob("*.wav")) + list(input_path.rglob("*.mp3"))
        
        # 出力先の predicts フォルダ自体を読み込んで無限ループになるのを防ぐ
        audio_files = [f for f in audio_files if "predicts" not in f.parts]
        
        for file_path in audio_files:
            file_path_str = str(file_path)
            filename = file_path.name
            base_filename = file_path.stem
            
            try:
                note, waveform = self.predict_file(file_path_str, return_waveform=True) # type: ignore
                
                if note and waveform is not None:
                    # 予測されたノート（クラス）のフォルダを作成
                    note_dir = os.path.join(output_dir, note)
                    os.makedirs(note_dir, exist_ok=True)
                    
                    save_path = os.path.join(note_dir, f"{base_filename}.wav")
                    # PyTorchのテンソルをWAVファイルとして書き出し
                    torchaudio.save(save_path, waveform, sample_rate=44100)
                    
                    # ターミナル出力のレイアウト崩れを防ぐためファイル名を切り詰め
                    display_name = filename if len(filename) <= 40 else filename[:37] + "..."
                    print(f"{display_name:<40} | {note:<10} | {save_path}")
                else:
                    print(f"{filename:<40} | {'FAILED':<10} | -")
            except Exception as e:
                print(f"{filename:<40} | {'ERROR':<10} | {e}")
                
        print("-" * 80)

if __name__ == "__main__":
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Initialize model
    model = DrumResNet(num_classes=len(CLASSES)).to(device)
    
    evaluater = Evaluater(model, device=device)

    model_ver = "dualnorm_allnorm_v14" # 該当するモデルのバージョン名
    best_model_path = os.path.join("models", model_ver, "best.pth")
    evaluater.load_model(best_model_path)
    
    # ディレクトリの設定
    input_dir = ""
    output_dir = ""
    
    evaluater.predict_dir(input_dir, output_dir)