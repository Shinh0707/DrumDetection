import os
from pathlib import Path
import torch
import torchaudio
from torch import nn
from dataset import prepare, CLASSES
from model import DrumResNet
import logging

class Evaluater:
    def __init__(self, model, device: str | torch.device="cpu", logger: logging.Logger | None = None):
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.logger.info(f"Loading weights from {model_path}...")
            # Check for weights_only support (Torch 2.4+)
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True)['model_state_dict'])
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        else:
            self.logger.warning(f"Warning: Model file {model_path} not found.")

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
            self.logger.error(f"Error predicting {audio_path}: {e}")
            if return_waveform:
                return None, None
            return None

    def predict_dir(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("-" * 80)
        self.logger.info(f"Detecting and saving sorted files to: {output_dir}")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Filename':<40} | {'Prediction':<10} | Saved Path")
        self.logger.info("-" * 80)

        input_path = Path(input_dir)
        
        # サブディレクトリを含めて .wav と .mp3 を再帰的に取得
        audio_files = list(input_path.rglob("*.wav")) + list(input_path.rglob("*.mp3"))
        
        # 出力先のフォルダ自体を読み込んで無限ループになるのを防ぐ
        audio_files = [f for f in audio_files if os.path.dirname(output_dir) not in f.parts]
        
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
                    self.logger.info(f"{display_name:<40} | {note:<10} | {save_path}")
                else:
                    self.logger.warning(f"{filename:<40} | {'FAILED':<10} | -")
            except Exception as e:
                self.logger.error(f"{filename:<40} | {'ERROR':<10} | {e}")
                
        self.logger.info("-" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Drum Detection Model")
    parser.add_argument("--model-ver", type=str, default="v6", help="Model version name (e.g., v6)")
    parser.add_argument("-i", "--input", type=str, default="", help="Input directory containing audio files")
    parser.add_argument("-o", "--output", type=str, default="", help="Output directory to save predictions")
    parser.add_argument("--log", "--log-level", "--loglevel", type=str, default="INFO", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Specify the verbosity of the logs")
    args = parser.parse_args()

    # Logging setup
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

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
    
    evaluater = Evaluater(model, device=device, logger=logger)

    model_ver = args.model_ver
    best_model_path = os.path.join("models", model_ver, "best.pth")
    evaluater.load_model(best_model_path)
    
    # ディレクトリの設定
    input_dir = args.input
    output_dir = args.output
    
    if input_dir and output_dir:
        if (len(input_dir) > 0 and os.path.exists(input_dir)) and len(output_dir) > 0:
            evaluater.predict_dir(input_dir, output_dir)
    else:
        print("Please provide --input and --output arguments.")
        parser.print_help()