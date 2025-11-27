import argparse
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_file(file_path: Path, delete_original: bool):
    mp3_path = file_path.with_suffix(".mp3")
    if mp3_path.exists():
        return  # Skip if output already exists

    try:
        audio = AudioSegment.from_file(file_path, format="mp4")
        audio = audio.set_frame_rate(16000).set_channels(1)
        tmp_path = mp3_path.with_suffix(".mp3.tmp")
        audio.export(tmp_path, format="mp3")
        tmp_path.rename(mp3_path)
        if delete_original:
            file_path.unlink()
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def convert_mp4_to_mp3(input_dir, workers=1, delete_original=False):
    input_path = Path(input_dir)
    mp4_files = list(input_path.rglob("*.mp4"))

    if workers == 1:
        # Sequential processing
        for file_path in tqdm(mp4_files, desc="Converting MP4 to MP3"):
            convert_file(file_path, delete_original)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(convert_file, f, delete_original): f for f in mp4_files}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting MP4 to MP3"):
                pass  # tqdm just tracks progress

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 files to 16kHz mono MP3s")
    parser.add_argument("input_dir", help="Input directory containing MP4 files")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--delete", action="store_true", help="Delete original MP4 after conversion")
    args = parser.parse_args()

    convert_mp4_to_mp3(args.input_dir, workers=args.workers, delete_original=args.delete)
