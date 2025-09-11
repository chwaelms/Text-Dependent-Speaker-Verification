import os
import pandas as pd
import argparse

def merge_csv_files(original_csv_path, new_csv_path, output_path):
    # ===== 기존 CSV 파일 불러오기 =====
    if os.path.exists(original_csv_path):
        df_original = pd.read_csv(original_csv_path)
        df_original = df_original.drop('Unnamed: 0', axis=1, errors='ignore')
    else:
        raise FileNotFoundError(f"Original CSV file not found: {original_csv_path}")

    # ===== 새로 병합할 CSV 파일 불러오기 ======
    if os.path.exists(new_csv_path):
        df_new = pd.read_csv(new_csv_path)
        df_new = df_new.drop('Unnamed: 0', axis=1, errors='ignore')
    else:
        raise FileNotFoundError(f"New CSV file not found: {new_csv_path}")

    # ===== NaN이 있는 행 제거 =====
    df_original.dropna(inplace=True)
    df_new.dropna(inplace=True)

    # ===== 기존 데이터의 가장 마지막 화자 ID 값 가져오기 =====
    last_label = df_original['utt_spk_int_labels'].max() if not df_original.empty else -1

    # ===== 새로운 CSV의 화자 ID 값을 기존 값 다음부터 이어지도록 조정 =====
    df_new['utt_spk_int_labels'] += last_label + 1

    # ===== 두 데이터프레임 병합 =====
    df_combined = pd.concat([df_original, df_new], ignore_index=True)

    print("Combined DataFrame:")
    print(df_combined.head())
    print(df_combined.tail())
    print(f"Total rows in combined DataFrame: {len(df_combined)}")

    # ===== 병합된 CSV 파일 저장 =====
    try:
        df_combined.to_csv(output_path, index=True, index_label='')
        print(f"Updated CSV saved at {output_path}")
    except OSError as err:
        print(f"Error while saving {output_path}: {err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CSV files.")
    parser.add_argument('--original_csv', type=str, required=True, help="Path to the original CSV file.")
    parser.add_argument('--new_csv', type=str, required=True, help="Path to the new CSV file to be merged.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the merged CSV file.")

    args = parser.parse_args()

    merge_csv_files(
        original_csv_path=args.original_csv,
        new_csv_path=args.new_csv,
        output_path=args.output_path
    )