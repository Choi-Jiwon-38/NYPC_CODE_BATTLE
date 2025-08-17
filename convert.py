import json
import pickle
import gzip # gzip 라이브러리 추가

json_filename = 'ev_table_AB.json'
# 저장될 파일 이름에 .gz 확장자를 붙여주는 것이 일반적입니다.
binary_filename = 'data.bin'

print(f"Loading {json_filename}...")
with open(json_filename, 'r') as f:
    ev_table = json.load(f)

print(f"Compressing and saving to {binary_filename}...")
# --- [수정] open 대신 gzip.open 사용 ---
with gzip.open(binary_filename, 'wb') as f:
    # pickle.dump는 그대로 사용합니다. gzip이 알아서 압축해줍니다.
    pickle.dump(ev_table, f)

print("Compression complete!")