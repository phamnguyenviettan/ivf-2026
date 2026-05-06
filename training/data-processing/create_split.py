import os
import random
import json
import argparse
from pathlib import Path
import csv

STAGES = ['t2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+']

# Danh sách bệnh nhân bị loại bỏ (Blacklist) do tế bào chết hoặc không phát triển
BLACKLIST = ['AAL839-6', 'AL884-2', 'ALR493-10', 'ALR493-6', 'AMT360-1-9', 'AS1015-2', 'AS662-2', 'BA782-2', 'BC277-10', 'BC396-1', 'BE645-3', 'BJ3371-9', 'BJ492-11', 'BJ492-8', 'BL285-1-3', 'BM016-2', 'BM256-1', 'BM256-4', 'BM655-10', 'BM968-3', 'BN1010-5', 'BN356-3', 'BS1086-1', 'BS648-2-4', 'BS648-7', 'BS836-11', 'BV646-6', 'CA063-10', 'CA063-6', 'CA364-7', 'CA390-2', 'CA390-6', 'CA658-12', 'CA658-6', 'CA704-2', 'CAV074-1', 'CAV074-3', 'CC938-4', 'CJ261-10', 'CK601-2', 'CM627-8', 'CM892-5', 'CS552-2', 'CS552-4', 'CZ594-1', 'CZ594-5', 'DC307-1', 'DC307-2', 'DH1012-1', 'DHDPI042-3', 'DHDPI042-7', 'DHDPI042-8', 'DJC641-4', 'DL020-3', 'DM1046-12', 'DRL1048-1', 'DS17-2', 'DS61-1', 'DS666-9', 'DS947-2', 'DSE41-2', 'DV210-4', 'DV210-8', 'DV305-3', 'EH315-3', 'EH315-8', 'EJ393-3', 'FA344-5', 'FA662-6', 'FC1164-11', 'FE14-020', 'FH658-4', 'FM1017-5', 'FM162-6', 'FN852-1', 'GA1087-6', 'GA122-8', 'GA664-3', 'GA664-8', 'GC340-10', 'GC658-3', 'GC836-4', 'GC851-5', 'GE1055-6', 'GE218-3', 'GE294-4', 'GE663-5', 'GF083-5', 'GF1042-1-3', 'GJ165-5', 'GJ316-1', 'GM293-2', 'GM456-3', 'GS334-6', 'GS400-7', 'GS415-5', 'GS430-2', 'GS490-2', 'GS490-7', 'GS490-_6', 'GS811-3', 'GS826-2', 'GS980-2', 'HE444-3', 'HE444-4', 'HH569-2', 'HH569-4', 'HM69-4', 'HS15-11', 'JE021-4', 'JV227-2', 'JV227-5', 'KF460-11', 'KF460-7', 'KJ1077-3', 'LA367-4', 'LBE649-3', 'LBM519-1', 'LBM659-6', 'LD400-1', 'LD400-6', 'LEG557-3', 'LM985-4', 'LNA592-8', 'LP284-3', 'LS058-7', 'LS058-8', 'LS123-3', 'LS366-1', 'LTA908-2', 'LV488-7', 'LV683-2-3', 'LV683-2-8', 'LV723-9', 'LZ865-2', 'MA1007-3', 'MA1059-3', 'MA505-2', 'MAS094-5', 'MAS203-4', 'MAS203-6', 'MC427-1', 'MC833-6', 'MC933-2', 'ME799-5', 'MM445-2-2', 'MM445-2-9', 'MM834-5', 'MM84-8', 'MRA165-6', 'MRA165-7T', 'MV750-5', 'NC636-4', 'OA333-6', 'OC110-5', 'OJ319-5', 'OJ319-7', 'PA214-5', 'PA276-3', 'PA916-1-10', 'PC55-2', 'PC758-2', 'PC809-7', 'PE863-4', 'PG209-3', 'PH394-2', 'PI1004-3', 'PMDPI029-1-1', 'PMDPI029-1-10', 'PMDPI029-1-11', 'PMDPI029-1-2', 'PMDPI029-1-3', 'PMDPI029-1-6', 'PN636-1-6', 'PO13-3', 'PV361-2', 'RA803-4', 'RC1103-1', 'RC545-2-8', 'RC545-2-9', 'RC755-4', 'RD167-7', 'RL461-4', 'RL948-2', 'RLFS800-2', 'RM126-1', 'RM126-10', 'RM126-4', 'RM126-6', 'RM126-9', 'RM29-5', 'RMN410-3', 'RV454-6', 'SC385-11', 'SK308-10', 'SK308-7', 'SK902-1-8', 'SLM044-1-1', 'SM307-1-9', 'SN586-8', 'ST586-7', 'TA12-1', 'TA757-9', 'TC1047-2', 'TD958-2-1', 'TJ899-2', 'TL179-5', 'TM312-6', 'TM428-3', 'TN359-10', 'TN359-9', 'TN807-3', 'TV654-4', 'UL050-_10', 'UL050-_9', 'VA225-6', 'VF269-7', 'VN484-1', 'VS321-7', 'VS510-2', 'WA1014-3', 'WA402-7', 'WS531-4', 'ZL1077-1', 'ZS435-6']

def load_stage_counts(csv_path):
    """Dem so luong frames cho cac giai doan (t2-t9+) trong mot phoi."""
    counts = {s: 0 for s in STAGES}
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    stage = row[0].strip().lower()
                    if stage in counts:
                        start = int(row[1])
                        end = int(row[2])
                        counts[stage] += (end - start + 1)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return counts

def create_splits(data_root, annotations_root, output_file):
    """
    Quet du lieu va su dung thuat toan Greedy Multi-Way Number Partitioning 
    de chia phoi vao Train/Val/Test dam bao so luong anh cua tung class 
    bam sat ti le 70/15/15 nhat co the.
    """
    data_path = Path(data_root)
    ann_path = Path(annotations_root)
    output_path = Path(output_file)

    if not data_path.exists() or not ann_path.exists():
        print(f"[ERROR] Khong tim thay thu muc {data_path} hoac {ann_path}.")
        return

    # 1. Quét thông tin phôi
    print(f"[INFO] Dang phan tich phoi tai {data_path}...")
    embryo_info = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            # Kiem tra blacklist
            if item.name in BLACKLIST:
                continue
                
            csv_path = ann_path / f"{item.name}_phases.csv"
            if csv_path.exists():
                counts = load_stage_counts(csv_path)
                total_frames = sum(counts.values())
                if total_frames > 0:
                    embryo_info.append({
                        "id": item.name,
                        "counts": counts,
                        "total_frames": total_frames
                    })

    total_embryos = len(embryo_info)
    print(f"[SUCCESS] Tim thay {total_embryos} phoi hop le.")
    if total_embryos == 0: return

    # 2. Xao tron ngau nhien truoc khi tinh toan de da dang hoa
    random.seed(42)
    random.shuffle(embryo_info)
    
    # Uu tien xep cac phoi co nhieu anh nhat vao truoc de toi uu hoa moc dung sai (Greedy Scheduling)
    embryo_info.sort(key=lambda x: x["total_frames"], reverse=True)

    # 3. Tinh tong so frame cua tung class tren toan cuc
    class_totals = {c: sum(e["counts"][c] for e in embryo_info) for c in STAGES}

    targets = {"train": 0.70, "val": 0.15, "test": 0.15}
    splits = {"train": [], "val": [], "test": []}
    
    current_counts = {
        "train": {c: 0 for c in STAGES},
        "val": {c: 0 for c in STAGES},
        "test": {c: 0 for c in STAGES}
    }

    # 4. Thuat toan Greedy Scheduling - Phan bo can bang theo tai trong cua Class
    for e in embryo_info:
        best_split = None
        best_score = float('inf')

        for split, weight in targets.items():
            max_rel_load = 0
            for c in STAGES:
                if e["counts"][c] > 0 and class_totals[c] > 0:
                    future_val = current_counts[split][c] + e["counts"][c]
                    target_capacity = class_totals[c] * weight
                    rel_load = future_val / target_capacity
                    max_rel_load = max(max_rel_load, rel_load)
            
            if max_rel_load == 0:
                max_rel_load = len(splits[split]) / max(1, (total_embryos * weight))

            if max_rel_load < best_score:
                best_score = max_rel_load
                best_split = split
        
        splits[best_split].append(e["id"])
        for c in STAGES:
            current_counts[best_split][c] += e["counts"][c]

    # Kiem tra lai so phoi trong tung tap
    print("\n" + "-" * 75)
    print(f"{'Class (Stage)':<15} | {'Train (70%)':>13} | {'Val (15%)':>13} | {'Test (15%)':>13} | {'Total':>9}")
    print("-" * 75)
    
    for c in STAGES:
        t_c = current_counts["train"][c]
        v_c = current_counts["val"][c]
        te_c = current_counts["test"][c]
        tot = t_c + v_c + te_c
        
        # Tinh phan tram thuc te de hien thi
        t_pct = (t_c / tot * 100) if tot > 0 else 0
        v_pct = (v_c / tot * 100) if tot > 0 else 0
        te_pct = (te_c / tot * 100) if tot > 0 else 0
        
        print(f"{c.upper():<15} | {t_c:6,d} ({t_pct:4.1f}%) | {v_c:6,d} ({v_pct:4.1f}%) | {te_c:6,d} ({te_pct:4.1f}%) | {tot:9,d}")
    
    print("-" * 75)
    t_id = len(splits['train'])
    v_id = len(splits['val'])
    te_id = len(splits['test'])
    tot_id = t_id + v_id + te_id
    
    t_id_pct = (t_id / tot_id * 100)
    v_id_pct = (v_id / tot_id * 100)
    te_id_pct = (te_id / tot_id * 100)
    
    print(f"{'Total Embryos':<15} | {t_id:6,d} ({t_id_pct:4.1f}%) | {v_id:6,d} ({v_id_pct:4.1f}%) | {te_id:6,d} ({te_id_pct:4.1f}%) | {tot_id:9,d}")
    print("-" * 75)

    # Ghi de file splits.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"\n[DONE] File phan chia chuan muc da duoc luu tai: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stratified Train/Val/Test splits (70/15/15)")
    parser.add_argument("--data_root", type=str, default=r"d:\SDH\Paper\ivf\embryo_dataset", help="Path to embryo image folders")
    parser.add_argument("--annotations_root", type=str, default=r"d:\SDH\Paper\ivf\embryo_dataset_annotations", help="Path to annotation CSVs")
    parser.add_argument("--output_file", type=str, default=r"d:\SDH\Paper\ivf\splits.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    create_splits(args.data_root, args.annotations_root, args.output_file)
