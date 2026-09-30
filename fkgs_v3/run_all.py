"""
run_all.py — Chạy pipeline FKGS v3 cho MỘT bộ dữ liệu.

  python run_all.py --dataset diabetes            # chạy đầy đủ (tạo run mới)
  python run_all.py --dataset heart --quick       # quy mô nhỏ để kiểm tra
  python run_all.py --dataset diabetes --only E3  # chạy lại một bước TRONG run hiện có
  python run_all.py --dataset diabetes --allow-synthetic   # chỉ để thử pipeline

Truy vết (QA-03): mỗi lần chạy đầy đủ tạo run_id mới, XOÁ kết quả cũ của bộ dữ
liệu đó và ghi manifest.json (mã băm mã nguồn, dữ liệu, cấu hình hiệu lực, môi
trường, thời điểm). --only chỉ được dùng khi manifest hiện có cùng mã băm mã
nguồn và cấu hình; bước REPORT từ chối gộp các tệp khác run_id (QA Stage 3).
"""
import argparse, time, json, os, sys, glob, datetime, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

ORDER = ["PREPROCESS", "E1", "E2", "E3", "E4", "REPORT"]


def apply_quick():
    C.SCALE.N_EXP = 300
    C.KFOLD.K = 3
    C.E1.N_VALUES = [10, 12]
    C.E1.N_REPEATS = 3
    C.E2.THETA_VALUES = [0.1, 0.3]
    C.E2.N_SEEDS_RANDOM = 3
    C.E3.THETA_GRID = [0.1, 0.3, 0.5, 1.0]
    C.E4.N_BLOCKS = 2
    C.E4.BLOCK_SIZE = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--pair-scope", default=None, choices=["global", "adjacent", "upper"])
    ap.add_argument("--results-suffix", default="")
    a = ap.parse_args()

    C.set_dataset(a.dataset)
    if a.results_suffix:
        C.PATHS.RESULTS_DIR += "_" + a.results_suffix
        C.PATHS.DATA_CLEAN = os.path.join(C.PATHS.RESULTS_DIR, "Data_clean.csv")
    if a.quick:
        apply_quick()
    if a.pair_scope:
        C.FISA.PAIR_SCOPE = a.pair_scope
    C.RUN.ALLOW_SYNTHETIC = a.allow_synthetic
    from experiments.common import code_fingerprint, environment, save_json, load_json

    code_hash, per_file = code_fingerprint()
    snap = C.snapshot()
    snap["quick"] = a.quick
    only = [s.strip().upper() for s in a.only.split(",") if s.strip()]
    for s in only:
        if s not in ORDER:
            raise SystemExit(f"Bước không hợp lệ: {s}")
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    man_path = os.path.join(C.PATHS.RESULTS_DIR, "manifest.json")

    if only:
        if not os.path.exists(man_path):
            raise SystemExit("Chưa có run nào cho bộ dữ liệu này -- chạy đầy đủ trước khi dùng --only")
        man = json.load(open(man_path, encoding="utf-8"))
        if man["code_sha256"] != code_hash or man["config"] != json.loads(json.dumps(snap)):
            raise SystemExit("Mã nguồn hoặc cấu hình đã khác run hiện có -- phải chạy lại đầy đủ "
                             "(không trộn kết quả của hai phiên bản)")
        C.RUN.RUN_ID = man["run_id"]
        steps = [s for s in ORDER if s in only]
        if any(s in ("E1", "E2", "E3", "E4") for s in steps) and not os.path.exists(C.PATHS.DATA_CLEAN):
            steps = ["PREPROCESS"] + steps
    else:
        for f in glob.glob(os.path.join(C.PATHS.RESULTS_DIR, "*")):
            if os.path.isfile(f):
                os.remove(f)
        C.RUN.RUN_ID = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        steps = ORDER
        man = dict(run_id=C.RUN.RUN_ID, started=datetime.datetime.now().isoformat(timespec="seconds"),
                   dataset=a.dataset, code_sha256=code_hash, code_files=per_file,
                   config=json.loads(json.dumps(snap)), environment=environment(), steps={})

    print(f"RUN {C.RUN.RUN_ID} | dataset={a.dataset} | quick={a.quick} | steps={steps}")
    t_all = time.time()
    for step in steps:
        print(f"\n{'#' * 60}\n# {step}\n{'#' * 60}", flush=True)
        t0 = time.time()
        if step == "PREPROCESS":
            from experiments import run_00_preprocess
            prov, info = run_00_preprocess.main()
            man["data"] = prov
            man["preprocess"] = info
        elif step == "E1":
            from experiments.run_01_E1_theory_validation import run_e1
            run_e1()
        elif step == "E2":
            from experiments.run_02_E2_greedy_vs_random import run_e2
            run_e2()
        elif step == "E3":
            from experiments.run_03_E3_theta_curve import run_e3
            run_e3()
        elif step == "E4":
            from experiments.run_04_E4_four_strategies import run_e4
            run_e4()
        elif step == "REPORT":
            with open(man_path, "w", encoding="utf-8") as fh:
                json.dump(man, fh, ensure_ascii=False, indent=1)
            from experiments import run_05_generate_report
            run_05_generate_report.main()
        man["steps"][step] = dict(finished=datetime.datetime.now().isoformat(timespec="seconds"),
                                  seconds=round(time.time() - t0, 2))
        with open(man_path, "w", encoding="utf-8") as fh:
            json.dump(man, fh, ensure_ascii=False, indent=1)
    print(f"\nHOÀN TẤT {C.RUN.RUN_ID} trong {(time.time() - t_all) / 60:.1f} phút")


if __name__ == "__main__":
    main()
