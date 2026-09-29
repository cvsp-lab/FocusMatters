import os
import json
import csv
import argparse


def load_chair_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    om = d.get("overall_metrics", d)
    out = {
        "CHAIRs":    om.get("CHAIRs",    om.get("chair_s")),
        "CHAIRi":    om.get("CHAIRi",    om.get("chair_i")),
        "Recall":    om.get("Recall",    om.get("recall")),
        "Precision": om.get("Precision", om.get("precision")),
        "F1":        om.get("F1",        om.get("f1")),
    }
    if out["F1"] is None and out["Precision"] is not None and out["Recall"] is not None:
        p, r = out["Precision"], out["Recall"]
        if p + r > 0:
            out["F1"] = 2 * p * r / (p + r)
    return out


def f(v, fmt="{:.3f}", default="-"):
    return fmt.format(v) if v is not None else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True,
        help="root folder, e.g. results/value_sweep")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--interventions", nargs="+", default=["low_mean"],
        choices=["low_mean"])
    p.add_argument("--split_ratios", nargs="+", type=float, default=[0.25])
    p.add_argument("--baseline_files", nargs="*", default=[],
        help="Format: <model>=<path/to/chair_eval_full.json>")
    p.add_argument("--out_md",  default=None)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    if args.out_md  is None: args.out_md  = os.path.join(args.base_dir, "value_sweep.md")
    if args.out_csv is None: args.out_csv = os.path.join(args.base_dir, "value_sweep.csv")

    baseline_map = {}
    for s in args.baseline_files:
        if "=" in s:
            k, v = s.split("=", 1)
            baseline_map[k] = v

    rows = []
    for model in args.models:
        if model in baseline_map:
            cm = load_chair_metrics(baseline_map[model])
            if cm is not None:
                rows.append({
                    "Model": model, "Intervention": "baseline",
                    "SplitRatio": 0.0,
                    "CHAIRs": cm["CHAIRs"], "CHAIRi": cm["CHAIRi"], "F1": cm["F1"],
                })

        for intv in args.interventions:
            for sr in args.split_ratios:
                tag = f"{intv}_sr{sr}"
                path = os.path.join(args.base_dir, model, tag, "chair_eval_full.json")
                cm = load_chair_metrics(path)
                if cm is None:
                    rows.append({
                        "Model": model, "Intervention": intv,
                        "SplitRatio": sr,
                        "CHAIRs": None, "CHAIRi": None, "F1": None,
                    })
                else:
                    rows.append({
                        "Model": model, "Intervention": intv,
                        "SplitRatio": sr,
                        "CHAIRs": cm["CHAIRs"], "CHAIRi": cm["CHAIRi"], "F1": cm["F1"],
                    })

    lines = []
    for model in args.models:
        lines.append(f"### {model}\n")
        lines.append("| Intervention | split_ratio | CHAIR_S↓ | CHAIR_I↓ | F1↑ |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in rows:
            if r["Model"] != model: continue
            lines.append("| {} | {} | {} | {} | {} |".format(
                r["Intervention"],
                f"{r['SplitRatio']:.2f}" if r["SplitRatio"] > 0 else "-",
                f(r["CHAIRs"]), f(r["CHAIRi"]), f(r["F1"]),
            ))
        lines.append("")

    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as fp:
        fp.write(md)
    print(md)
    print(f"[Saved] {args.out_md}")

    with open(args.out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["Model", "Intervention", "SplitRatio", "CHAIRs", "CHAIRi", "F1"])
        for r in rows:
            w.writerow([r["Model"], r["Intervention"], r["SplitRatio"],
                        r["CHAIRs"], r["CHAIRi"], r["F1"]])
    print(f"[Saved] {args.out_csv}")


if __name__ == "__main__":
    main()
