def to_float(val):
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def store_simple(results, timings, name, out, elapsed):
    results[name] = {
        "accuracy_uncompressed":   to_float(out["accuracy"]["uncompressed"]),
        "accuracy_compressed":     to_float(out["accuracy"]["compressed"]),
        "accuracy_mlp_baseline":   to_float(out["accuracy"].get("mlp_baseline",   float("nan"))),
        "accuracy_mlp_compressed": to_float(out["accuracy"].get("mlp_compressed", float("nan"))),
        "std_uncompressed":    to_float(out.get("accuracy_std", {}).get("uncompressed",    0.0)),
        "std_compressed":      to_float(out.get("accuracy_std", {}).get("compressed",      0.0)),
        "std_mlp_baseline":    to_float(out.get("accuracy_std", {}).get("mlp_baseline",    0.0)),
        "std_mlp_compressed":  to_float(out.get("accuracy_std", {}).get("mlp_compressed",  0.0)),
        "accuracy_compressed_global":  to_float(out["accuracy"].get("compressed_global",  float("nan"))),
        "accuracy_compressed_dynamic": to_float(out["accuracy"].get("compressed_dynamic", float("nan"))),
        "accuracy_compressed_int4":    to_float(out["accuracy"].get("compressed_int4",    float("nan"))),
        "std_compressed_global":  to_float(out.get("accuracy_std", {}).get("compressed_global",  0.0)),
        "std_compressed_dynamic": to_float(out.get("accuracy_std", {}).get("compressed_dynamic", 0.0)),
        "std_compressed_int4":    to_float(out.get("accuracy_std", {}).get("compressed_int4",    0.0)),
        "mse_uncompressed":    to_float(out.get("mse", {}).get("uncompressed",   float("nan"))),
        "mse_compressed":      to_float(out.get("mse", {}).get("compressed",     float("nan"))),
        "mse_mlp_baseline":    to_float(out.get("mse", {}).get("mlp_baseline",   float("nan"))),
        "mse_mlp_compressed":  to_float(out.get("mse", {}).get("mlp_compressed", float("nan"))),
        "mse_std_uncompressed":   to_float(out.get("mse_std", {}).get("uncompressed",   0.0)),
        "mse_std_compressed":     to_float(out.get("mse_std", {}).get("compressed",     0.0)),
        "mse_std_mlp_baseline":   to_float(out.get("mse_std", {}).get("mlp_baseline",   0.0)),
        "mse_std_mlp_compressed": to_float(out.get("mse_std", {}).get("mlp_compressed", 0.0)),
        "size_uncompressed":     out["sizes"]["uncompressed"],
        "size_compressed":       out["sizes"]["compressed"],
        "size_mlp_uncompressed":   out["sizes"].get("mlp_uncompressed"),
        "size_mlp_compressed":     out["sizes"].get("mlp_compressed"),
        "size_compressed_global":  out["sizes"].get("compressed_global"),
        "size_compressed_dynamic": out["sizes"].get("compressed_dynamic"),
        "size_compressed_int4":    out["sizes"].get("compressed_int4"),
        "time_seconds": elapsed,
        "num_seeds":    out.get("num_seeds", 1),
        "curve_data":   out.get("curve_data"),
        "loss_history": out.get("loss_history"),
    }
    timings[name] = elapsed
