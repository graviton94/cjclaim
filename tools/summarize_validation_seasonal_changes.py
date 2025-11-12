import pandas as pd


def main():
    path = "artifacts/validation_seasonal_flag_changes.csv"
    df = pd.read_csv(path)

    total = len(df)
    is_orig_seasonal = df['orig_seasonal'].astype(str) != "[0, 0, 0, 0]"
    is_rerun_seasonal = df['rerun_seasonal'].astype(str) != "[0, 0, 0, 0]"

    s_to_ns = ((is_orig_seasonal) & (~is_rerun_seasonal)).sum()
    ns_to_s = ((~is_orig_seasonal) & (is_rerun_seasonal)).sum()
    unchanged = ((is_orig_seasonal == is_rerun_seasonal).sum())

    print(f"Total rows: {total}")
    print(f"Seasonal -> Non-seasonal: {s_to_ns}")
    print(f"Non-seasonal -> Seasonal: {ns_to_s}")
    print(f"Unchanged (same flag): {unchanged}")

    # Top by absolute AIC delta
    df['aic_delta_num'] = pd.to_numeric(df['aic_delta'], errors='coerce')
    top_aic = df.dropna(subset=['aic_delta_num']).assign(abs_delta=df['aic_delta_num'].abs()).sort_values('abs_delta', ascending=False).head(10)
    if not top_aic.empty:
        print('\nTop 10 by |AIC delta|:')
        for _, r in top_aic.iterrows():
            ntrain = int(r['n_train']) if not pd.isna(r['n_train']) else 'NA'
            print(f"{r['series_id']} | n_train={ntrain} | orig={r['orig_seasonal']} -> rerun={r['rerun_seasonal']} | aic_delta={r['aic_delta']} | plot={r['plot']}")

    # Top by max_abs_arroot
    df['max_abs_arroot_num'] = pd.to_numeric(df['max_abs_arroot'], errors='coerce')
    top_ar = df.dropna(subset=['max_abs_arroot_num']).sort_values('max_abs_arroot_num', ascending=False).head(10)
    if not top_ar.empty:
        print('\nTop 10 by max_abs_arroot:')
        for _, r in top_ar.iterrows():
            ntrain = int(r['n_train']) if not pd.isna(r['n_train']) else 'NA'
            print(f"{r['series_id']} | n_train={ntrain} | max_abs_arroot={r['max_abs_arroot']} | plot={r['plot']}")


if __name__ == '__main__':
    main()
