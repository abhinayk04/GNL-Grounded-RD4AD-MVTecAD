print("=" * 70)
print("BATCH 11Y-A — RECOVER ORIGINAL EFDM-GNL EVALUATION")
print("=" * 70)

# Find the previous full evaluation cell in notebook history
matches = []

for i, source in enumerate(In):
    if (
        "efdm_gnl_all15" in source
        and "calculate_anomaly_map" in source
        and ("EFDM" in source or "extract_efdm_features" in source)
    ):
        matches.append((i, source))

if not matches:
    print("Previous evaluation cell not found in notebook history.")
    print("Available matching cells:")

    for i, source in enumerate(In):
        if "efdm_gnl" in source.lower():
            print(f"Cell In[{i}]")

else:
    index, source = matches[-1]

    print(f"Recovered evaluation cell: In[{index}]")
    print(f"Source length: {len(source)} characters")
    print("\n" + "=" * 70)
    print("RECOVERED SOURCE")
    print("=" * 70)

    print(source)

    # Preserve recovered source for the next consolidated execution batch
    from pathlib import Path

    output = Path(
        "/kaggle/working/project_clean/docs/"
        "member3_batch11y_recovered_evaluation.py"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")

    print("\nSaved:", output)