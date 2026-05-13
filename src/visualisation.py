import os
import matplotlib.pyplot as plt


def plot_eeg_heo_blinks_with_cues_dual_axis(
    df,
    results_dir,
    start_time=20,
    end_time=40,
    eeg_channel="C3",
    heo_channel="HEO",
):
    os.makedirs(results_dir, exist_ok=True)

    segment = df[
        (df["Time"] >= start_time) &
        (df["Time"] <= end_time)
    ].copy()

    fig, ax_eeg = plt.subplots(figsize=(15, 6))

    eeg_line, = ax_eeg.plot(
        segment["Time"],
        segment[eeg_channel],
        linewidth=1.2,
        label=f"EEG {eeg_channel}",
    )

    ax_eeg.set_xlabel("Time (s)")
    ax_eeg.set_ylabel(f"EEG {eeg_channel} (µV)")
    ax_eeg.grid(True, alpha=0.3)

    ax_heo = ax_eeg.twinx()

    heo_line, = ax_heo.plot(
        segment["Time"],
        segment[heo_channel],
        linewidth=1.0,
        alpha=0.8,
        color='orange',
        label=f"{heo_channel} / eye movement",
    )

    ax_heo.set_ylabel(f"{heo_channel} (µV)")

    blink_rows = segment[segment["Blinks"] == 1]

    blink_scatter = ax_heo.scatter(
        blink_rows["Time"],
        blink_rows[heo_channel],
        marker="x",
        s=80,
        linewidths=2,
        label="Blink",
    )

    cue_handles = []

    for cue, label in [("Left", "Left MI"), ("Right", "Right MI")]:
        cue_segment = segment[segment["Cues"] == cue]

        if not cue_segment.empty:
            patch = ax_eeg.axvspan(
                cue_segment["Time"].min(),
                cue_segment["Time"].max(),
                alpha=0.12,
                label=label,
            )
            cue_handles.append(patch)

    ax_eeg.set_title(
        "Dual-Axis Timeline: EEG, HEO, Blink Markers, and Motor Imagery Cues"
    )

    handles = [eeg_line, heo_line, blink_scatter] + cue_handles
    labels = [h.get_label() for h in handles]

    ax_eeg.legend(handles, labels, loc="upper right")

    output_path = os.path.join(
        results_dir,
        "eeg_heo_blink_cue_timeline_dual_axis.png",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_clean_vs_blink_accuracy(
    clean_accuracy,
    all_accuracy,
    results_dir,
):
    os.makedirs(results_dir, exist_ok=True)

    labels = ["All epochs", "Clean epochs"]
    values = [all_accuracy * 100, clean_accuracy * 100]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values)

    plt.ylabel("Accuracy (%)")
    plt.title("MI Classification: All vs Clean Epochs")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
        )

    output_path = os.path.join(
        results_dir,
        "classification_clean_vs_all.png",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_realtime_predictions(prediction_log, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(14, 5))

    plt.plot(
        prediction_log.index,
        prediction_log["true_label"],
        linestyle="--",
        linewidth=1.5,
        label="True label",
    )

    plt.plot(
        prediction_log.index,
        prediction_log["predicted_label"],
        linewidth=1.2,
        label="Predicted label",
    )

    plt.xlabel("Sliding window index")
    plt.ylabel("Class label")
    plt.title("Real-Time Simulation: Predictions Over Sliding Windows")
    plt.yticks([1, 2], ["Left", "Right"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(results_dir, "realtime_predictions.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_latency_over_time(prediction_log, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(14, 5))

    plt.plot(
        prediction_log.index,
        prediction_log["latency_ms"],
        linewidth=1.2,
    )

    plt.xlabel("Sliding window index")
    plt.ylabel("Latency (ms)")
    plt.title("Processing Latency Over Sliding Windows")
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(results_dir, "latency_over_time.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")