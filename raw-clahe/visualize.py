import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_latency_breakdown(df):
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    x         = df["Number"]
    loading_y = df["Loading_Time"]
    clahe_y   = df["CLAHE_Time"]
    bf_y      = df["BF_Time"]
    tm_y      = df["ToneMapping_Time"]
    d2h_y     = df["D2H_Time"]

    ax.bar(x, loading_y,                                              label="H2D Transfer",    color="#55a868", alpha=0.9, edgecolor="black", linewidth=0.4, width=0.55)
    ax.bar(x, clahe_y,   bottom=loading_y,                           label="CLAHE",            color="#4c72b0", alpha=0.9, edgecolor="black", linewidth=0.4, width=0.55)
    ax.bar(x, bf_y,      bottom=loading_y + clahe_y,                 label="Bilateral Filter", color="#8172b3", alpha=0.9, edgecolor="black", linewidth=0.4, width=0.55)
    ax.bar(x, tm_y,      bottom=loading_y + clahe_y + bf_y,          label="Tone Mapping",     color="#dd8452", alpha=0.9, edgecolor="black", linewidth=0.4, width=0.55)
    ax.bar(x, d2h_y,     bottom=loading_y + clahe_y + bf_y + tm_y,  label="D2H Transfer",     color="#c44e52", alpha=0.9, edgecolor="black", linewidth=0.4, width=0.55)

    ax.axhline(y=15.0, color="#8172b3", linestyle="--", linewidth=1.5, label="Pre-Processing Limit (15 ms)")

    ax.set_title("Latency Breakdown Per Stage", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("# of Images", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(df["Total_Pipeline"].max() * 1.2, 20.0))
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    output_filename = "preprocessing_latency_breakdown.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"\n>> 階段堆疊圖已儲存至：{os.path.abspath(output_filename)}")
    plt.close()


def plot_scatter(df):
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    ax.scatter(x=df["Number"], y=df["Total_Preprocessing"],
               color="blue", marker="o", label="Preprocessing Time (ms)", s=40, alpha=0.7)

    ax.set_title("Preprocessing Time Per Image", fontsize=14)
    ax.set_xlabel("# of Images", fontsize=12)
    ax.set_ylabel("Processing Time (ms)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    output_filename = "preprocessing_latency_chart.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"\n>> 散佈圖已儲存至：{os.path.abspath(output_filename)}")
    plt.close()
