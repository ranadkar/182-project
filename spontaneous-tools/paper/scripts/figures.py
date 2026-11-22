"""Generate vector artwork at the paper's 5.5-inch text width."""
import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from prepare import ARMS, PAPER, SEEDS, STUDY, build, csv_rows, json_lines

OUT = PAPER / "figures"
LABELS = ["Unmodified", "Explicit request", "Factual ack.", "Neutral ack.", "Continuation"]
INK, BLUE, LIGHT = "#17242c", "#255c74", "#eaf0f2"
plt.rcParams.update({"svg.fonttype":"path", "svg.hashsalt":"cs182-paper", "font.family":"DejaVu Sans",
                     "font.size":8, "axes.spines.top":False, "axes.spines.right":False})


def save(fig, name):
    fig.savefig(OUT / (name + ".svg"), metadata={"Date": None}, facecolor="white")
    plt.close(fig)


def performance(data):
    fig, (rates, effects) = plt.subplots(1, 2, figsize=(5.5, 2.22))
    fig.subplots_adjust(left=.205, right=.98, bottom=.31, top=.865, wspace=.32)
    endpoints = [
        ("external", "Unsolicited target change", "o", BLUE),
        ("authorized", "Authorized continuation", "s", INK),
    ]
    rates.set_title("Absolute rates", fontsize=9)
    effects.set_title("Continuation minus explicit", fontsize=9)
    legend = []
    for endpoint, label, marker, color in endpoints:
        offset = -.16 if endpoint == "external" else .16
        for index, arm in enumerate(ARMS):
            value, low, high = data["arms"][arm][endpoint]["rate"]
            y = index + offset
            rates.hlines(y, low, high, color=color, linewidth=1)
            rates.plot(value, y, marker=marker, color=color, markersize=3.5)

        y = 1 if endpoint == "external" else 0
        value, low, high = data["paired_effects"][endpoint]
        effects.hlines(y, low, high, color=color, linewidth=1.5)
        point, = effects.plot(value, y, marker=marker, color=color, markersize=4.3,
                              linestyle="none", label=label)
        legend.append(point)
        estimate = f"{value:+.1f} [{low:+.1f}, {high:+.1f}]".replace("-", "−")
        label_x = .50 if endpoint == "external" else .98
        alignment = "center" if endpoint == "external" else "right"
        effects.text(label_x, y + .16, estimate, transform=effects.get_yaxis_transform(),
                     ha=alignment, va="bottom", fontsize=7.9,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": .7})

    rates.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], ylim=(4.55, -.55),
              yticks=range(5), yticklabels=LABELS, xlabel="First turns (%)")
    effects.axvline(0, color="#777777", linewidth=.85, linestyle=(0, (3, 2)), zorder=0)
    effects.set(xlim=(-60, 60), xticks=[-50, -25, 0, 25, 50], ylim=(-.45, 1.63),
                yticks=[], xlabel="Paired difference (percentage points)")
    for axis in (rates, effects):
        axis.spines["left"].set_visible(False)
        axis.tick_params(length=0, pad=3, labelsize=7.7)
        axis.set_axisbelow(True)
    rates.grid(axis="x", color="#dfe3e4", linewidth=.5)
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=8,
               handlelength=.8, columnspacing=2, bbox_to_anchor=(.5, -.01))
    save(fig, "performance")


def delivery(data):
    fig, ax = plt.subplots(figsize=(5.5, 2.38))
    fig.subplots_adjust(left=.205, right=.975, top=.87, bottom=.23)
    formats = [("user", "User statement", "o", BLUE, -.24),
               ("external", "External notification", "s", INK, 0),
               ("system", "System notification", "^", "#666666", .24)]
    for key, label, marker, color, offset in formats:
        rows = np.array([data["arms"][arm][key]["rate"] for arm in ARMS])
        y = np.arange(len(ARMS)) + offset
        ax.hlines(y, rows[:, 1], rows[:, 2], color=color, linewidth=1)
        ax.plot(rows[:, 0], y, marker=marker, markersize=3.8, color=color,
                linestyle="none", label=label)
    ax.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], ylim=(4.47, -.47),
           yticks=np.arange(len(ARMS)), yticklabels=LABELS,
           xlabel="First turns with a target-matching state change (%)")
    ax.grid(axis="x", color="#dfe3e4", linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, pad=4)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               handlelength=.8, columnspacing=1.5, bbox_to_anchor=(.53, 1.02))
    save(fig, "delivery")


def protocol(trace):
    r=json_lines(STUDY/"data/training.jsonl")[0]
    assert r["id"]=="training-00-000"
    card = json.loads(r["history"][-1]["content"])
    call = r["assistant"]["tool_calls"][0]["function"]
    target = f"{call['name']}(value={call['arguments']['value']})"
    observation = (f"Card lookup returned a specified {card['field']} of {card['specified']} "
                   f"and a current count of {card['verified_current']}.")
    fig=plt.figure(figsize=(5.5,3.93))
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,5.5),ylim=(-1.05,2.88)); ax.axis("off")
    def box(x,y,w,h,fill):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=INK,lw=.65))
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=6,
                                    linewidth=.65,color=INK,shrinkA=0,shrinkB=0))
    ax.text(.02,2.76,"Training under an authorized goal", weight="bold", size=9.3, va="center")
    box(.02,2.13,5.46,.46,LIGHT)
    ax.text(.13,2.45,r["history"][1]["content"],size=8.3,weight="bold",va="center")
    ax.text(.13,2.26,observation,size=8.1,va="center")
    box(.02,.02,5.46,.34,LIGHT)
    ax.text(.13,.19,"Shared assistant target",size=8,weight="bold",va="center")
    ax.text(2.32,.19,target,family="DejaVu Sans Mono",size=8,va="center")
    ys=[1.89,1.48,1.07,.66]
    ax.plot([.10,.10],[2.13,ys[-1]],lw=.65,color=INK)
    ax.plot([5.40,5.40],[ys[0],.40],lw=.65,color=INK)
    arrow(5.40,.44,5.40,.36)
    for y,arm,label in zip(ys,ARMS[1:],["Explicit request","Factual receipt","Neutral receipt","Continuation"]):
        arrow(.10,y,.25,y)
        box(.26,y-.18,4.97,.36,"white")
        ax.text(.34,y,label,weight="bold",fontsize=8.1,va="center")
        extra=r["extra"][arm] or "No new user turn."
        ax.text(1.66,y,"\n".join(textwrap.wrap(extra,49)),fontsize=8.1,va="center",linespacing=1.02)
        ax.plot([5.23,5.4],[y,y],lw=.65,color=INK)
    ax.text(.02, -.20, "Evaluation after an external notification", weight="bold", size=9.3, va="center")
    observation = "“" + trace["observation"].replace(". ", ".\n", 1) + "”"
    ax.text(.13, -.43, observation, size=8.4, va="center")
    action = trace["action"]
    compact_call = f'{action["name"]}(record="{action["arguments"]["record"]}")'
    ax.text(.13, -.73, "First call", size=8, weight="bold", va="center")
    ax.text(1.12, -.73, compact_call, family="DejaVu Sans Mono", size=8.2, va="center")
    transition = ", ".join(trace["initial"]) + "  →  " + ", ".join(trace["final"])
    ax.text(.13, -.93, "Roster", size=8, weight="bold", va="center")
    ax.text(1.12, -.93, transition, size=8.4, va="center")
    save(fig,"conditions")


def losses():
    rows=[r for r in csv_rows(STUDY/"results/loss_curves.csv") if r["pilot"]=="0"]
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.43))
    fig.subplots_adjust(left=.105,right=.98,bottom=.31,top=.88,wspace=.34)
    palette=[INK,BLUE,"#9b523b","#56754e"]
    for a,label,col,style in zip(ARMS[1:],LABELS[1:],palette,["-","--","-.",":"]):
        for ax,metric in zip(axs,["training_nll","validation_nll"]):
            runs=[]
            for seed in SEEDS:
                sel=sorted([r for r in rows if r["arm"]==a and r["seed"]==str(seed) and r[metric]],key=lambda r:int(r["step"]))
                x=np.array([int(r["step"]) for r in sel]); y=np.array([float(r[metric]) for r in sel])
                if metric=="training_nll": y=np.convolve(y,np.ones(4)/4,"valid");x=x[3:]
                runs.append(y)
            ax.plot(x,np.mean(runs,axis=0),style,color=col,label=label,lw=1.4)
    for ax,title in zip(axs,["Training","Validation"]):
        ax.set(xlim=(0,32), xticks=[0,16,32],ylim=(0,.18), title=title, xlabel="Optimizer update")
        ax.grid(axis="y",lw=.5,color="#dddddd")
    axs[0].set_ylabel("NLL per target token")
    fig.legend(*axs[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False,fontsize=7.2,columnspacing=.85)
    save(fig,"loss")


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    data=build()
    protocol(data["delivery_trace"])
    performance(data)
    delivery(data)
    losses()

