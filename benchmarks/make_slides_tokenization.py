"""Render the tokenization-program slide deck to a landscape 16:9 PDF.

Academic seminar format for the tokenization working group + HEP/ML experts:
motivation -> data & protocol -> objective definitions -> evaluation
methodology (why each metric) -> Campaigns 1-5 (one slide each, with results
tables) -> Campaign 6 alternate schemes (hypotheses, results, systematics) ->
revised conclusions -> outlook -> references.

Numbers: benchmarks/README.md + benchmarks/figures/library_metrics.json
(committed eval output). Chart palette validated (dataviz six checks, white
surface): blue #2a78d6 = continuous/VAE · green #0e9e78 = physics grid · amber
#a9761a = VQ family · purple #8250df = FSQ recon · red #cf3b3f = FSQ+kernel.
"""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

PAPER="#ffffff"; SURF="#f5f8fa"; INK="#141b24"; INKS="#38444f"
MUT="#6b7883"; FAINT="#9aa7b2"; LINE="#d8dfe5"
SIGNAL="#0e9e78"; AMBER="#a9761a"; NEG="#cf3b3f"; POS=SIGNAL
BLUE="#2a78d6"; PURPLE="#8250df"
SER="DejaVu Serif"; SANS="DejaVu Sans"; MONO="DejaVu Sans Mono"
from pathlib import Path as _P
_REPO=_P(__file__).resolve().parent.parent
OUT=str(_REPO/"docs"/"tokenization_slides.pdf")
plt.rcParams.update({"pdf.fonttype":42,"font.family":SANS})

TOTAL=26

def emit(fig):
    pdf.savefig(fig,facecolor=PAPER); plt.close(fig)

def slide(n, dark_tick=SIGNAL):
    fig=plt.figure(figsize=(13.333,7.5), dpi=200, facecolor=PAPER)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Rectangle((0,0),1,1,color=PAPER,zorder=-5))
    ax.add_patch(Rectangle((0.06,0.905),0.032,0.005,color=dark_tick,zorder=2))
    ax.plot([0.06,0.94],[0.066,0.066],color=LINE,lw=0.8)
    ax.text(0.06,0.042,"heptokens · physics-defined tokenization for AI-ready HEP datasets",
            family=MONO,fontsize=8,color=FAINT,va="center")
    ax.text(0.94,0.042,f"{n:02d} / {TOTAL:02d}",family=MONO,fontsize=8,color=MUT,
            va="center",ha="right")
    return fig,ax

def eyebrow(ax,t,color=SIGNAL):
    ax.text(0.06,0.868,t.upper(),family=MONO,fontsize=10,color=color,va="center",
            fontweight="bold")
def title(ax,t,y=0.815,size=25):
    fig=ax.figure
    txt=ax.text(0.06,y,t,family=SER,fontsize=size,color=INK,va="top",fontweight="bold")
    fig.canvas.draw()
    maxw=(0.94-0.06)*fig.bbox.width
    w=txt.get_window_extent().width
    if w>maxw: txt.set_fontsize(size*maxw/w*0.98)
    return txt
def body(ax,t,y,size=13.5,color=INKS,width=100,x=0.06,lh=1.45,ha="left",family=None):
    ax.text(x,y,textwrap.fill(t,width=width),family=family or SANS,fontsize=size,
            color=color,va="top",ha=ha,linespacing=lh)

def bullets(ax,items,y,x=0.06,size=13,gap=None,width=100,lh=1.35):
    for t in items:
        if isinstance(t,tuple): t,c=t
        else: c=INKS
        wrapped=textwrap.fill(t,width=width)
        ax.text(x,y,"–",family=SANS,fontsize=size,color=MUT,va="top")
        ax.text(x+0.018,y,wrapped,family=SANS,fontsize=size,color=c,va="top",linespacing=lh)
        n=wrapped.count("\n")+1
        y-=(gap if gap else 0.031+0.0295*n)
    return y

def table(ax,cols,headers,rows,y0,rowh=0.047,fs=11,hfs=9,hl=None):
    """cols: [(x, ha)]; rows: list of list of str|(str,color)|(str,color,bold)."""
    x1=max(x for x,_ in cols)+0.004
    for (x,ha),h in zip(cols,headers):
        ax.text(x,y0,h.upper(),family=MONO,fontsize=hfs,color=MUT,va="top",ha=ha)
    ax.plot([cols[0][0],x1],[y0-0.026,y0-0.026],color=LINE,lw=0.8)
    yy=y0-0.042
    for i,row in enumerate(rows):
        if hl is not None and i in hl:
            ax.add_patch(Rectangle((cols[0][0]-0.008,yy-rowh+0.014),x1-cols[0][0]+0.016,
                                   rowh-0.002,color=SURF,zorder=0))
        for (x,ha),cell in zip(cols,row):
            t,c,b=(cell if isinstance(cell,tuple) else (cell,INKS,False)) if not (
                isinstance(cell,tuple) and len(cell)==2) else (cell[0],cell[1],False)
            if isinstance(cell,tuple) and len(cell)==3: t,c,b=cell
            ax.text(x,yy,t,family=MONO if ha=="right" else SANS,fontsize=fs,color=c,
                    va="top",ha=ha,fontweight="bold" if b else "normal")
        yy-=rowh
    return yy

def chart_axes(fig,rect):
    iax=fig.add_axes(rect)
    for s in ("top","right"): iax.spines[s].set_visible(False)
    for s in ("left","bottom"): iax.spines[s].set_color(LINE)
    iax.tick_params(colors=MUT,labelsize=9,length=3)
    iax.grid(axis="y",color=LINE,lw=0.6,alpha=0.7,zorder=0)
    iax.set_axisbelow(True)
    return iax

def setup_line(ax,t,y=0.105):
    ax.text(0.06,y,t,family=MONO,fontsize=8.5,color=FAINT,va="center")

with PdfPages(OUT) as pdf:
    # ---- 1 · title ---------------------------------------------------------
    fig,ax=slide(1)
    ax.text(0.06,0.70,"Physics-Defined Tokenization for",family=SER,fontsize=37,
            color=INK,va="center",fontweight="bold")
    ax.text(0.06,0.625,"AI-Ready HEP Datasets",family=SER,fontsize=37,color=INK,
            va="center",fontweight="bold")
    ax.text(0.06,0.515,"Six experimental campaigns: the particle-library objective, discrete and",
            family=SER,fontsize=17,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.472,"continuous bottlenecks, and alternate tokenization schemes.",
            family=SER,fontsize=17,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.355,"heptokens working group",family=SANS,fontsize=13,color=INKS,va="center")
    ax.text(0.06,0.315,"Data: CMS Aspen Open Jets RunG (1.96 M jets) · 620 k-jet 5-class benchmark",
            family=SANS,fontsize=11.5,color=MUT,va="center")
    ax.text(0.06,0.278,"August 2026",family=SANS,fontsize=11.5,color=MUT,va="center")
    import numpy as np; rng=np.random.default_rng(6)
    for gx,gy,c in [(0.84,0.58,SIGNAL),(0.89,0.41,AMBER),(0.82,0.28,BLUE),(0.87,0.65,MUT)]:
        for _ in range(9):
            a=rng.random()*6.28; r=rng.random()**0.6*0.035
            ax.add_patch(plt.Circle((gx+np.cos(a)*r,gy+np.sin(a)*r*1.6),0.006,
                color=c,alpha=0.45,zorder=0))
    emit(fig)

    # ---- 2 · outline ---------------------------------------------------------
    fig,ax=slide(2); eyebrow(ax,"outline"); title(ax,"Outline")
    items=[("1","Introduction — the AI-ready-dataset mission and the tokenizer question"),
           ("2","Data, objectives and evaluation methodology — what we measure, and why"),
           ("3","Campaigns 1–5 — generic SSL, the particle library, hybrid objectives, compression, VAE vs VQ"),
           ("4","Campaign 6 — alternate tokenization schemes: hypotheses, results, systematic effects"),
           ("5","Summary — revised conclusions, open problems, outlook")]
    yy=0.68
    for n_,t_ in items:
        ax.text(0.08,yy,n_,family=MONO,fontsize=16,color=SIGNAL,va="top",fontweight="bold")
        ax.text(0.125,yy,t_,family=SANS,fontsize=15.5,color=INK,va="top")
        yy-=0.093
    emit(fig)

    # ---- 3 · motivation --------------------------------------------------------
    fig,ax=slide(3); eyebrow(ax,"1 · introduction")
    title(ax,"Motivation: the token dataset is the product")
    yy=bullets(ax,[
        "Experiments reconstruct particle-flow candidates that external groups cannot access; those groups "
        "want to train foundation models on the events. We publish a per-constituent token as the shareable "
        "artifact — versioned by dimension, preprocessing hash and checkpoint.",
        "Release requirements: information sufficiency for unknown future tasks; detector abstraction; "
        "an auditable statement of what the token forgets; compression at the 178 M-jet scale.",
        "Baseline tokenizer (pre-study): a residual VQ-VAE emitting integer indices. Known concerns: "
        "codebook collapse, an implicit metric defined by the reconstruction objective, and lossiness by construction.",
        ],0.72,size=13.5,width=105)
    ax.add_patch(FancyBboxPatch((0.06,0.155),0.85,0.135,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.168),0.005,0.108,color=SIGNAL,zorder=2))
    ax.text(0.085,0.268,"RESEARCH QUESTION",family=MONO,fontsize=9,color=SIGNAL,va="top")
    ax.text(0.085,0.235,"Which objective and which bottleneck should define the released per-particle token —",
            family=SANS,fontsize=13.5,color=INK,va="top")
    ax.text(0.085,0.198,"and can any single token be simultaneously compact, information-faithful, and criterion-organized?",
            family=SANS,fontsize=13.5,color=INK,va="top")
    emit(fig)

    # ---- 4 · data & protocol -----------------------------------------------------
    fig,ax=slide(4); eyebrow(ax,"2 · data & methodology")
    title(ax,"Datasets and fairness protocol")
    ax.text(0.06,0.70,"Datasets",family=SANS,fontsize=14,color=INK,va="top",fontweight="bold")
    bullets(ax,[
        "jets5c — 620 k jets, 5 balanced classes, ≤150 constituents × (pT, Δη, Δφ). Labelled: used for "
        "consumer-view downstream probes.",
        "Aspen Open Jets RunG — 1.96 M real CMS 2016 jets; constituents (pT, η, φ, PID, charge), "
        "PDG ∈ {±211, ±13, ±11, 22, 130}. Unlabelled: all criterion evaluations.",
        ],0.655,size=12.5,width=104)
    ax.text(0.06,0.435,"Protocol (held fixed across every comparison)",family=SANS,fontsize=14,
            color=INK,va="top",fontweight="bold")
    bullets(ax,[
        "Matched budgets: identical encoders, inputs, splits (70/15/15, seed 42), schedules (15–20 epochs); "
        "only the objective/bottleneck under test differs.",
        "Consumer view: downstream probes see only the exported token files — never the constituents.",
        "Untrained nulls are mandatory: PID-one-hot ⊕ scaled kinematics (192 b); from C6, a criterion-aligned "
        "feature grid (~23–26 b). Learning must beat the null to claim rent.",
        "Seeds: 3 seeds wherever a comparison is promoted to a claim; single-seed numbers are flagged.",
        ],0.39,size=12.5,width=104)
    setup_line(ax,"hardware: single A100 MIG 2g.20gb slice · torch 2.5.1 · full config in benchmarks/")
    emit(fig)

    # ---- 5 · objectives -------------------------------------------------------------
    fig,ax=slide(5); eyebrow(ax,"2 · data & methodology")
    title(ax,"Objective definitions")
    ax.add_patch(FancyBboxPatch((0.06,0.50),0.86,0.235,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.515),0.005,0.205,color=SIGNAL,zorder=2))
    ax.text(0.085,0.715,"PARTICLE LIBRARY (method of record, C2)",family=MONO,fontsize=9.5,color=SIGNAL,va="top")
    ax.text(0.085,0.678,"Positives:  same PID  ∧  same charge  ∧  ΔpT/pT ≤ 0.10  ∧  |Δη| ≤ 0.1  ∧  |Δφ| ≤ 0.1",
            family=MONO,fontsize=12.5,color=INK,va="top")
    bullets(ax,[
        "Context-free per-particle MLP Φ (PID/charge embedded); the same particle maps to the same token in any event.",
        "Graded Gaussian kernel soft-target contrastive loss: hard gates on PID/charge, smooth grading on the "
        "kinematic windows. No augmentations, no views — the metric is stated in physics variables.",
        ],0.635,x=0.085,size=12,width=104)
    ax.add_patch(FancyBboxPatch((0.06,0.265),0.86,0.185,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.278),0.005,0.157,color=AMBER,zorder=2))
    ax.text(0.085,0.43,"VQ-VAE BASELINE (matched, C2b)",family=MONO,fontsize=9.5,color=AMBER,va="top")
    bullets(ax,[
        "Same encoder → residual VQ (n_q × K codebook, straight-through + EMA) → decoder reconstructing the "
        "full particle (L1 kinematics + CE PID + CE charge). Rate = n_q · log₂K bits/particle.",
        "Deliberately not straw-manned: PID and charge are forced through the bottleneck.",
        ],0.40,x=0.085,size=12,width=98)
    body(ax,"Design rule (project convention): before training, state what each objective makes the embedding "
            "forget; generic SSL recipes appear only as ablations.",0.20,size=12,color=MUT,width=105)
    emit(fig)

    # ---- 6 · metrics I ------------------------------------------------------------
    fig,ax=slide(6); eyebrow(ax,"2 · data & methodology")
    title(ax,"Primary metric: availability-aware criterion recall@8")
    ax.add_patch(FancyBboxPatch((0.06,0.575),0.86,0.14,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.text(0.085,0.685,"recall@8  =  mean over anchors a with ≥1 partner of   |top-8 neighbours(a) ∩ criterion-partners(a)|  /  min(8, #partners(a))",
            family=MONO,fontsize=11,color=INK,va="top")
    ax.text(0.085,0.635,"cosine neighbours in a 30 k-constituent pool · criterion partners computed exactly from raw features",
            family=MONO,fontsize=10,color=MUT,va="top")
    ax.text(0.06,0.51,"Why not precision@k?",family=SANS,fontsize=14,color=INK,va="top",fontweight="bold")
    bullets(ax,[
        "Criterion density in the pool is ~3·10⁻⁵: a typical anchor has far fewer than k true partners "
        "available, so precision@k is bounded by availability, not by embedding quality — it flattens all "
        "methods to the same small number.",
        "Recall against the availability ceiling (divide by min(k, #partners)) is 1.0 for a perfect embedding "
        "at any density, and it is what the release promises: partners that exist are found.",
        "recall@8 is the library objective's native metric — it is flagged as such in every table, and "
        "reconstruction-flavoured metrics are always reported beside it.",
        ],0.465,size=12.5,width=104)
    setup_line(ax,"a tighter variant (ΔpT ≤ 5%) is reported as recall@8-dpt5; both are computed by benchmarks/eval_library.py")
    emit(fig)

    # ---- 7 · metrics II -------------------------------------------------------------
    fig,ax=slide(7); eyebrow(ax,"2 · data & methodology")
    title(ax,"The metric suite: each entry guards against a known failure")
    cols=[(0.06,"left"),(0.335,"left"),(0.63,"left")]
    rows=[
        ["criterion recall@8","criterion geometry (native)","precision@k availability ceiling (~3·10⁻⁵ density)"],
        ["kNN PID purity","species organization (neutral)","alone it misleads: 0.996 purity with 0.383 recall (C5)"],
        ["within-PID pT R²","kinematic information retention","catches PID-shortcut tokens; release needs observables"],
        ["linear PID/charge probes","consumer linear decodability","library structure is metric, not linear (probe ≈ 0.37)"],
        ["effective rank","dimensional collapse","VICReg/VQ collapse modes (C1: quantizer-0 <10 % codes)"],
        ["bits / particle","release rate","n_q·log₂K · fp16 dim·16 · KL/ln2 (caveat: slide 18)"],
        ["downstream probe (jets5c)","consumer value of the export","tokens are the product; probes never see constituents"],
    ]
    table(ax,cols,["metric","what it measures","failure it guards against"],rows,0.72,rowh=0.062,fs=11)
    body(ax,"No single metric is trusted alone: every campaign table reports the full suite, with the "
            "objective-native column flagged.",0.155,size=12,color=MUT,width=105)
    emit(fig)

    # ---- 8 · campaign 1 ---------------------------------------------------------------
    fig,ax=slide(8); eyebrow(ax,"3 · campaigns 1–5")
    title(ax,"Campaign 1 — generic SSL vs VQ-VAE (jets5c, consumer view)")
    body(ax,"Setup: identical 4-layer transformer encoders; smearing-VICReg contrastive (dim 16) vs residual "
            "VQ (512×3, dim 16); probes see only the exported tokens.",0.735,size=12.5,width=108)
    cols=[(0.06,"left"),(0.44,"right"),(0.58,"right"),(0.74,"right")]
    rows=[
        ["raw features (ceiling)","0.743","0.931","0.249"],
        ["contrastive tokens","0.701","0.917",("0.486",INK,True)],
        ["VQ tokens (baseline)",("0.748",INK,True),("0.934",INK,True),"0.243"],
    ]
    table(ax,cols,["interface","probe acc","macro AUC","kNN class purity"],rows,0.66,rowh=0.05)
    bullets(ax,[
        "The augmentation list silently defines the metric: smearing pT traded away pT sensitivity "
        "(distance–ΔpT correlation 0.31 vs 0.61) and cost ≈1.5 AUC downstream.",
        "VQ pathology observed live: quantizer 0 collapsed to <10 % codebook utilization.",
        "Anti-collapse machinery works (effective rank 8.3/16, all dims active) — collapse is not the "
        "argument against continuous tokens; the implicit metric is.",
        ],0.44,size=12.5,width=104)
    setup_line(ax,"620 k jets · 20 epochs · matched budgets · probes: 2-layer transformer on 250 k jets")
    emit(fig)

    # ---- 9 · campaign 2 -----------------------------------------------------------------
    fig,ax=slide(9); eyebrow(ax,"3 · campaigns 1–5")
    title(ax,"Campaign 2 — the particle library on real CMS data (AOJ)")
    body(ax,"Setup: the explicit physics criterion replaces the augmentation-implied metric; context-free MLP; "
            "no labels, no views. Evaluation: criterion recall@8 in a 30 k pool.",0.735,size=12.5,width=108)
    cols=[(0.06,"left"),(0.46,"right"),(0.60,"right"),(0.73,"right")]
    rows=[
        [("library (graded kernel)",INK,True),("0.9992",POS,True),"0.910","0.18"],
        ["library (hard windows)","0.9935","0.664","0.30"],
        ["null: PID-one-hot ⊕ kinematics","0.9833","0.993",("0.95",INK,True)],
        ["raw kinematics only","0.300","0.290","0.95"],
        ["v1 jet-context transformer",("0.052",NEG,True),"0.284","0.84"],
    ]
    table(ax,cols,["provider (AOJ)","recall@8","kNN purity","pT R²"],rows,0.66,rowh=0.047)
    bullets(ax,[
        "Criterion achieved: 20× fewer retrieval misses than the untrained null — the learning pays rent on "
        "the criterion itself. Graded kernel beats hard windows on every axis (no window-edge cliff).",
        "Context-free is decisive: jet-context encoding destroys the library property (0.052).",
        "Known deficiency at this stage: within-species pT recovery R² 0.18 → motivates Campaign 3.",
        ],0.37,size=12.5,width=104)
    setup_line(ax,"AOJ RunG · 1.96 M jets · 15 epochs · seed 42")
    emit(fig)

    # ---- 10 · campaign 2b -----------------------------------------------------------------
    fig,ax=slide(10); eyebrow(ax,"3 · campaigns 1–5")
    title(ax,"Campaign 2b — matched library vs VQ: complementarity")
    body(ax,"Setup: same data, inputs, encoder, schedule; only the objective differs. 3 seeds each; the VQ arm "
            "reconstructs the full particle so identity must pass the bottleneck.",0.735,size=12.5,width=108)
    cols=[(0.06,"left"),(0.40,"right"),(0.53,"right"),(0.65,"right"),(0.76,"right"),(0.87,"right")]
    rows=[
        [("library",INK,True),("0.9991±.0002",POS,True),"0.905","0.366","0.169","256"],
        [("VQ, matched (RVQ 3×512)",INK,True),"0.9271±.0062",("0.996",INK,True),("1.000",INK,True),("0.971",INK,True),("27",INK,True)],
        ["library → post-hoc RVQ","0.832","0.863","0.380","0.174","27"],
        ["null (untrained)","0.9833","0.993","1.000","0.949","—"],
    ]
    table(ax,cols,["interface","recall@8","purity","pid probe","pT R²","bits"],rows,0.665,rowh=0.047,fs=10.5)
    bullets(ax,[
        "Complementary, not competing: the library owns criterion geometry (~80× fewer misses); the VQ owns "
        "information sufficiency (PID/charge linearly decodable, pT R² 0.97).",
        "Library species structure is metric, not linear: kNN purity 0.91 but linear PID probe ≈ majority class.",
        "No downstream cost: library tokens probe at the raw-feature ceiling on jets5c (0.744 acc / 0.931 AUC).",
        ],0.375,size=12.5,width=104)
    setup_line(ax,"AOJ · 3 seeds · matched ParticleEncoder MLP 128-128-64, dim 16")
    emit(fig)

    # ---- 11 · campaign 3 ---------------------------------------------------------------------
    fig,ax=slide(11); eyebrow(ax,"3 · campaigns 1–5")
    title(ax,"Campaign 3 — hybrid objective and token dimensionality")
    body(ax,"Question: can a reconstruction term (weight w) restore kinematic information without breaking the "
            "criterion — and how small can the token be?",0.735,size=12.5,width=108)
    cols=[(0.06,"left"),(0.42,"right"),(0.57,"right"),(0.72,"right")]
    rows=[
        ["kernel only, dim 16",("0.9992",POS,True),"0.175","0.906"],
        ["hybrid w=1.0","0.9984","0.867","0.915"],
        ["hybrid w=3.0","0.9955",("0.908",INK,True),"0.935"],
        [("hybrid w=2.0, dim 8",INK,True),"0.9982","0.613",("0.987",INK,True)],
        ["hybrid w=2.0, dim 4","0.9801","0.799","0.987"],
    ]
    table(ax,cols,["run (AOJ)","recall@8","pT R²","kNN purity"],rows,0.665,rowh=0.046)
    bullets(ax,[
        "w is a smooth, controllable frontier between organization and information — but no single weight "
        "reaches the target corner (recall ≥ 0.999 ∧ R² ≥ 0.9): best point misses recall by ~0.004.",
        "The token halves for free: 16→8 dims raises recall and purity; dim 4 is the floor.",
        ],0.35,size=12.5,width=104)
    setup_line(ax,"AOJ · 15 epochs · seed 42 · target corner quantified by C2b")
    emit(fig)

    # ---- 12 · campaign 4 ----------------------------------------------------------------------
    fig,ax=slide(12); eyebrow(ax,"3 · campaigns 1–5")
    title(ax,"Campaign 4 — compression: shape the codebook, or shrink the token?")
    body(ax,"Question: the continuous token is anti-compressive (96–256 b vs ~50 b raw). Does criterion-shaping "
            "a VQ codebook give a compact token that keeps the criterion?",0.735,size=12.5,width=108)
    cols=[(0.06,"left"),(0.44,"right"),(0.56,"right"),(0.70,"right")]
    rows=[
        [("library dim 6 (continuous)",INK,True),("96",INK,True),("0.9996",POS,True),("0.743",INK,True)],
        ["library dim 4 (continuous)","64","0.9807","0.684"],
        ["plain VQ (RVQ 3×512)","27","0.927±.006","0.971"],
        [("contrastive-VQ, kernel→z_q",NEG,False),"27",("0.708",NEG,True),"0.719"],
        [("contrastive-VQ, kernel→z_e",NEG,False),"27",("0.772",NEG,True),"0.295"],
        ["VQ single 8192 codebook","13",("0.028",NEG,True),"0.821"],
    ]
    table(ax,cols,["artifact","bits","recall@8","pT R²"],rows,0.665,rowh=0.0435,fs=10.5)
    bullets(ax,[
        "Headline negative: the kernel makes the VQ worse than plain reconstruction at equal rate, on both "
        "placements — the kernel spreads representation to effective rank 15+, which quantization cannot tile.",
        "Compression works on the continuous side: dim-6 (96 b) sharpens the criterion and raises pT R².",
        "Granularity beats bit-count: the single 8192 codebook collapses to 1.7 % utilization.",
        ],0.335,size=12,width=104)
    setup_line(ax,"AOJ · 15 epochs · seed 42 (plain VQ: 3 seeds) · 8192 arm at batch 256 (memory) — flagged as a confound in C6 review")
    emit(fig)

    # ---- 13 · campaign 5 ------------------------------------------------------------------------
    fig,ax=slide(13); eyebrow(ax,"3 · campaigns 1–5")
    title(ax,"Campaign 5 — a 2×2×2 factorial: bottleneck × objective × context")
    body(ax,"Factors: bottleneck {continuous VAE · RVQ 3×1024 = 30 b} × objective {reconstruction · + physics "
            "kernel on the latent} × encoder {context-free MLP · attention}. Everything else matched.",
            0.735,size=12.5,width=108)
    cols=[(0.06,"left"),(0.44,"right"),(0.57,"right"),(0.70,"right")]
    rows=[
        ["VAE dim 4, recon only","2.6 (KL)",("0.383",NEG,True),"0.836"],
        [("VAE dim 4 + kernel",INK,True),"1.4 (KL)",("0.916",POS,True),"0.665"],
        [("plain VQ 3×1024",INK,True),"30",("0.945",INK,True),("0.970",INK,True)],
        [("contrastive-VQ (kernel→z_e)",NEG,False),"30",("0.827",NEG,True),"0.331"],
        ["VAE recon + attention","2.6 (KL)",("0.051",NEG,True),"0.479"],
        ["plain VQ + attention","30","0.967","0.967"],
    ]
    table(ax,cols,["artifact (AOJ)","bits","recall@8","pT R²"],rows,0.66,rowh=0.0435,fs=10.5)
    bullets(ax,[
        "Placement is decisive: the same kernel lifts the continuous latent (0.383→0.916) and hurts the "
        "codebook (0.945→0.827) — the clean-gradient continuous quadrant is where physics shaping works.",
        "Context is asymmetric: it helps the 30-bit VQ slightly but destroys the low-rate token (0.383→0.051).",
        ],0.325,size=12,width=104)
    setup_line(ax,"AOJ · 20 epochs · seed 42 · VAE rate = KL/ln2 (revisited on slide 18)")
    emit(fig)

    # ---- 14 · interim summary --------------------------------------------------------------------
    fig,ax=slide(14); eyebrow(ax,"3 · campaigns 1–5")
    title(ax,"Interim summary after five campaigns")
    yy=0.71
    for n_,h_,s_ in [
        ("i","Explicit physics criterion ≫ augmentation-implied metric","0.9992 vs 0.052; the augmentation list is a hidden metric choice."),
        ("ii","Criterion and information are separable currencies","library vs VQ complementarity; w-dial frontier; corner unreached."),
        ("iii","The contrastive term needs a smooth continuous latent","works on μ, hurts through STE+EMA codebooks (twice, both placements)."),
        ("iv","Compress the continuous token; do not quantize the criterion","dim-6 at 96 b dominates the VQ compression curve."),
        ("v","Context-free encoding preserves the library property","required for the cross-experiment goal; context helps only at high rate.")]:
        ax.text(0.06,yy,n_,family=SER,fontsize=14,color=SIGNAL,va="top",fontweight="bold",fontstyle="italic")
        ax.text(0.11,yy,h_,family=SANS,fontsize=14.5,color=INK,va="top",fontweight="bold")
        ax.text(0.11,yy-0.037,s_,family=SANS,fontsize=12,color=MUT,va="top")
        yy-=0.098
    ax.add_patch(FancyBboxPatch((0.06,0.115),0.86,0.085,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.text(0.085,0.178,"OPEN PROBLEM",family=MONO,fontsize=9,color=NEG,va="top")
    ax.text(0.21,0.178,"No trained token is simultaneously sub-raw (≲50 b), faithful (R² ≥ 0.9), and criterion-perfect (recall ≥ 0.999).",
            family=SANS,fontsize=12,color=INK,va="top")
    emit(fig)

    # ---- 15 · alternate schemes survey -----------------------------------------------------------
    fig,ax=slide(15); eyebrow(ax,"4 · campaign 6 — alternate schemes")
    title(ax,"Alternate tokenization schemes: what each one probes")
    body(ax,"A full-repo audit + 2023–26 literature survey (discrete / continuous / HEP-specific), then an "
            "adversarial pass over every recorded conclusion. Each scheme isolates one mechanism:",
            0.75,size=12.5,width=118)
    cols=[(0.06,"left"),(0.27,"left"),(0.84,"left")]
    rows=[
        [("β-sweep (rate-matched VAE)",INK,True),"is the C5 VAE-vs-VQ verdict a rate-mismatch artifact? traces the R–D–criterion surface",("→ 6a",SIGNAL,True)],
        [("physics grid",INK,True),"does any learned discrete token beat untrained, criterion-aligned binning of raw features?",("→ 6b",SIGNAL,True)],
        [("FSQ (codebook-free)",INK,True),"was the criterion-shaped-VQ failure codebook/EMA dynamics — or quantization itself?",("→ 6c",SIGNAL,True)],
        ["rotation-trick · SimVQ","gradient geometry through the quantizer; dead-code dynamics behind q0 collapse","queued"],
        ["Gumbel / dVAE","stochastic relaxation as the discrete analogue of the clean-gradient placement result","queued"],
        ["REPA distillation","frozen-target regression vs live co-training: is the C3 corner an optimization artifact?","queued"],
        ["symbols ⊕ latent","exact categorical channel: restores linear decodability by construction","queued"],
        ["flow tokens","bijective map ⇒ lossless; how much criterion fits in the leading dimensions?","queued"],
        ["masked-continuous FM probe","which export does a foundation-model consumer actually prefer?",("→ C7",BLUE,True)],
    ]
    table(ax,cols,["scheme","mechanism isolated","status"],rows,0.665,rowh=0.052,fs=10.5)
    emit(fig)

    # ---- 16 · C6 design ---------------------------------------------------------------------------
    fig,ax=slide(16); eyebrow(ax,"4 · campaign 6 — alternate schemes")
    title(ax,"Campaign 6 design: three arms, pre-registered hypotheses")
    yy=0.71
    for tag,c,h_,d1,d2 in [
        ("6a",BLUE,"Rate-matched contrastive-VAE frontier","β × dim × free-bits sweep, kernel on the posterior mean μ; 11 arms.",
         "H1: at VQ-level rate the VAE matches VQ on information (bar: recall ≥ 0.945 ∧ R² ≥ 0.97 at ≤ 30 b)."),
        ("6b",SIGNAL,"Criterion-aligned physics grid (untrained)","log-pT bins of ln(1.1), η/φ bins of 0.1, exact PID/charge; a half-bin-offset second grid repairs boundary pairs. Evaluation-only.",
         "H2: trained discrete tokens outperform criterion-aligned binning (any learned token below the grid is retired)."),
        ("6c",NEG,"FSQ-bottleneck library tokenizer","per-dimension rounding of a bounded latent; zero learned parameters, exact designer-chosen bits; kernel on pre-rounding z_e (z_q and reconstruction ablations); 8 arms.",
         "H3: the C4/C5 negative was codebook/EMA dynamics — removing them rescues the criterion-shaped discrete token.")]:
        ax.text(0.06,yy,tag,family=MONO,fontsize=13,color=c,va="top",fontweight="bold")
        ax.text(0.115,yy,h_,family=SANS,fontsize=14,color=INK,va="top",fontweight="bold")
        ax.text(0.115,yy-0.036,textwrap.fill(d1,width=104),family=SANS,fontsize=11.5,color=INKS,va="top",linespacing=1.3)
        d2y=yy-0.036-0.031*(textwrap.fill(d1,width=104).count("\n")+1)-0.008
        ax.text(0.115,d2y,textwrap.fill(d2,width=104),family=SANS,fontsize=11.5,color=MUT,va="top",linespacing=1.3)
        yy-=0.20
    setup_line(ax,"23 training runs · AOJ · context-free MLP · 20 epochs · headline arms at 3 seeds · outcomes pre-registered in docs/research/campaign6_alternate_schemes.md")
    emit(fig)

    # ---- 17 · 6a results chart ----------------------------------------------------------------------
    fig,ax=slide(17); eyebrow(ax,"4 · campaign 6 — results")
    title(ax,"6a — recall and pT R² along the β dial (H1: rejected)")
    betas=[0.03,0.1,0.3,1,3]
    rec_d6=[0.9996,0.992,0.977,0.974,0.9995]; r2_d6=[0.757,0.767,0.821,0.803,0.568]
    kl_d6=[13.3,10.7,0.57,0.18,0.001]
    d4=[(0.1,0.970,0.673),(0.3,0.951,0.634)]; d8=[(0.1,0.996,0.643),(0.3,0.977,0.746)]
    recon=(0.1,0.976,0.920)
    for ip,(ylab,y6,ci,ry,ref,rlab,ylim) in enumerate([
        ("criterion recall@8",rec_d6,1,recon[1],0.945,"plain VQ 30 b  0.945",(0.94,1.006)),
        ("within-PID pT R²",r2_d6,2,recon[2],0.970,"plain VQ 30 b  0.970",(0.50,1.01))]):
        iax=chart_axes(fig,[0.085+ip*0.47,0.225,0.38,0.50])
        iax.set_xscale("log"); iax.set_xlim(0.02,5); iax.set_ylim(*ylim)
        iax.axhline(ref,color=AMBER,lw=1.4,ls=(0,(4,3)),zorder=1)
        iax.text(4.6,ref,rlab,family=MONO,fontsize=8.5,color=AMBER,va="bottom",ha="right")
        iax.plot(betas,y6,color=BLUE,lw=2,marker="o",ms=7,zorder=3,label="kernel VAE · dim 6")
        iax.plot([b for b,_,_ in d4],[v[ci] for v in d4],ls="none",marker="^",ms=7,
                 markerfacecolor="none",markeredgecolor=BLUE,markeredgewidth=1.6,zorder=3,label="dim 4")
        iax.plot([b for b,_,_ in d8],[v[ci] for v in d8],ls="none",marker="s",ms=6.5,
                 markerfacecolor="none",markeredgecolor=BLUE,markeredgewidth=1.6,zorder=3,label="dim 8")
        iax.plot([recon[0]],[ry],ls="none",marker="*",ms=13,color=PURPLE,zorder=4,
                 label="recon-only · dim 6")
        iax.set_xlabel("β  (KL weight, log)",fontsize=10,color=INKS)
        iax.set_ylabel(ylab,fontsize=10,color=INKS)
        if ip==0:
            for b,y,k in zip(betas,rec_d6,kl_d6):
                iax.annotate(f"{k:g} bits",(b,y),xytext=(0,-14),textcoords="offset points",
                             fontsize=7.5,family=MONO,color=MUT,ha="center")
            iax.legend(fontsize=8.5,loc="lower left",frameon=False)
    body(ax,"Recall reaches 0.9984 ± 0.0015 at β = 0.03 (3 seeds) — parity with the 96-bit deterministic dim-6 "
            "library — but R² peaks at ≈ 0.82: the retirement criterion (R² ≥ 0.97) is not met at any β. "
            "Complementarity survives rate-matching.",0.115,size=11.5,color=INKS,width=145)
    emit(fig)

    # ---- 18 · KL artifact -------------------------------------------------------------------------------
    fig,ax=slide(18); eyebrow(ax,"4 · campaign 6 — systematic effects",color=NEG)
    title(ax,"Systematic effect: ELBO rate vs information in a deterministic release")
    ax.add_patch(FancyBboxPatch((0.06,0.60),0.86,0.115,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.text(0.085,0.685,"OBSERVATION (β = 3, dim 6)",family=MONO,fontsize=9,color=NEG,va="top")
    ax.text(0.085,0.652,"measured KL rate ≈ 0.001 bits/particle   —   yet criterion recall@8 = 0.9995",
            family=MONO,fontsize=13.5,color=INK,va="top")
    bullets(ax,[
        "Mechanism: the released token is the posterior mean μ, a deterministic function of the input. At high "
        "β the posterior collapses onto the prior (KL → 0) while μ retains a tiny-norm but structured layout — "
        "and cosine retrieval is scale-invariant, so the criterion geometry survives at 'zero' rate.",
        "The KL rate correctly prices transmitting posterior samples (bits-back), not the deterministic μ we "
        "actually export. It is therefore not a valid release-rate claim for this token family.",
        "Consequence for prior results: Campaign 5's '~1.4-bit' contrastive VAE was this effect in milder form. "
        "Until an operational entropy codec exists, continuous rates are quoted as fp16 dim×16, and KL-bits "
        "are reported as a diagnostic only. free-bits (0.05/dim) does not repair the accounting (0.41 b, 0.974).",
        ],0.545,size=12.5,width=106)
    setup_line(ax,"all rate columns in the campaign-6 tables and figures follow this convention")
    emit(fig)

    # ---- 19 · recon-only revision ---------------------------------------------------------------------------
    fig,ax=slide(19); eyebrow(ax,"4 · campaign 6 — systematic effects",color=NEG)
    title(ax,"Revisiting C5: 'reconstruction buys species, not the criterion'")
    cols=[(0.06,"left"),(0.42,"right"),(0.55,"right"),(0.68,"right"),(0.82,"right")]
    rows=[
        ["recon-only VAE, β=1 (C5)","2.6 (KL)",("0.383",NEG,True),"0.836","1.00"],
        [("recon-only VAE, β=0.1 (C6)",INK,True),"11.0 (KL)",("0.976±.004",INK,True),("0.920±.011",INK,True),"1.00"],
        ["null (untrained)","—","0.9833","0.949","1.00"],
        ["library / kernel VAE (β=.03)","96 (fp16)",("0.9984±.0015",POS,True),"0.758","0.59"],
    ]
    table(ax,cols,["arm","bits","recall@8","pT R²","pid probe"],rows,0.70,rowh=0.049,fs=10.5)
    bullets(ax,[
        "The C5 conclusion was a rate artifact, not a property of the objective: at β = 0.1 the recon-only VAE "
        "recovers 0.976 recall with 0.920 R² and perfect linear probes (3 seeds).",
        "It still does not exceed the untrained null on either axis — the physics kernel remains the only "
        "objective that has (0.9984 vs 0.9833).",
        "Methodological lesson: single-β conclusions about objectives conflate the objective with its rate.",
        ],0.44,size=12.5,width=104)
    setup_line(ax,"AOJ · 20 epochs · 3 seeds on both revised arms")
    emit(fig)

    # ---- 20 · 6b grid ------------------------------------------------------------------------------------------
    fig,ax=slide(20); eyebrow(ax,"4 · campaign 6 — results")
    title(ax,"6b — the untrained physics grid vs learned discrete tokens (H2: rejected)")
    cols=[(0.06,"left"),(0.44,"right"),(0.57,"right"),(0.70,"right"),(0.84,"right")]
    rows=[
        [("physics grid, double",INK,True),("26.3",INK,True),("0.9655",POS,True),"0.949","1.00"],
        ["physics grid, single","23.3","0.9553","0.948","1.00"],
        ["plain VQ 3×1024 (trained)","30","0.945",("0.970",INK,True),"1.00"],
        ["plain VQ 3×512 (trained)","27","0.927±.006","0.971","1.00"],
        ["library → post-hoc RVQ","27","0.832","0.174","0.38"],
        ["FSQ recon-only (trained)","24","0.851","0.840","0.81"],
    ]
    table(ax,cols,["discrete interface","bits","recall@8","pT R²","pid probe"],rows,0.70,rowh=0.046,fs=10.5)
    bullets(ax,[
        "The zero-parameter grid exceeds every learned discrete token on recall at equal or fewer bits; the "
        "trained VQ's surviving advantage is +0.02 pT R² and +0.017 purity.",
        "At the 23–30-bit tier, training a discrete token currently buys almost nothing over criterion-aligned "
        "binning — and the grid is computable on any detector, a natural cross-experiment release tier.",
        "H2 fails below the pre-registered auto-retirement threshold (0.99) but the practical ordering stands.",
        ],0.40,size=12.5,width=104)
    setup_line(ax,"grid = ln(1.1) log-pT × 0.1 η × 0.1 φ bins ⊕ exact PID/charge; double grid adds a half-bin-offset copy (~+3 b)")
    emit(fig)

    # ---- 21 · 6c FSQ map --------------------------------------------------------------------------------------------
    fig,ax=slide(21); eyebrow(ax,"4 · campaign 6 — results")
    title(ax,"6c — FSQ: criterion-shaping fails at every rate (H3: rejected)")
    iax=chart_axes(fig,[0.09,0.20,0.52,0.52])
    iax.set_xlim(7,33); iax.set_ylim(0,1.02)
    series=[("FSQ + kernel",NEG,"o",[(10,0.039),(15.5,0.136),(18,0.411),(24,0.530)],True),
            ("FSQ recon-only",PURPLE,"o",[(15.5,0.633),(24,0.851)],True),
            ("physics grid (untrained)",SIGNAL,"*",[(23.3,0.955),(26.3,0.966)],False),
            ("trained VQ (RVQ)",AMBER,"s",[(27,0.927),(30,0.945)],False)]
    for name,c,mk,pts,ln in series:
        xs,ys=zip(*pts)
        iax.plot(xs,ys,color=c,lw=2 if ln else 0,marker=mk,
                 ms=13 if mk=="*" else 7,ls="-" if ln else "none",zorder=3,label=name)
    iax.plot([27],[0.832],ls="none",marker="D",ms=6.5,markerfacecolor="none",
             markeredgecolor=AMBER,markeredgewidth=1.6,zorder=3,label="post-hoc RVQ")
    iax.set_xlabel("release rate (bits / particle)",fontsize=10,color=INKS)
    iax.set_ylabel("criterion recall@8",fontsize=10,color=INKS)
    iax.legend(fontsize=8.5,loc="upper left",frameon=False)
    for xy,t,dy in [((24,0.530),"0.53",10),((24,0.851),"0.85",10),((26.3,0.966),"0.966",9),
                    ((30,0.945),"0.945",-16),((15.5,0.136),"0.14",10)]:
        iax.annotate(t,xy,xytext=(0,dy),textcoords="offset points",fontsize=8,
                     family=MONO,color=MUT,ha="center")
    body(ax,"With no codebook, no EMA and no commitment loss, the kernel still degrades the FSQ token relative "
            "to plain reconstruction at 10, 15.5, 18 and 24 bits; both trained variants remain below the "
            "untrained grid.",0.64,size=12.5,color=INKS,width=42,x=0.645)
    body(ax,"Conclusion: the C4/C5 negative was not caused by codebook dynamics. Together, C4–C6 close the "
            "criterion-shaped discrete program: the conflict is between the kernel geometry and quantization "
            "itself.",0.40,size=12.5,color=INKS,width=42,x=0.645)
    setup_line(ax,"AOJ · 20 epochs · kernel→z_q ablation: 0.092; kernel + recon w=1: 0.571 (15.5 b)")
    emit(fig)

    # ---- 22 · 6c mechanism -----------------------------------------------------------------------------------------------
    fig,ax=slide(22); eyebrow(ax,"4 · campaign 6 — results")
    title(ax,"6c — mechanism: kernel-induced code degeneracy")
    iax=chart_axes(fig,[0.10,0.21,0.42,0.46])
    arms=["kernel → z_e","kernel → z_q","kernel + recon","recon-only"]
    occ=[2.2,3.0,25.0,27.8]
    ypos=range(len(arms))[::-1]
    iax.barh(list(ypos),occ,height=0.5,color=BLUE,zorder=3)
    for y,v in zip(ypos,occ):
        iax.annotate(f"{v:.1f}%",(v,y),xytext=(5,0),textcoords="offset points",
                     fontsize=9.5,family=MONO,color=INKS,va="center")
    iax.set_yticks(list(ypos),arms,fontsize=10)
    iax.set_xlim(0,32); iax.grid(axis="y",visible=False); iax.grid(axis="x",color=LINE,lw=0.6,alpha=0.7)
    iax.set_xlabel("grid points occupied (% of 46,656 · 15.5-bit FSQ)",fontsize=10,color=INKS)
    body(ax,"The kernel's fixed point drives criterion partners toward coincidence in the latent; any "
            "quantizer maps coincidence to identical codes. Kernel arms occupy ~2 % of the grid vs ~28 % "
            "for reconstruction.",0.67,size=12.5,color=INKS,width=44,x=0.575)
    body(ax,"Retrieval cannot rank true partners among thousands of identical-code impostors, so recall "
            "collapses. The mechanism is quantizer-independent — one cause for the C4 (RVQ) and C6c (FSQ) "
            "negatives.",0.475,size=12.5,color=INKS,width=44,x=0.575)
    body(ax,"Implication: install the criterion on a continuous latent; a discrete tier should quantize the "
            "features (6b), not the criterion-shaped embedding.",0.28,size=12.5,color=MUT,width=44,x=0.575)
    emit(fig)

    # ---- 23 · decision map -------------------------------------------------------------------------------------------------
    fig,ax=slide(23); eyebrow(ax,"5 · summary")
    title(ax,"All release candidates on one axis (honest rates)")
    iax=chart_axes(fig,[0.10,0.175,0.60,0.565])
    iax.set_xscale("log",base=2); iax.set_xlim(11,300); iax.set_ylim(0,1.05)
    iax.axvline(50,color=FAINT,lw=1.2,ls=(0,(1,2)))
    iax.text(50,0.04," raw particle ~50 b",family=MONO,fontsize=8,color=MUT,va="bottom")
    pts=[(BLUE,"o",96,0.9996,"library d6 ≈ cVAE β=0.03",(0,9),"center"),
         (BLUE,"o",96,0.976,"recon VAE β=0.1",(8,-3),"left"),
         (SIGNAL,"*",23.3,0.955,"grid",(-9,0),"right"),
         (SIGNAL,"*",26.3,0.966,"grid ×2",(2,10),"left"),
         (AMBER,"s",30,0.945,"VQ 3×1024",(9,-2),"left"),
         (AMBER,"s",27,0.927,"VQ 3×512",(-8,-11),"right"),
         (PURPLE,"o",24,0.851,"FSQ recon",(0,-16),"center"),
         (NEG,"o",15.5,0.136,"FSQ + kernel",(8,-3),"left"),
         (FAINT,"o",192,0.983,"null (untrained)",(0,9),"center"),
         (FAINT,"o",48,0.300,"raw kinematics",(8,-3),"left")]
    for c,mk,x,y,lab,off,ha in pts:
        iax.plot([x],[y],ls="none",marker=mk,ms=13 if mk=="*" else 8,color=c,zorder=3)
        iax.annotate(lab,(x,y),xytext=off,textcoords="offset points",fontsize=8,
                     family=MONO,color=INKS,ha=ha,va="center")
    iax.plot([27],[0.832],ls="none",marker="D",ms=7,markerfacecolor="none",
             markeredgecolor=AMBER,markeredgewidth=1.6,zorder=3)
    iax.annotate("post-hoc RVQ",(27,0.832),xytext=(8,-3),textcoords="offset points",
                 fontsize=8,family=MONO,color=INKS,ha="left",va="center")
    for c,lab in [(BLUE,"continuous (fp16 rate)"),(SIGNAL,"physics grid"),
                  (AMBER,"VQ family"),(PURPLE,"FSQ recon"),(NEG,"FSQ + kernel"),
                  (FAINT,"references")]:
        iax.plot([],[],ls="none",marker="o",ms=7,color=c,label=lab)
    iax.legend(fontsize=8.5,loc="lower right",frameon=False)
    iax.set_xlabel("release rate (bits / particle, log)",fontsize=10,color=INKS)
    iax.set_ylabel("criterion recall@8",fontsize=10,color=INKS)
    body(ax,"Continuous tokens at their fp16 release size (slide 18 convention); the 13.3 KL-bits of the "
            "β=0.03 arm is a diagnostic. Full map with every C1–6 arm: "
            "benchmarks/figures/library_rate.png.",0.60,size=12,color=INKS,width=34,x=0.735)
    emit(fig)

    # ---- 24 · revised conclusions ----------------------------------------------------------------------------------------------
    fig,ax=slide(24); eyebrow(ax,"5 · summary")
    title(ax,"Revisions to previously recorded conclusions")
    cols=[(0.06,"left"),(0.19,"left"),(0.60,"left")]
    rows=[
        [("revised",NEG,True),"'reconstruction buys species, not the criterion' (C5)","β=1 rate artifact; 0.976 at β=0.1 (3 seeds)"],
        [("relabelled",NEG,True),"the '~1.4-bit' contrastive VAE (C5)","KL ≠ release rate for deterministic μ; quote fp16"],
        [("closed",NEG,True),"criterion-shaped discrete tokens (C4→C6c)","kernel hurts every quantizer at every rate tested"],
        [("confirmed",POS,True),"criterion/information complementarity (C2b→C6a)","survives rate-matching: 0.9984 recall, R² ≤ 0.82"],
        [("new baseline",POS,True),"untrained physics grid (C6b)","0.966 @ 26.3 b — the bar for any discrete proposal"],
    ]
    table(ax,cols,["status","conclusion","evidence"],rows,0.70,rowh=0.062,fs=11.5)
    body(ax,"All pre-registered outcomes and full tables: benchmarks/README.md §Campaign 6; design and audit: "
            "docs/research/campaign6_alternate_schemes.md.",0.24,size=11.5,color=MUT,width=108)
    emit(fig)

    # ---- 25 · conclusions & outlook -----------------------------------------------------------------------------------------------
    fig,ax=slide(25); eyebrow(ax,"5 · summary")
    title(ax,"Conclusions and outlook")
    ax.text(0.06,0.72,"Conclusions",family=SANS,fontsize=14,color=INK,va="top",fontweight="bold")
    bullets(ax,[
        "The released-token decision reduces to a stated trade: criterion-organized and compact → the "
        "contrastive VAE / dim-6 library (recall 0.998, 96 fp16 b); a discrete tier → the untrained double "
        "grid (0.966 @ 26.3 b, R² 0.949); maximal fidelity → plain VQ at 30 b (R² 0.970).",
        "Installing the physics criterion requires a smooth continuous latent; quantizing a criterion-shaped "
        "embedding fails for a quantizer-independent reason (code degeneracy).",
        "Rate claims for continuous tokens must be operational (fp16 or a codec), not ELBO-derived.",
        ],0.675,size=12.5,width=105)
    ax.text(0.06,0.40,"Outlook",family=SANS,fontsize=14,color=INK,va="top",fontweight="bold")
    bullets(ax,[
        "Campaign 7 (runs today): masked-continuous FM probe over the release menu — no C3–C6 winner has a "
        "consumer-side number yet; this is the mission's own success metric.",
        "The open corner: REPA-style frozen-teacher distillation and the symbols ⊕ latent factorized release; "
        "plus d0/dz feature restoration and the protocol's anomaly-detection probe.",
        "North star (data-blocked): cross-experiment invariance — the context-free library and the grid tier "
        "are both constructed for it; a second-detector sample is the missing input.",
        ],0.355,size=12.5,width=105)
    emit(fig)

    # ---- 26 · references -----------------------------------------------------------------------------------------------------------
    fig,ax=slide(26); eyebrow(ax,"references")
    title(ax,"References")
    refs=[
        ("VQ-VAE","van den Oord, Vinyals, Kavukcuoglu — Neural Discrete Representation Learning, arXiv:1711.00937"),
        ("FSQ","Mentzer et al. — Finite Scalar Quantization: VQ-VAE Made Simple, arXiv:2309.15505"),
        ("BSQ / LFQ","Zhao et al., arXiv:2406.07548 · Yu et al. (MAGVIT-v2), arXiv:2310.05737"),
        ("VQ gradient fixes","rotation trick, arXiv:2410.06424 · SimVQ, arXiv:2411.02038"),
        ("REPA / RAE","Yu et al., arXiv:2410.06940 · arXiv:2510.11690"),
        ("continuous AR tokens","MAR, arXiv:2406.11838 · GIVT, arXiv:2312.02116 · JetFormer, arXiv:2411.19722"),
        ("HEP tokenization","OmniJet-α, arXiv:2403.05618 · MPM, arXiv:2401.13537 · tokenization-free MPM, arXiv:2409.12589"),
        ("SSL objectives","VICReg, arXiv:2105.04906 · JetCLR, arXiv:2108.04253 · HEP-JEPA, arXiv:2502.03933 · J-JEPA, arXiv:2412.05333"),
        ("per-particle representations","EFN/PFN, Komiske, Metodiev, Thaler — arXiv:1810.05165"),
        ("data","CMS Aspen Open Jets, RunG 2016 open data release"),
        ("this work","benchmarks/README.md (campaign record) · docs/research/ (method of record, C6 design & audit)"),
    ]
    yy=0.72
    for k,v in refs:
        ax.text(0.06,yy,k,family=MONO,fontsize=10.5,color=SIGNAL,va="top")
        ax.text(0.30,yy,v,family=SANS,fontsize=11,color=INKS,va="top")
        yy-=0.055
    emit(fig)

    d=pdf.infodict(); d["Title"]="Physics-Defined Tokenization for AI-Ready HEP Datasets"
    d["Author"]="heptokens working group"
    d["Subject"]="Six campaigns: particle-library objective, evaluation methodology, alternate tokenization schemes"

print("wrote", OUT)
