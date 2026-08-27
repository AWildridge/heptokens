"""Render the tokenization-program slide deck to a landscape 16:9 PDF.

Academic seminar format, findings-first: motivation -> data & protocol ->
evaluation methodology (what we measure and why) -> six combined findings with
evidence pooled across all campaigns -> decision map -> ranked verdicts (what
works / what to test next / closed directions) -> conclusions -> references.

Numbers: benchmarks/README.md + benchmarks/figures/library_metrics.json
(committed eval output). Chart palette validated (dataviz six checks, white
surface): blue #2a78d6 = continuous/VAE · green #0e9e78 = physics grid · amber
#a9761a = VQ family · purple #8250df = FSQ recon · red #cf3b3f = criterion-
shaped quantizers.
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

TOTAL=19

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

def table(ax,cols,headers,rows,y0,rowh=0.047,fs=11,hfs=9):
    """cols: [(x, ha)]; rows: list of list of str|(str,color,bold)."""
    x1=max(x for x,_ in cols)+0.004
    for (x,ha),h in zip(cols,headers):
        ax.text(x,y0,h.upper(),family=MONO,fontsize=hfs,color=MUT,va="top",ha=ha)
    ax.plot([cols[0][0],x1],[y0-0.026,y0-0.026],color=LINE,lw=0.8)
    yy=y0-0.042
    for row in rows:
        for (x,ha),cell in zip(cols,row):
            if isinstance(cell,tuple) and len(cell)==3: t,c,b=cell
            else: t,c,b=cell,INKS,False
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
    ax.text(0.06,0.515,"Combined findings from six experimental campaigns, and a ranked",
            family=SER,fontsize=17,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.472,"assessment of tokenization schemes for the released token.",
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
           ("2","Methodology — data, fairness protocol, and what we measure (and why)"),
           ("3","Combined findings — six results, evidence pooled across all campaigns"),
           ("4","Rankings — what works, what to test next, and what is closed"),
           ("5","Conclusions")]
    yy=0.68
    for n_,t_ in items:
        ax.text(0.08,yy,n_,family=MONO,fontsize=16,color=SIGNAL,va="top",fontweight="bold")
        ax.text(0.125,yy,t_,family=SANS,fontsize=15.5,color=INK,va="top")
        yy-=0.093
    emit(fig)

    # ---- 3 · motivation --------------------------------------------------------
    fig,ax=slide(3); eyebrow(ax,"1 · introduction")
    title(ax,"Motivation: the token dataset is the product")
    bullets(ax,[
        "Experiments reconstruct particle-flow candidates that external groups cannot access; those groups "
        "want to train foundation models on the events. We publish a per-constituent token as the shareable "
        "artifact — versioned by dimension, preprocessing hash and checkpoint.",
        "Release requirements: information sufficiency for unknown future tasks; detector abstraction; "
        "an auditable statement of what the token forgets; compression at the 178 M-jet scale.",
        "Starting point: a residual VQ-VAE emitting integer indices — with codebook collapse, an implicit "
        "metric set by the reconstruction objective, and lossiness by construction. Six campaigns "
        "(2026-07 → 2026-08) tested the alternatives; this talk presents the combined outcome.",
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
    fig,ax=slide(4); eyebrow(ax,"2 · methodology")
    title(ax,"Data, the library objective, and the fairness protocol")
    ax.add_patch(FancyBboxPatch((0.06,0.63),0.86,0.10,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.text(0.085,0.705,"THE PHYSICS CRITERION (defines 'same-physics particle')",family=MONO,
            fontsize=9,color=SIGNAL,va="top")
    ax.text(0.085,0.672,"positives:  same PID  ∧  same charge  ∧  ΔpT/pT ≤ 0.10  ∧  |Δη| ≤ 0.1  ∧  |Δφ| ≤ 0.1"
            "     (graded kernel, hard PID/charge gates)",family=MONO,fontsize=11,color=INK,va="top")
    bullets(ax,[
        "Data: Aspen Open Jets RunG — 1.96 M real CMS jets, constituents (pT, η, φ, PID, charge); all "
        "criterion results. jets5c (620 k labelled jets) for downstream consumer probes.",
        "Matched budgets everywhere: identical encoders, inputs, splits, schedules; only the "
        "objective/bottleneck under test differs. Context-free per-particle MLP unless stated.",
        "Consumer view: downstream probes see only the exported token files, never the constituents.",
        "Untrained nulls are mandatory (PID-one-hot ⊕ kinematics; criterion-aligned feature grid): "
        "learning must beat them to claim rent. 3 seeds wherever a number becomes a claim.",
        ],0.575,size=12.5,width=105)
    setup_line(ax,"single A100 MIG slice · torch 2.5.1 · 15-20 epochs per arm · full configs in benchmarks/")
    emit(fig)

    # ---- 5 · metrics -----------------------------------------------------------------
    fig,ax=slide(5); eyebrow(ax,"2 · methodology")
    title(ax,"What we measure, and why")
    ax.add_patch(FancyBboxPatch((0.06,0.615),0.86,0.115,boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.text(0.085,0.70,"PRIMARY: availability-aware criterion recall@8   =   mean over anchors of  |top-8 ∩ partners| / min(8, #partners)",
            family=MONO,fontsize=10.5,color=INK,va="top")
    ax.text(0.085,0.662,"criterion density is ~3·10⁻⁵, so precision@k is bounded by partner availability and flattens all methods —",
            family=SANS,fontsize=10.5,color=INKS,va="top")
    ax.text(0.085,0.634,"recall against the availability ceiling is 1.0 for a perfect embedding at any density. Native to the library objective; always flagged.",
            family=SANS,fontsize=10.5,color=INKS,va="top")
    cols=[(0.06,"left"),(0.32,"left"),(0.60,"left")]
    rows=[
        ["kNN PID purity","species organization (neutral)","misleads alone: 0.996 purity with 0.383 recall"],
        ["within-PID pT R²","kinematic information retention","catches PID-shortcut tokens; releases need observables"],
        ["linear PID/charge probes","consumer linear decodability","kernel structure is metric, not linear (probe ≈ 0.37)"],
        ["effective rank","dimensional collapse","VICReg/VQ collapse modes (quantizer-0 <10 % codes)"],
        ["bits / particle","release rate","n_q·log₂K · fp16 dim·16 · KL-rate is diagnostic only (F6)"],
        ["downstream probe (jets5c)","consumer value of the export","tokens are the product; probes never see constituents"],
    ]
    table(ax,cols,["supporting metric","what it measures","failure it guards against"],rows,0.565,rowh=0.052,fs=10.5)
    body(ax,"No metric is trusted alone: every result table reports the suite, with the objective-native column flagged.",
         0.185,size=11.5,color=MUT,width=110)
    emit(fig)

    # ---- 6 · F1 -------------------------------------------------------------------------
    fig,ax=slide(6); eyebrow(ax,"3 · combined findings")
    title(ax,"F1 — state the metric in physics; keep the encoder context-free")
    cols=[(0.06,"left"),(0.52,"right"),(0.66,"right")]
    rows=[
        [("library: explicit graded criterion, context-free MLP",INK,True),("0.9992",POS,True),"0.910"],
        ["explicit criterion, hard windows","0.9935","0.664"],
        ["implicit metric: smearing-VICReg + jet-context transformer",("0.052",NEG,True),"0.284"],
    ]
    table(ax,cols,["token objective (AOJ)","recall@8","kNN purity"],rows,0.70,rowh=0.048)
    bullets(ax,[
        "The augmentation list of a generic SSL recipe silently defines the metric: smearing pT traded away "
        "pT sensitivity and cost ≈1.5 AUC on labelled downstream probes.",
        "Graded kernels beat hard windows on every axis — hard windows kink the geometry at the window edge.",
        "Jet context breaks the library property (the same particle must map to the same token in any event) "
        "and destroys low-rate tokens (recall 0.383 → 0.051); its only gain is at high rate (+0.022, 1 seed).",
        ],0.48,size=12.5,width=104)
    setup_line(ax,"evidence: campaigns 1, 2, 5 · benchmarks/README.md")
    emit(fig)

    # ---- 7 · F2 --------------------------------------------------------------------------
    fig,ax=slide(7); eyebrow(ax,"3 · combined findings")
    title(ax,"F2 — geometry and information are separable currencies; the corner is unreached")
    cols=[(0.06,"left"),(0.40,"right"),(0.56,"right"),(0.68,"right"),(0.80,"right"),(0.90,"right")]
    rows=[
        [("library dim 6 (kernel)",INK,True),"96",("0.9996",POS,True),"0.743","0.49","0.980"],
        [("contrastive VAE β=.03",INK,True),"96 ᵃ","0.9984±.0015","0.758","0.59","0.989"],
        ["hybrid kernel+recon w=3","256","0.9955",("0.908",INK,True),"—","0.935"],
        ["recon-only VAE β=0.1","96 ᵃ","0.976±.004","0.920±.011",("1.00",INK,True),"0.995"],
        [("plain VQ 3×1024",INK,True),"30","0.945",("0.970",INK,True),("1.00",INK,True),"0.996"],
        ["null: PID-1hot ⊕ kin (untrained)","192","0.9833","0.949","1.00","0.991"],
    ]
    table(ax,cols,["interface","bits","recall@8","pT R²","pid probe","purity"],rows,0.695,rowh=0.0435,fs=10.5)
    bullets(ax,[
        "No trained token reaches the corner (recall ≥ 0.999 ∧ R² ≥ 0.9): the best single artifact, hybrid "
        "w=3, misses recall by 0.004; the kernel arms own geometry, the reconstructive arms own information.",
        "Reconstruction never installs the criterion beyond the untrained null; the physics kernel is the "
        "only objective that has beaten it (0.9984 vs 0.9833).",
        ],0.375,size=12.5,width=104)
    setup_line(ax,"evidence: campaigns 2b, 3, 5, 6a · ᵃ fp16 release size; measured KL-rate 11-13 bits (see F6)")
    emit(fig)

    # ---- 8 · F3 chart ------------------------------------------------------------------------
    fig,ax=slide(8); eyebrow(ax,"3 · combined findings")
    title(ax,"F3 — quantizing a criterion-shaped embedding fails: any quantizer, any rate")
    iax=chart_axes(fig,[0.09,0.20,0.52,0.52])
    iax.set_xlim(7,33); iax.set_ylim(0,1.02)
    series=[("kernel → FSQ grid",NEG,"o",[(10,0.039),(15.5,0.136),(18,0.411),(24,0.530)],True,False),
            ("kernel → RVQ codebook",NEG,"s",[(27,0.772),(30,0.827)],False,True),
            ("FSQ recon-only",PURPLE,"o",[(15.5,0.633),(24,0.851)],True,False),
            ("plain VQ (recon)",AMBER,"s",[(27,0.927),(30,0.945)],False,False),
            ("physics grid (untrained)",SIGNAL,"*",[(23.3,0.955),(26.3,0.966)],False,False)]
    for name,c,mk,pts,ln,open_ in series:
        xs,ys=zip(*pts)
        kw=dict(markerfacecolor="none",markeredgecolor=c,markeredgewidth=1.8) if open_ else dict(color=c)
        iax.plot(xs,ys,lw=2 if ln else 0,marker=mk,ms=13 if mk=="*" else 7.5,
                 ls="-" if ln else "none",zorder=3,label=name,
                 **(kw if not ln else dict(color=c)))
    iax.set_xlabel("release rate (bits / particle)",fontsize=10,color=INKS)
    iax.set_ylabel("criterion recall@8",fontsize=10,color=INKS)
    iax.legend(fontsize=8.5,loc="lower right",frameon=False)
    for xy,t,dy,dx in [((24,0.530),"0.53",10,0),((24,0.851),"0.85",10,0),((26.3,0.966),"0.966",9,0),
                       ((30,0.945),"0.945",9,0),((30,0.827),"0.827",-16,0),((15.5,0.136),"0.14",10,0)]:
        iax.annotate(t,xy,xytext=(dx,dy),textcoords="offset points",fontsize=8,
                     family=MONO,color=MUT,ha="center")
    body(ax,"The physics kernel degrades every quantized bottleneck it touches: learned RVQ codebooks "
            "(0.945 → 0.827 at 30 b; both z_q and z_e placements) and codebook-free FSQ grids at 10, "
            "15.5, 18 and 24 bits.",0.66,size=12.5,color=INKS,width=42,x=0.645)
    body(ax,"Removing the suspected culprits — codebook, EMA, commitment loss — made it worse, not "
            "better. The same kernel on a smooth continuous latent lifts recall 0.383 → 0.916.",
            0.46,size=12.5,color=INKS,width=42,x=0.645)
    setup_line(ax,"evidence: campaigns 4, 5, 6c · kernel→z_q ablation 0.092 · kernel+recon w=1: 0.571")
    emit(fig)

    # ---- 9 · F3 mechanism -----------------------------------------------------------------------
    fig,ax=slide(9); eyebrow(ax,"3 · combined findings")
    title(ax,"F3 (mechanism) — kernel-induced code degeneracy")
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
            "collapses. The mechanism is quantizer-independent — one cause explains the RVQ and FSQ "
            "negatives.",0.475,size=12.5,color=INKS,width=44,x=0.575)
    body(ax,"Implication: install the criterion on a continuous latent; a discrete tier should quantize the "
            "features (F5), not the criterion-shaped embedding.",0.28,size=12.5,color=MUT,width=44,x=0.575)
    emit(fig)

    # ---- 10 · F4 beta dial -------------------------------------------------------------------------
    fig,ax=slide(10); eyebrow(ax,"3 · combined findings")
    title(ax,"F4 — the continuous token compresses; β trades information, never the criterion")
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
    body(ax,"Recall holds ≥ 0.95 across the whole dial while R² peaks at ≈ 0.82 (β = 0.3) — the information "
            "side never reaches VQ. Dimensionality: 16 → 8 is free, dim 6 is the sweet spot, dim 4 the floor; "
            "the encoder itself halves without loss (width 64 ≈ 256).",0.115,size=11.5,color=INKS,width=145)
    emit(fig)

    # ---- 11 · F5 grid -------------------------------------------------------------------------------
    fig,ax=slide(11); eyebrow(ax,"3 · combined findings")
    title(ax,"F5 — untrained references are strong; learned discrete tokens barely pay rent")
    cols=[(0.06,"left"),(0.44,"right"),(0.57,"right"),(0.70,"right"),(0.84,"right")]
    rows=[
        [("physics grid, double (untrained)",INK,True),("26.3",INK,True),("0.9655",POS,True),"0.949","1.00"],
        ["physics grid, single (untrained)","23.3","0.9553","0.948","1.00"],
        ["plain VQ 3×1024 (trained)","30","0.945",("0.970",INK,True),"1.00"],
        ["plain VQ 3×512 (trained)","27","0.927±.006","0.971","1.00"],
        ["FSQ recon-only (trained)","24","0.851","0.840","0.81"],
        ["library → post-hoc RVQ","27","0.832","0.174","0.38"],
    ]
    table(ax,cols,["discrete interface","bits","recall@8","pT R²","pid probe"],rows,0.70,rowh=0.046,fs=10.5)
    bullets(ax,[
        "The zero-parameter criterion-aligned grid (ln(1.1) log-pT × 0.1 η/φ bins ⊕ exact PID/charge, "
        "half-bin-offset double grid) exceeds every learned discrete token on recall at equal or fewer bits.",
        "The trained VQ's entire surviving advantage is +0.02 pT R² and +0.017 purity — and the grid is "
        "computable on any detector, making it a natural cross-experiment release tier.",
        ],0.40,size=12.5,width=104)
    setup_line(ax,"evidence: campaign 6b vs 2b/4/5/6c discrete arms · the 192-b null plays the same role for continuous tokens (F2)")
    emit(fig)

    # ---- 12 · F6 systematics ---------------------------------------------------------------------------
    fig,ax=slide(12); eyebrow(ax,"3 · combined findings",color=NEG)
    title(ax,"F6 — measurement caveats that changed recorded conclusions")
    yy=0.71
    for h_,s_ in [
        ("KL-rate is not a release rate for deterministic-μ tokens",
         "At β = 3, measured KL ≈ 0.001 bits yet recall = 0.9995: μ is deterministic and cosine retrieval is "
         "scale-invariant, so structure survives at 'zero' ELBO rate. Continuous rates are now quoted as fp16 "
         "dim×16; the earlier '~1.4-bit' result is relabelled."),
        ("Objective conclusions must be rate-controlled",
         "'Reconstruction buys species, not the criterion' (recall 0.383) was a β = 1 artifact: at β = 0.1 the "
         "recon-only VAE reaches 0.976 ± 0.004 with R² 0.920 — still below the null, but the original causal "
         "story was wrong."),
        ("Seed discipline",
         "Measured seed std is ±0.006 recall; single-seed deltas of similar size (e.g. context helping the "
         "30-bit VQ by +0.022) are flagged, and headline claims are promoted only at 3 seeds."),
        ]:
        ax.text(0.06,yy,h_,family=SANS,fontsize=14.5,color=INK,va="top",fontweight="bold")
        wrapped=textwrap.fill(s_,width=112)
        ax.text(0.06,yy-0.040,wrapped,family=SANS,fontsize=12,color=INKS,va="top",linespacing=1.35)
        yy-=0.055+0.032*(wrapped.count("\n")+1)+0.035
    setup_line(ax,"evidence: campaign 6a systematics · benchmarks/README.md §Campaign 6, findings 2-3")
    emit(fig)

    # ---- 13 · decision map -------------------------------------------------------------------------------
    fig,ax=slide(13); eyebrow(ax,"3 · combined findings")
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
    body(ax,"Continuous tokens at their fp16 release size (F6 convention); the 13.3 KL-bits of the β=0.03 "
            "arm is a diagnostic. Full map with every campaign-1–6 arm: "
            "benchmarks/figures/library_rate.png.",0.60,size=12,color=INKS,width=34,x=0.735)
    emit(fig)

    # ---- 14 · ranking: what works -----------------------------------------------------------------------
    fig,ax=slide(14); eyebrow(ax,"4 · rankings")
    title(ax,"What works — ranked release candidates")
    cols=[(0.06,"left"),(0.10,"left"),(0.40,"left"),(0.66,"left")]
    rows=[
        [("1",POS,True),("contrastive VAE β=.03 / dim-6 library",INK,True),
         "the criterion carrier — compact","recall 0.998–0.9996 @ 96 fp16 b; R² 0.76"],
        [("2",POS,True),("hybrid kernel + recon, w=3",INK,True),
         "best single-token compromise","recall 0.9955, R² 0.908 @ 256 b"],
        [("3",POS,True),("physics grid, double (untrained)",INK,True),
         "the discrete tier — any detector","recall 0.966, R² 0.949 @ 26.3 b"],
        [("4",AMBER,True),("plain VQ 3×1024",INK,True),
         "max-fidelity discrete, if needed","recall 0.945, R² 0.970 @ 30 b"],
    ]
    table(ax,cols,["#","artifact","role","key numbers"],rows,0.71,rowh=0.062,fs=11.5)
    bullets(ax,[
        "Caveats per rank: (1) species structure is metric, not linear — nonlinear consumers only; "
        "(2) misses the corner by 0.004 recall and is the largest artifact; (3) within-bin detail is lost "
        "and its R² is by construction; (4) buys only +0.02 R² over the grid at more bits, with the "
        "quantizer-0 collapse still unexplained.",
        "Recommended release: (1) as the primary continuous token, (3) as the discrete companion tier — "
        "the pair covers both consumer types at 96 + 26 bits.",
        ],0.40,size=12.5,width=105)
    emit(fig)

    # ---- 15 · ranking: test next --------------------------------------------------------------------------
    fig,ax=slide(15); eyebrow(ax,"4 · rankings")
    title(ax,"What to test next — ranked by decision value")
    cols=[(0.06,"left"),(0.10,"left"),(0.43,"left"),(0.86,"left")]
    rows=[
        [("1",POS,True),("masked-continuous FM probe",INK,True),
         "which export do FM consumers prefer? the mission's own metric","runs today"],
        [("2",POS,True),("REPA frozen-teacher distillation",INK,True),
         "is the unreached corner an optimization artifact? (frozen target)","small"],
        [("3",POS,True),("factorized release: symbols ⊕ latent",INK,True),
         "exact PID/charge channel: linear decodability by construction","small"],
        [("4",INKS,True),("entropy codec for μ tokens",INK,True),
         "turns fp16-vs-KL into an operational sub-raw rate claim (F6)","moderate"],
        [("5",INKS,True),("d0/dz + PUPPI restoration; anomaly probe",INK,True),
         "closes the audit gaps: release completeness, mandated probe","small"],
        [("6",INKS,True),("VQ hygiene (k-means init, resets, SimVQ)",INK,True),
         "was plain VQ underestimated by q0 collapse? (plain arm only)","small"],
    ]
    table(ax,cols,["#","experiment","what it decides","effort"],rows,0.71,rowh=0.068,fs=11)
    bullets(ax,[
        "Deprioritized by F3's quantizer-independent mechanism: Gumbel/dVAE bottlenecks and flow tokens "
        "(raw-parity rate). Data-blocked: cross-experiment invariance — the north star; needs a "
        "second-detector sample (a synthetic-response prototype can only falsify).",
        ],0.245,size=12,width=105)
    emit(fig)

    # ---- 16 · closed directions ------------------------------------------------------------------------------
    fig,ax=slide(16); eyebrow(ax,"4 · rankings",color=NEG)
    title(ax,"Closed directions — do not revisit without new mechanism")
    cols=[(0.06,"left"),(0.47,"left")]
    rows=[
        [("criterion-shaped discrete tokens",NEG,True),"any quantizer, any rate, both placements — code degeneracy (F3)"],
        [("jet-context encoders for the released token",NEG,True),"breaks the library property; destroys low-rate tokens (F1)"],
        [("hard criterion windows",NEG,True),"geometry kinks at window edges; purity 0.66 vs 0.91 (F1)"],
        [("single large codebooks (8192×1)",NEG,True),"collapses to 1.7 % utilization (batch-size confound noted)"],
        [("post-hoc quantization of criterion tokens",NEG,True),"0.832 vs 0.927 trained at 27 b — trained defends geometry"],
        [("KL-bits as a release-rate claim",NEG,True),"decouples from deterministic-μ information content (F6)"],
    ]
    table(ax,cols,["direction","reason"],rows,0.70,rowh=0.062,fs=11.5)
    body(ax,"Each closure is evidence-bounded, not dogma: the condition that would reopen it is a genuinely "
            "different mechanism (as FSQ was for the codebook explanation — tested, and the closure broadened).",
            0.255,size=12,color=MUT,width=108)
    emit(fig)

    # ---- 17 · conclusions ---------------------------------------------------------------------------------------
    fig,ax=slide(17); eyebrow(ax,"5 · conclusions")
    title(ax,"Conclusions")
    bullets(ax,[
        "The release decision reduces to a stated, quantified trade — criterion geometry vs kinematic "
        "information vs bits — and we recommend shipping two tiers: the dim-6 contrastive/library token "
        "(96 fp16 b, recall ≈ 0.999) and the untrained double physics grid (26.3 b, R² 0.949, computable on "
        "any detector).",
        "Installing the physics criterion requires a smooth continuous latent. Quantizing a criterion-shaped "
        "embedding fails for a quantizer-independent reason (code degeneracy) — this direction is closed.",
        "Untrained references did the heaviest epistemic lifting: the 192-b null and the 26-b grid bound what "
        "learning is actually buying, and every future scheme must beat them first.",
        "Rate claims must be operational (fp16 today, an entropy codec next); ELBO rates are diagnostics.",
        "The single most decision-relevant missing number is consumer-side: the masked-continuous FM probe "
        "over the release menu is queued as the next experiment.",
        ],0.72,size=13,width=106)
    emit(fig)

    # ---- 18 · backup pointer / outlook ------------------------------------------------------------------------------
    fig,ax=slide(18); eyebrow(ax,"5 · conclusions")
    title(ax,"Outlook and material")
    bullets(ax,[
        "North star (data-blocked): cross-experiment invariance — same-physics particles from CMS and a "
        "second detector sharing one library. The context-free encoder and the grid tier are both "
        "constructed for it; a second-detector sample is the missing input.",
        "All numbers in this talk are reproducible from the repository: benchmarks/README.md (campaign "
        "record, per-campaign tables), benchmarks/run_campaign*.sh (idempotent sweeps), "
        "eval_library.py (metric suite), docs/research/ (method of record; campaign-6 design, "
        "pre-registration and adversarial audit).",
        "Per-campaign detail (setups, full tables, seed studies) is available as backup material on request.",
        ],0.70,size=13,width=106)
    emit(fig)

    # ---- 19 · references -----------------------------------------------------------------------------------------------
    fig,ax=slide(19); eyebrow(ax,"references")
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
    d["Subject"]="Combined findings across six campaigns and ranked tokenization-scheme assessment"

print("wrote", OUT)
