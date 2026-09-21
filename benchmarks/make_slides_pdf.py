"""Render the Particle-Library slide deck to a landscape 16:9 PDF (matplotlib).

Reliable, dependency-free (no browser/network): one figure per slide, the three
campaign plots embedded, the deck's "event-display" identity on a light ground.
"""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import image as mpimg

# ---- palette (light, print-friendly) + type ----
PAPER="#ffffff"; SURF="#f5f8fa"; INK="#141b24"; INKS="#38444f"
MUT="#6b7883"; FAINT="#9aa7b2"; LINE="#d8dfe5"
SIGNAL="#0e9e78"; AMBER="#a9761a"; NEG="#cf3b3f"; POS=SIGNAL
SER="DejaVu Serif"; SANS="DejaVu Sans"; MONO="DejaVu Sans Mono"
from pathlib import Path as _P
_REPO=_P(__file__).resolve().parent.parent
FIG=str(_REPO/"benchmarks"/"figures"); OUT=str(_REPO/"docs"/"particle_library_slides.pdf")
plt.rcParams.update({"pdf.fonttype":42,"font.family":SANS})

def emit(fig):
    pdf.savefig(fig,facecolor=PAPER); plt.close(fig)

def slide(n, total=16, dark_tick=SIGNAL):
    fig=plt.figure(figsize=(13.333,7.5), dpi=200, facecolor=PAPER)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Rectangle((0,0),1,1,color=PAPER,zorder=-5))
    ax.add_patch(Rectangle((0.06,0.895),0.032,0.006,color=dark_tick,zorder=2))  # accent tick
    ax.plot([0.06,0.94],[0.072,0.072],color=LINE,lw=0.8)                        # footer rule
    ax.text(0.06,0.045,"The Particle Library  ·  heptokens",family=MONO,fontsize=8.5,
            color=FAINT,va="center")
    ax.text(0.94,0.045,f"{n:02d} / {total:02d}",family=MONO,fontsize=8.5,color=MUT,
            va="center",ha="right")
    return fig,ax

def eyebrow(ax,t,color=SIGNAL):
    ax.text(0.06,0.855,t.upper(),family=MONO,fontsize=11,color=color,va="center",
            fontweight="bold")
def title(ax,t,y=0.75,size=33):
    fig=ax.figure
    txt=ax.text(0.06,y,t,family=SER,fontsize=size,color=INK,va="top",fontweight="bold")
    fig.canvas.draw()                         # measure the rendered width...
    maxw=(0.94-0.06)*fig.bbox.width
    w=txt.get_window_extent().width
    if w>maxw: txt.set_fontsize(size*maxw/w*0.98)   # ...and shrink to fit the margin
    return txt
def body(ax,t,y,size=16.5,color=INKS,width=74,x=0.06,lh=1.5,ha="left"):
    ax.text(x,y,textwrap.fill(t,width=width),family=SANS,fontsize=size,color=color,
            va="top",ha=ha,linespacing=lh)

def pole(ax,x,w,k,v,lines=None,arrow=None,arrow_color=POS,accent=SIGNAL,y=0.30,h=0.34):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((x,y+0.02),0.006,h-0.04,color=accent,zorder=2))
    cx=x+0.03
    ax.text(cx,y+h-0.05,k.upper(),family=MONO,fontsize=10.5,color=MUT,va="top")
    ax.text(cx,y+h-0.115,v,family=SER,fontsize=19,color=INK,va="top",fontweight="bold")
    if arrow:
        ax.text(x+w/2,y+h*0.40,arrow,family=MONO,fontsize=30,color=arrow_color,
                va="center",ha="center",fontweight="bold")
    if lines:
        yy=y+h-0.185
        for ln,col in lines:
            ax.text(cx,yy,"•  "+ln,family=SANS,fontsize=13.5,color=col,va="top")
            yy-=0.052

def plot_slide(pdf,n,eyeb,imgfile,caption):
    fig,ax=slide(n); eyebrow(ax,eyeb)
    img=mpimg.imread(f"{FIG}/{imgfile}"); ih,iw=img.shape[0],img.shape[1]; ar=iw/ih
    # fit the image into the content box preserving its *physical* aspect
    # (page is 13.333x7.5 in, so 1 x-unit = 13.333in, 1 y-unit = 7.5in)
    bx0,by0,bw,bh=0.10,0.17,0.80,0.60
    if (bw/bh)*(13.333/7.5)>ar: ah=bh; aw=bh*ar*7.5/13.333       # fit height
    else: aw=bw; ah=bw/ar*13.333/7.5                            # fit width
    axx=bx0+(bw-aw)/2; axy=by0+(bh-ah)/2
    ax.add_patch(FancyBboxPatch((axx-0.012,axy-0.016),aw+0.024,ah+0.032,
        boxstyle="round,pad=0,rounding_size=0.02",linewidth=1,edgecolor=LINE,
        facecolor="#ffffff",mutation_aspect=13.333/7.5,zorder=1))
    iax=fig.add_axes([axx,axy,aw,ah]); iax.imshow(img); iax.axis("off"); iax.set_zorder(2)
    ax.text(0.5,by0-0.035,textwrap.fill(caption,width=100),family=SANS,fontsize=11.5,
            color=MUT,va="top",ha="center",linespacing=1.4)
    emit(fig)

with PdfPages(OUT) as pdf:
    # 1 — title
    fig,ax=slide(1)
    ax.text(0.06,0.80,"heptokens · particle-library tokenizer · real CMS data".upper(),
            family=MONO,fontsize=11,color=SIGNAL,va="center",fontweight="bold")
    ax.text(0.06,0.63,"The Particle Library",family=SER,fontsize=62,color=INK,va="center",
            fontweight="bold")
    ax.text(0.06,0.45,"A physics-defined token for AI-ready HEP datasets —",family=SER,
            fontsize=23,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.40,"five campaigns, one criterion.",family=SER,fontsize=23,color=INKS,
            va="center",fontstyle="italic")
    # faint eta-phi dot motif, right side
    import numpy as np; rng=np.random.default_rng(3)
    for gx,gy,c in [(0.85,0.60,SIGNAL),(0.90,0.43,AMBER),(0.83,0.30,SIGNAL),(0.88,0.67,MUT)]:
        for _ in range(9):
            a=rng.random()*6.28; r=rng.random()**0.6*0.035
            ax.add_patch(plt.Circle((gx+np.cos(a)*r,gy+np.sin(a)*r*1.6),0.006,
                color=c,alpha=0.5,zorder=0))
    emit(fig)

    # 2 — problem
    fig,ax=slide(2); eyebrow(ax,"the problem"); title(ax,"Ship tokens, not particle-flow candidates")
    body(ax,"CMS and ATLAS reconstruct particles that outside groups can't access — yet those groups "
            "want to train foundation models on the events. Publish a per-constituent token as the "
            "shareable artifact and the consumer never needs the reconstruction.",0.60,width=78)
    body(ax,"The token is the release interface — versioned by dimension, preprocessing hash, and "
            "checkpoint.",0.36,size=15,color=MUT,width=70)
    emit(fig)

    # 3 — the idea / criterion
    fig,ax=slide(3); eyebrow(ax,"the idea"); title(ax,"Define the metric in physics, not augmentations")
    ax.add_patch(FancyBboxPatch((0.06,0.34),0.62,0.30,boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.36),0.006,0.26,color=SIGNAL,zorder=2))
    body(ax,"Same PID, same charge, similar pT (ΔpT ≤ 10%) and same detector region "
            "(|Δη|, |Δφ| ≤ 0.1) → positives. Everything else is pushed away.",0.585,size=19,
            color=INK,width=48,x=0.10)
    body(ax,"Explicit, auditable, versionable — and computable on any experiment's particles, so "
            "cross-detector alignment can fall out of mixed batches.",0.27,size=14.5,color=MUT,width=82)
    emit(fig)

    # 4 — achieved big number
    fig,ax=slide(4); eyebrow(ax,"campaign 1 · achieved")
    title(ax,"The criterion is achieved on real CMS data")
    ax.text(0.06,0.44,"0.9992",family=MONO,fontsize=118,color=POS,va="center",fontweight="bold")
    body(ax,"criterion recall@8 — of an anchor's available same-PID/charge/pT/region partners, the "
            "fraction ranked in its top-8. A 20× reduction in retrieval misses vs the untrained "
            "hand-crafted null.",0.24,size=15,color=MUT,width=88)
    emit(fig)

    # 5 — plot islands
    plot_slide(pdf,5,"campaign 1 · species map","library_pid_islands.png",
        "t-SNE of library tokens, real CMS. π, γ, K_L, e, μ separate cleanly with kinematic "
        "substructure inside each island — the species structure is metric, not linear.")

    # 6 — context-free
    fig,ax=slide(6); eyebrow(ax,"campaign 1 · the pivotal call"); title(ax,"Context-free was decisive")
    ax.text(0.06,0.50,"context transformer 0.052",family=MONO,fontsize=25,color=MUT,va="center")
    ax.text(0.06,0.41,"→   MLP 0.9992",family=MONO,fontsize=40,color=POS,va="center",fontweight="bold")
    body(ax,"A jet-context encoder makes the same particle embed differently in different events — "
            "which destroys the library property that cross-experiment alignment needs. The token "
            "must be a function of one particle.",0.27,size=15,color=MUT,width=88)
    emit(fig)

    # 7 — complementary (two poles)
    fig,ax=slide(7); eyebrow(ax,"campaign 2 · library vs VQ-VAE")
    title(ax,"Complementary, not competing",size=31)
    pole(ax,0.06,0.42,"Physics library","owns the criterion",
         lines=[("recall@8  0.9991",INKS),("~80× fewer retrieval misses",INKS),("species structure is metric",MUT)],
         accent=SIGNAL)
    pole(ax,0.52,0.42,"Reconstruction VQ","owns the information",
         lines=[("within-PID pT R²  0.97",INKS),("PID / charge linearly decodable",INKS),("27 bits, faithful",MUT)],
         accent=AMBER)
    body(ax,"Geometry and information are two axes — and there's no downstream cost: library tokens "
            "probe at the raw-feature ceiling (0.744 acc / 0.931 AUC).",0.20,size=14,color=MUT,width=92)
    emit(fig)

    # 8 — plot pareto
    plot_slide(pdf,8,"campaign 3 · the frontier","library_pareto.png",
        "Organization vs information. The hybrid weight traces the frontier toward the shaded "
        "'both at once' corner but never reaches it — so recon_weight is a release dial, not a solved optimum.")

    # 9 — halves for free
    fig,ax=slide(9); eyebrow(ax,"campaign 3 · a free win"); title(ax,"The token halves for free")
    ax.text(0.06,0.44,"16-D  →  8-D",family=MONO,fontsize=58,color=POS,va="center",fontweight="bold")
    body(ax,"Compressing the token raised criterion recall (0.9968 → 0.9982) and PID purity "
            "(0.918 → 0.987). The 16-D token was larger than its 5-column input; the criterion never "
            "needed the extra dimensions.",0.26,size=15,color=MUT,width=88)
    emit(fig)

    # 10 — C4 negative (two poles)
    fig,ax=slide(10); eyebrow(ax,"campaign 4 · robust negative",color=NEG)
    title(ax,"Don't quantize the criterion — shrink the token",size=29)
    pole(ax,0.06,0.42,"Criterion-shaped VQ","fails",
         lines=[("recall 0.772  vs plain VQ 0.927",NEG),("kernel spreads to 15+ dims",MUT),("quantization can't hold it",MUT)],
         accent=NEG)
    pole(ax,0.52,0.42,"Low-dim continuous","works",
         lines=[("dim-6 = 96 bits, recall 0.9996",POS),("pT R² rises 0.18 → 0.74",INKS),("granularity beats bit-count",MUT)],
         accent=SIGNAL)
    emit(fig)

    # 11 — plot rate
    plot_slide(pdf,11,"campaign 4 · compression","library_rate.png",
        "Recall@8 vs release rate (bits/particle, log scale). Every token on one axis; the dotted "
        "line is the ~50-bit raw particle. Triangles mark the context-aware attention encoder.")

    # 12 — 2x2x2 setup
    fig,ax=slide(12); eyebrow(ax,"campaign 5 · a clean 2×2×2"); title(ax,"Put every choice on one axis")
    body(ax,"Bottleneck {VAE continuous · VQ discrete} × objective {reconstruction · + physics term} "
            "× encoder {context-free MLP · context-aware attention} — everything else matched, all on "
            "one bits axis (the VAE's rate read from its KL).",0.58,size=17,width=76)
    body(ax,"Which of the three choices actually decides the token?",0.34,size=15,color=MUT,width=70)
    emit(fig)

    # 13 — placement decisive
    fig,ax=slide(13); eyebrow(ax,"campaign 5 · the answer"); title(ax,"Placement is decisive")
    pole(ax,0.06,0.42,"Physics term on VAE latent","",arrow="0.383 → 0.916",arrow_color=POS,accent=SIGNAL)
    ax.text(0.09,0.335,"smooth mean → clean gradient → it works",family=SANS,fontsize=12.5,color=MUT,va="center")
    pole(ax,0.52,0.42,"Same term on VQ codebook","",arrow="0.945 → 0.827",arrow_color=NEG,accent=NEG)
    ax.text(0.55,0.335,"straight-through + EMA → it hurts",family=SANS,fontsize=12.5,color=MUT,va="center")
    body(ax,"Same term, opposite sign. And the contrastive VAE reaches recall 0.916 at ~1.4 bits — "
            "sub-raw and criterion-strong.",0.20,size=14,color=MUT,width=92)
    emit(fig)

    # 14 — context asymmetric
    fig,ax=slide(14); eyebrow(ax,"campaign 5 · context"); title(ax,"Attention is an asymmetric trade")
    pole(ax,0.06,0.42,"Low-rate VAE + attention","",arrow="0.383 → 0.051",arrow_color=NEG,accent=NEG)
    ax.text(0.09,0.335,"context crowds out the criterion",family=SANS,fontsize=12.5,color=MUT,va="center")
    pole(ax,0.52,0.42,"High-rate VQ + attention","",arrow="0.945 → 0.967",arrow_color=POS,accent=SIGNAL)
    ax.text(0.55,0.335,"it can afford both",family=SANS,fontsize=12.5,color=MUT,va="center")
    body(ax,"Context buys jet information — but folds the jet environment into the per-particle code. "
            "Only a high-rate token affords both context and criterion.",0.20,size=14,color=MUT,width=92)
    emit(fig)

    # 15 — synthesis takeaways
    fig,ax=slide(15); eyebrow(ax,"synthesis"); title(ax,"What five campaigns establish")
    items=[("01","State the metric explicitly","physics criterion beats an implicit augmentation metric."),
           ("02","Keep the encoder context-free","the same particle, the same token, everywhere."),
           ("03","Geometry ≠ information","a controllable frontier, not one optimum."),
           ("04","Compress the continuous token","don't force the criterion through a codebook."),
           ("05","Placement and rate decide","not the architecture.")]
    yy=0.64
    for n_,h_,s_ in items:
        ax.text(0.07,yy,n_,family=MONO,fontsize=16,color=SIGNAL,va="top",fontweight="bold")
        ax.text(0.115,yy,h_,family=SANS,fontsize=16.5,color=INK,va="top",fontweight="bold")
        ax.text(0.115,yy-0.040,s_,family=SANS,fontsize=13.5,color=MUT,va="top")
        yy-=0.108
    emit(fig)

    # 16 — frontier
    fig,ax=slide(16); eyebrow(ax,"frontier"); title(ax,"Where this points next")
    ax.add_patch(FancyBboxPatch((0.06,0.46),0.62,0.18,boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.48),0.006,0.14,color=NEG,zorder=2))
    ax.text(0.09,0.605,"DATA-BLOCKED",family=MONO,fontsize=10.5,color=NEG,va="top")
    ax.text(0.09,0.565,"Cross-experiment invariance",family=SER,fontsize=17,color=INK,va="top",fontweight="bold")
    ax.text(0.09,0.515,"CMS + ATLAS in mixed batches — the north star; needs a second-detector sample.",
            family=SANS,fontsize=13,color=MUT,va="top")
    ax.add_patch(FancyBboxPatch((0.06,0.24),0.62,0.18,boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.26),0.006,0.14,color=SIGNAL,zorder=2))
    ax.text(0.09,0.385,"RUNS TODAY",family=MONO,fontsize=10.5,color=SIGNAL,va="top")
    ax.text(0.09,0.345,"Downstream FM probe  ·  rate-matched VAE ↔ VQ",family=SER,fontsize=17,color=INK,va="top",fontweight="bold")
    ax.text(0.09,0.295,"Train a backbone on the exported tokens; β-sweep the VAE to a 30-bit point.",
            family=SANS,fontsize=13,color=MUT,va="top")
    ax.text(0.06,0.16,"The token to release is a stated choice along sub-raw · faithful · criterion —",
            family=SER,fontsize=14.5,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.125,"which is exactly why the criterion is written down.",family=SER,fontsize=14.5,
            color=INKS,va="center",fontstyle="italic")
    emit(fig)

    d=pdf.infodict(); d["Title"]="The Particle Library — Slides"
    d["Author"]="heptokens"; d["Subject"]="Physics-defined tokenizer — five campaigns on real CMS data"

print("wrote", OUT)
