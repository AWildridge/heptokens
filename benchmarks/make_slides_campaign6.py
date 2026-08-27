"""Render the Campaign-6 slide deck to a landscape 16:9 PDF (matplotlib).

Audience: the tokenization working group + HEP/ML experts. Content: the
alternate-scheme evaluation, Campaign 6 outcomes (rate-matched VAE frontier,
physics grid, FSQ), the claims ledger, and the release menu. Same visual
identity as make_slides_pdf.py; numbers from benchmarks/figures/
library_metrics.json (committed eval output).

Categorical palette validated (dataviz six checks, light/white surface):
blue #2a78d6 = VAE/continuous · green #0e9e78 = physics grid · amber #a9761a =
VQ/recon-discrete · purple #8250df = FSQ recon-only · red #cf3b3f = FSQ+kernel.
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
FIG=str(_REPO/"benchmarks"/"figures"); OUT=str(_REPO/"docs"/"campaign6_slides.pdf")
plt.rcParams.update({"pdf.fonttype":42,"font.family":SANS})

TOTAL=15

def emit(fig):
    pdf.savefig(fig,facecolor=PAPER); plt.close(fig)

def slide(n, dark_tick=SIGNAL):
    fig=plt.figure(figsize=(13.333,7.5), dpi=200, facecolor=PAPER)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Rectangle((0,0),1,1,color=PAPER,zorder=-5))
    ax.add_patch(Rectangle((0.06,0.895),0.032,0.006,color=dark_tick,zorder=2))
    ax.plot([0.06,0.94],[0.072,0.072],color=LINE,lw=0.8)
    ax.text(0.06,0.045,"Campaign 6 · beyond the VQ-VAE · heptokens",family=MONO,
            fontsize=8.5,color=FAINT,va="center")
    ax.text(0.94,0.045,f"{n:02d} / {TOTAL:02d}",family=MONO,fontsize=8.5,color=MUT,
            va="center",ha="right")
    return fig,ax

def eyebrow(ax,t,color=SIGNAL):
    ax.text(0.06,0.855,t.upper(),family=MONO,fontsize=11,color=color,va="center",
            fontweight="bold")
def title(ax,t,y=0.79,size=32):
    fig=ax.figure
    txt=ax.text(0.06,y,t,family=SER,fontsize=size,color=INK,va="top",fontweight="bold")
    fig.canvas.draw()
    maxw=(0.94-0.06)*fig.bbox.width
    w=txt.get_window_extent().width
    if w>maxw: txt.set_fontsize(size*maxw/w*0.98)
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
        ax.text(x+w/2,y+h*0.40,arrow,family=MONO,fontsize=28,color=arrow_color,
                va="center",ha="center",fontweight="bold")
    if lines:
        yy=y+h-0.185
        for ln,col in lines:
            ax.text(cx,yy,"•  "+ln,family=SANS,fontsize=13,color=col,va="top")
            yy-=0.052

def chart_axes(fig,rect):
    iax=fig.add_axes(rect)
    for s in ("top","right"): iax.spines[s].set_visible(False)
    for s in ("left","bottom"): iax.spines[s].set_color(LINE)
    iax.tick_params(colors=MUT,labelsize=9,length=3)
    iax.grid(axis="y",color=LINE,lw=0.6,alpha=0.7,zorder=0)
    iax.set_axisbelow(True)
    return iax

with PdfPages(OUT) as pdf:
    # ---- 1 · title -------------------------------------------------------
    fig,ax=slide(1)
    ax.text(0.06,0.80,"heptokens · campaign 6 · real CMS data (AOJ)".upper(),
            family=MONO,fontsize=11,color=SIGNAL,va="center",fontweight="bold")
    ax.text(0.06,0.62,"Beyond the VQ-VAE",family=SER,fontsize=58,color=INK,
            va="center",fontweight="bold")
    ax.text(0.06,0.47,"Alternate tokenization schemes, put to the test —",family=SER,
            fontsize=22,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.415,"rate-matched VAE frontier · physics grid · FSQ.",family=SER,
            fontsize=22,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.30,"23 runs · headline arms × 3 seeds · pre-registered outcomes",
            family=MONO,fontsize=12,color=MUT,va="center")
    import numpy as np; rng=np.random.default_rng(6)
    for gx,gy,c in [(0.85,0.60,SIGNAL),(0.90,0.43,AMBER),(0.83,0.30,BLUE),(0.88,0.67,MUT)]:
        for _ in range(9):
            a=rng.random()*6.28; r=rng.random()**0.6*0.035
            ax.add_patch(plt.Circle((gx+np.cos(a)*r,gy+np.sin(a)*r*1.6),0.006,
                color=c,alpha=0.5,zorder=0))
    emit(fig)

    # ---- 2 · recap -------------------------------------------------------
    fig,ax=slide(2); eyebrow(ax,"where campaigns 1–5 left us")
    title(ax,"Complementary tokens, one open corner")
    items=[("01","Library owns the criterion","recall@8 0.9991 vs matched VQ 0.927 — ~80× fewer retrieval misses."),
           ("02","VQ owns the information","within-PID pT R² 0.97, PID/charge linearly decodable, 27–30 bits."),
           ("03","Placement decides","physics kernel on a smooth latent works (0.38→0.92); through a codebook it hurts (0.945→0.827)."),
           ("04","Two robust negatives","criterion-shaped VQ codebooks; jet context at low rate."),
           ("05","The open corner","no trained token is sub-raw AND faithful (R²≥0.9) AND criterion-perfect (recall≥0.999).")]
    yy=0.66
    for n_,h_,s_ in items:
        ax.text(0.07,yy,n_,family=MONO,fontsize=15,color=SIGNAL,va="top",fontweight="bold")
        ax.text(0.115,yy,h_,family=SANS,fontsize=16,color=INK,va="top",fontweight="bold")
        ax.text(0.115,yy-0.038,s_,family=SANS,fontsize=13,color=MUT,va="top")
        yy-=0.104
    emit(fig)

    # ---- 3 · the evaluation ---------------------------------------------
    fig,ax=slide(3); eyebrow(ax,"the evaluation")
    title(ax,"Survey everything, rank the frontier, test the cheapest",size=29)
    body(ax,"Full-repo audit + 2023–26 literature scan across three families, then an adversarial "
            "pass over every 'settled' claim. Campaign 6 = the two sanctioned steps + the one new "
            "family that could still deliver a sub-raw criterion-strong discrete token.",0.665,
            size=14.5,color=INKS,width=100)
    cols=[("Discrete",NEG,[("FSQ / BSQ (codebook-free)","→ C6c"),("physics-grid binning","→ C6b"),
                           ("rotation-trick · SimVQ retest","queued"),("Gumbel / dVAE","queued")]),
          ("Continuous",BLUE,[("β-sweep: rate-matched VAE","→ C6a"),("REPA frozen-teacher distill","queued"),
                              ("invertible flow tokens","queued"),("pseudo-2nd-detector proto","queued")]),
          ("Release / consumer",AMBER,[("masked-continuous FM probe","→ C7"),("PID⊕charge symbols ⊕ latent","queued"),
                                       ("entropy codec for μ tokens","queued"),("jet-level side token","queued")])]
    x0=0.06
    for name,c,rows in cols:
        ax.add_patch(FancyBboxPatch((x0,0.135),0.272,0.40,boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
        ax.add_patch(Rectangle((x0,0.155),0.006,0.36,color=c,zorder=2))
        ax.text(x0+0.025,0.49,name.upper(),family=MONO,fontsize=11,color=c,va="top",fontweight="bold")
        yy=0.435
        for r,st in rows:
            ax.text(x0+0.025,yy,r,family=SANS,fontsize=11.5,color=INKS,va="top")
            ax.text(x0+0.025,yy-0.030,st,family=MONO,fontsize=9.5,
                    color=SIGNAL if st.startswith("→") else FAINT,va="top")
            yy-=0.074
        x0+=0.303
    emit(fig)

    # ---- 4 · design ------------------------------------------------------
    fig,ax=slide(4); eyebrow(ax,"campaign 6 · design")
    title(ax,"Three arms, outcomes pre-registered",size=30)
    boxes=[("6a · rate-matched VAE",BLUE,"β × dim × free-bits sweep, kernel on μ",
            "retire VQ if recall ≥ 0.945 AND R² ≥ 0.97 at ≤ 30 bits — else map the saturation"),
           ("6b · physics grid",SIGNAL,"UNTRAINED: ln(1.1) log-pT, 0.1 η/φ bins, exact PID/charge (+½-bin offset copy)",
            "any learned discrete token below this curve is retired"),
           ("6c · FSQ",NEG,"codebook-free quantizer, zero params; kernel on pre-rounding z_e (z_q + recon ablations)",
            "sub-raw criterion token — or upgrade the negative to 'quantization per se'")]
    yy=0.66
    for k,c,d1,d2 in boxes:
        ax.add_patch(FancyBboxPatch((0.06,yy-0.135),0.88,0.15,boxstyle="round,pad=0.006,rounding_size=0.015",
            linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
        ax.add_patch(Rectangle((0.06,yy-0.125),0.006,0.13,color=c,zorder=2))
        ax.text(0.085,yy-0.005,k.upper(),family=MONO,fontsize=11.5,color=c,va="top",fontweight="bold")
        ax.text(0.085,yy-0.048,d1,family=SANS,fontsize=13,color=INK,va="top")
        ax.text(0.085,yy-0.093,"pre-registered:  "+d2,family=SANS,fontsize=11.5,color=MUT,va="top")
        yy-=0.185
    emit(fig)

    # ---- 5 · 6a headline -------------------------------------------------
    fig,ax=slide(5); eyebrow(ax,"6a · rate-matched VAE · headline")
    title(ax,"Criterion parity at ~13 KL-bits — but VQ is not retired")
    ax.text(0.06,0.47,"0.9984 ± 0.0015",family=MONO,fontsize=76,color=POS,va="center",
            fontweight="bold")
    ax.text(0.06,0.335,"criterion recall@8 · β = 0.03, dim 6 · 3 seeds · ~13 KL-bits",
            family=MONO,fontsize=13,color=MUT,va="center")
    body(ax,"Parity with the 96-bit deterministic dim-6 library, and far above plain VQ "
            "(0.945 at 30 bits). But pT R² tops out at ≈ 0.82 anywhere on the sweep — short of the "
            "0.97 retirement bar. Organization-vs-information complementarity survives rate-matching.",
            0.255,size=14.5,color=INKS,width=96)
    emit(fig)

    # ---- 6 · 6a chart: the beta dial --------------------------------------
    fig,ax=slide(6); eyebrow(ax,"6a · the β dial")
    title(ax,"β trades information, not the criterion",size=29)
    betas=[0.03,0.1,0.3,1,3]
    rec_d6=[0.9996,0.992,0.977,0.974,0.9995]; r2_d6=[0.757,0.767,0.821,0.803,0.568]
    kl_d6=[13.3,10.7,0.57,0.18,0.001]
    d4=[(0.1,0.970,0.673),(0.3,0.951,0.634)]; d8=[(0.1,0.996,0.643),(0.3,0.977,0.746)]
    recon=(0.1,0.976,0.920)  # recon-only, 3 seeds
    for ip,(ylab,y6,c4i,c8i,ry,ref,rlab,ylim) in enumerate([
        ("criterion recall@8",rec_d6,1,1,recon[1],0.945,"plain VQ 30 b  0.945",(0.94,1.006)),
        ("within-PID pT R²",r2_d6,2,2,recon[2],0.970,"plain VQ 30 b  0.970",(0.50,1.01))]):
        iax=chart_axes(fig,[0.085+ip*0.47,0.235,0.38,0.46])
        iax.set_xscale("log"); iax.set_xlim(0.02,5); iax.set_ylim(*ylim)
        iax.axhline(ref,color=AMBER,lw=1.4,ls=(0,(4,3)),zorder=1)
        iax.text(4.6,ref,rlab,family=MONO,fontsize=8.5,color=AMBER,va="bottom",ha="right")
        iax.plot(betas,y6,color=BLUE,lw=2,marker="o",ms=7,zorder=3,label="kernel VAE · dim 6")
        iax.plot([b for b,_,_ in d4],[v[c4i] for v in d4],ls="none",marker="^",ms=7,
                 markerfacecolor="none",markeredgecolor=BLUE,markeredgewidth=1.6,zorder=3,label="dim 4")
        iax.plot([b for b,_,_ in d8],[v[c8i] for v in d8],ls="none",marker="s",ms=6.5,
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
    body(ax,"Recall stays ≥ 0.95 across the dial (labels: measured KL-rate); R² peaks ≈ 0.82 at "
            "β = 0.3 — never reaching VQ. Recon-only star: 0.976 / 0.920.",0.115,size=11.5,
            color=MUT,width=200)
    emit(fig)

    # ---- 7 · KL artifact ---------------------------------------------------
    fig,ax=slide(7); eyebrow(ax,"6a · an accounting artifact, measured",color=NEG)
    title(ax,"KL-rate is not a release rate for deterministic-μ tokens")
    ax.text(0.06,0.50,"recall 0.9995   @   0.001 KL-bits",family=MONO,fontsize=34,
            color=NEG,va="center",fontweight="bold")
    body(ax,"At β = 3 the posterior sits essentially on the prior — measured KL ≈ 0 — yet the "
            "released posterior mean still carries the full criterion geometry, because cosine "
            "ranking is scale-invariant: a tiny-norm but structured μ ranks partners perfectly.",
            0.40,size=15,color=INKS,width=94)
    body(ax,"Consequence: Campaign 5's celebrated '~1.4-bit' contrastive VAE was this artifact in "
            "milder form. Until an operational entropy codec exists, continuous-token rates are "
            "quoted as fp16 dim×16; KL-bits are diagnostics only.",0.235,size=14,color=MUT,width=96)
    emit(fig)

    # ---- 8 · recon-only revision ------------------------------------------
    fig,ax=slide(8); eyebrow(ax,"6a · a c5 claim, revised")
    title(ax,"'Reconstruction buys species, not the criterion' — a rate artifact",size=26)
    pole(ax,0.06,0.42,"recon-only VAE · β=1 (C5)","recall 0.383",
         lines=[("the 2.5-bit bottleneck was the cause",MUT)],accent=NEG,y=0.34,h=0.30)
    pole(ax,0.52,0.42,"recon-only VAE · β=0.1 (C6)","recall 0.976 ± 0.004",
         lines=[("pT R² 0.920 · PID/charge probes 1.0",INKS),("~11 KL-bits · 3 seeds",MUT)],
         accent=SIGNAL,y=0.34,h=0.30)
    body(ax,"But it still does not beat the untrained null (0.983 / 0.949) on either axis — the "
            "physics kernel remains the only objective that ever has (0.9984 vs 0.9833).",
            0.24,size=14,color=MUT,width=96)
    emit(fig)

    # ---- 9 · 6b grid verdict ------------------------------------------------
    fig,ax=slide(9); eyebrow(ax,"6b · the untrained reference bites")
    title(ax,"Zero parameters ≈ trained VQ, at fewer bits",size=30)
    pole(ax,0.06,0.42,"physics grid, double","0.9655 @ 26.3 bits",
         lines=[("pT R² 0.949 · probes 1.0",INKS),("computable on ANY detector",INKS),
                ("ln(1.1) pT · 0.1 η/φ · exact PID",MUT)],accent=SIGNAL)
    pole(ax,0.52,0.42,"plain VQ 3×1024 (trained)","0.945 @ 30 bits",
         lines=[("pT R² 0.970 · probes 1.0",INKS),("surviving edge: +0.02 R²",MUT),
                ("purity +0.017",MUT)],accent=AMBER)
    body(ax,"At the 23–30-bit tier, learning a discrete token currently buys almost nothing over "
            "criterion-aligned binning — and the grid is a natural cross-experiment release tier.",
            0.20,size=14,color=MUT,width=96)
    emit(fig)

    # ---- 10 · 6c discrete map ------------------------------------------------
    fig,ax=slide(10); eyebrow(ax,"6c · the discrete family, one map")
    title(ax,"FSQ: the kernel hurts at every rate",size=29)
    iax=chart_axes(fig,[0.10,0.16,0.55,0.55])
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
    body(ax,"No codebook, no EMA, no commitment loss — and criterion-shaping still loses to plain "
            "reconstruction at 10, 15.5, 18 and 24 bits. Both lose to the untrained grid. The C4/C5 "
            "negative was never about codebook dynamics.",0.60,size=13.5,color=INKS,width=38,x=0.68)
    emit(fig)

    # ---- 11 · 6c mechanism --------------------------------------------------
    fig,ax=slide(11); eyebrow(ax,"6c · the mechanism, caught on camera",color=NEG)
    title(ax,"The kernel collapses the code space",size=30)
    iax=chart_axes(fig,[0.10,0.20,0.42,0.46])
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
    body(ax,"The kernel's fixed point drives criterion partners toward coincidence. Any quantizer "
            "turns coincidence into code degeneracy — thousands of particles share one grid point — "
            "and retrieval cannot rank true partners among identical-code impostors.",
            0.62,size=13.5,color=INKS,width=40,x=0.58)
    body(ax,"Verdict: install the criterion on a continuous latent; if a discrete tier is needed, "
            "bin the features (6b), not the embedding.",0.33,size=13.5,color=MUT,width=40,x=0.58)
    emit(fig)

    # ---- 12 · decision map -----------------------------------------------------
    fig,ax=slide(12); eyebrow(ax,"everything on one axis")
    title(ax,"The decision map, at honest rates",size=29)
    iax=chart_axes(fig,[0.10,0.175,0.60,0.545])
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
    body(ax,"Continuous tokens plotted at their fp16 release size per slide 7 "
            "(the β=0.03 arm's 13.3 KL-bits is a diagnostic, not a rate). Full "
            "map incl. every C1–6 arm: benchmarks/figures/library_rate.png.",
            0.55,size=13,color=INKS,width=32,x=0.735)
    emit(fig)

    # ---- 13 · claims ledger ----------------------------------------------------
    fig,ax=slide(13); eyebrow(ax,"the claims ledger")
    title(ax,"What Campaign 6 changed",size=31)
    rows=[("OVERTURNED","'recon buys species, not the criterion'","β=1 rate artifact — 0.976 at β=0.1.",NEG),
          ("RELABELED","the '~1.4-bit' contrastive VAE","KL-rate ≠ release rate for deterministic μ; quote fp16.",NEG),
          ("CLOSED","criterion-shaped discrete tokens","kernel hurts every quantizer at every rate; code degeneracy.",NEG),
          ("CONFIRMED","complementarity (geometry vs information)","survives rate-matching: recall 0.998 @ 13 KL-bits, R² ≤ 0.82.",SIGNAL),
          ("NEW BAR","untrained physics grid","0.966 @ 26.3 bits — every discrete proposal must beat it.",SIGNAL)]
    yy=0.66
    for tag,h_,s_,c in rows:
        ax.text(0.06,yy,tag,family=MONO,fontsize=10.5,color=c,va="top",fontweight="bold")
        ax.text(0.21,yy,h_,family=SANS,fontsize=15.5,color=INK,va="top",fontweight="bold")
        ax.text(0.21,yy-0.038,s_,family=SANS,fontsize=12.5,color=MUT,va="top")
        yy-=0.105
    emit(fig)

    # ---- 14 · release menu ------------------------------------------------------
    fig,ax=slide(14); eyebrow(ax,"key takeaways")
    title(ax,"The release menu, as it stands",size=31)
    pole(ax,0.06,0.272,"criterion + compact","contrastive VAE",
         lines=[("recall 0.998 · dim 6",INKS),("96 fp16 bits (honest rate)",MUT)],
         accent=BLUE,y=0.30,h=0.36)
    pole(ax,0.365,0.272,"discrete tier","physics grid ×2",
         lines=[("0.966 @ 26.3 bits · R² 0.95",INKS),("zero params · any detector",MUT)],
         accent=SIGNAL,y=0.30,h=0.36)
    pole(ax,0.67,0.272,"max fidelity","plain VQ 30 b",
         lines=[("R² 0.970 · probes 1.0",INKS),("only if +0.02 R² matters",MUT)],
         accent=AMBER,y=0.30,h=0.36)
    body(ax,"The corner (sub-raw ∧ faithful ∧ criterion-perfect) is still open — and now provably "
            "not reachable by quantizing the criterion.",0.22,size=14,color=MUT,width=96)
    emit(fig)

    # ---- 15 · next ---------------------------------------------------------------
    fig,ax=slide(15); eyebrow(ax,"next")
    title(ax,"Campaign 7: let the consumer decide",size=31)
    ax.add_patch(FancyBboxPatch((0.06,0.46),0.62,0.18,boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.48),0.006,0.14,color=SIGNAL,zorder=2))
    ax.text(0.09,0.605,"RUNS TODAY",family=MONO,fontsize=10.5,color=SIGNAL,va="top")
    ax.text(0.09,0.565,"Masked-continuous FM probe over the release menu",family=SER,
            fontsize=17,color=INK,va="top",fontweight="bold")
    ax.text(0.09,0.515,"Pretrain a backbone on each frozen export — no Campaign 3–6 winner has a consumer number yet.",
            family=SANS,fontsize=12.5,color=MUT,va="top")
    ax.add_patch(FancyBboxPatch((0.06,0.24),0.62,0.18,boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1,edgecolor=LINE,facecolor=SURF,mutation_aspect=13.333/7.5,zorder=1))
    ax.add_patch(Rectangle((0.06,0.26),0.006,0.14,color=BLUE,zorder=2))
    ax.text(0.09,0.385,"THE OPEN CORNER",family=MONO,fontsize=10.5,color=BLUE,va="top")
    ax.text(0.09,0.345,"REPA-style distillation  ·  PID⊕charge symbols ⊕ kinematic latent",
            family=SER,fontsize=17,color=INK,va="top",fontweight="bold")
    ax.text(0.09,0.295,"Mechanism-changed attacks on recall ≥ 0.999 ∧ R² ≥ 0.9 — plus d0/dz restoration and the anomaly probe.",
            family=SANS,fontsize=12.5,color=MUT,va="top")
    ax.text(0.06,0.16,"North star unchanged and still data-blocked: cross-experiment invariance —",
            family=SER,fontsize=14.5,color=INKS,va="center",fontstyle="italic")
    ax.text(0.06,0.125,"the grid tier and the context-free library are both built for it.",
            family=SER,fontsize=14.5,color=INKS,va="center",fontstyle="italic")
    emit(fig)

    d=pdf.infodict(); d["Title"]="Beyond the VQ-VAE — Campaign 6 Slides"
    d["Author"]="heptokens"; d["Subject"]="Alternate tokenization schemes — rate-matched VAE, physics grid, FSQ"

print("wrote", OUT)
