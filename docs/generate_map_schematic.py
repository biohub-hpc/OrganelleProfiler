"""
Generate distinctiveness mAP schematic figure.
Output: docs/distinctiveness_map_schematic.png

Usage:
    python docs/generate_map_schematic.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

FW, FH = 26, 15
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor('white')

CB='#2166ac'; CR='#d6604d'; CG='#1a9850'; CY='#e6ab02'
CP='#762a83'; CT='#01665e'; CO='#b35806'; CK='#d73027'
CGRAY='#888888'; CLIGHT='#f4f4f4'; CDARK='#1a1a1a'; CNTC='#aaaaaa'

def add_ax(l,b,w,h): return fig.add_axes([l,b,w,h])
def ft(x,y,s,**kw): return fig.text(x,y,s,**kw)

np.random.seed(42)

GCOLS=[CB,CR,CG,CP,CY,CT,CK,CO]
GCTR=[(-2.1,1.4),(-1.0,-1.7),(0.9,1.9),(2.2,0.5),
      (-2.7,-0.5),(1.6,-1.9),(0.1,0.2),(2.7,2.3)]
CATMAP={0:'Traf',1:'Traf',5:'Traf',2:'Meta',4:'Meta',
        6:'Trans',7:'Trans',3:'Cell'}
CATCOL={'Traf':CB,'Meta':CG,'Trans':CR,'Cell':CY}

PTS=[]
for i,(cx,cy) in enumerate(GCTR):
    for _ in range(5):
        PTS.append((cx+np.random.randn()*0.22, cy+np.random.randn()*0.22, i))
NTC=[(np.random.randn()*0.30, np.random.randn()*0.28) for _ in range(14)]

def scatter_base(ax):
    ax.set_facecolor(CLIGHT)
    ax.set_xlim(-3.8,3.8); ax.set_ylim(-3.2,3.2)
    ax.set_xlabel('Embedding dim 1',fontsize=8,labelpad=2)
    ax.set_ylabel('Embedding dim 2',fontsize=8,labelpad=2)
    ax.tick_params(labelsize=7,length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('#cccccc')

ft(0.012,0.975,'A',fontsize=15,fontweight='black',color=CDARK)
ft(0.030,0.975,'Three mAP metric types',fontsize=11,fontweight='bold',color=CDARK)
ft(0.012,0.465,'B',fontsize=15,fontweight='black',color=CDARK)
ft(0.030,0.465,'Reporter subspace vs global embedding',fontsize=11,fontweight='bold',color=CDARK)

# ── A1: Activity ─────────────────────────────────────────────────────────
AX,AY,AW,AH = 0.028,0.525,0.128,0.385
ax1 = add_ax(AX,AY,AW,AH)
scatter_base(ax1)
for x,y in NTC:
    ax1.scatter(x,y,c=CNTC,s=45,alpha=0.75,zorder=3,edgecolors='white',linewidths=0.5)
ax1.add_patch(Ellipse((0,0),1.4,1.1,fill=True,facecolor='#e0e0e0',
                       edgecolor=CNTC,lw=1.5,zorder=2,alpha=0.5))
ax1.text(0,-0.04,'NTC',fontsize=8,ha='center',va='center',
         color='#666666',fontweight='bold',zorder=5)
for x,y,i in PTS:
    if i != 0:
        ax1.scatter(x,y,c=GCOLS[i],s=20,alpha=0.12,zorder=2,edgecolors='none')
qpts=[(p[0],p[1]) for p in PTS if p[2]==0]
for x,y in qpts:
    ax1.scatter(x,y,c=CB,s=72,alpha=0.92,zorder=5,edgecolors=CDARK,linewidths=1.5)
qcx=np.mean([p[0] for p in qpts]); qcy=np.mean([p[1] for p in qpts])
for nx,ny in NTC[:6]:
    ax1.plot([qcx,nx],[qcy,ny],color=CGRAY,lw=0.7,alpha=0.5,zorder=1,
             linestyle=(0,(3,3)))
ax1.text(qcx,qcy+0.65,'LAMP1 KO\n(query)',fontsize=8,color=CB,
         fontweight='bold',ha='center',zorder=6)
ax1.annotate('',xy=(0.2,0.05),xytext=(qcx-0.3,qcy-0.5),
             arrowprops=dict(arrowstyle='->',color=CGRAY,lw=1.1,
                             connectionstyle='arc3,rad=-0.2'))
ax1.text(-1.0,-0.7,'vs',fontsize=9,color=CGRAY,fontweight='bold',ha='center')
ft(AX+AW/2,AY+AH+0.012,'Activity mAP',ha='center',fontsize=9.5,fontweight='bold',color=CDARK)
ft(AX+AW/2,AY+AH+0.001,'pos: same geneKO reps  |  neg: NTC reps',
   ha='center',fontsize=7.5,color=CGRAY,style='italic')
ft(AX+AW/2,AY-0.048,'Does this geneKO perturb the\nreporter above NTC noise?',
   ha='center',fontsize=8,color=CDARK,va='top',
   bbox=dict(boxstyle='round,pad=0.25',fc='white',ec='#bbbbbb',lw=0.8))

# ── A2: Distinctiveness ────────────────────────────────────────────────────
BX,BY,BW,BH = 0.175,0.525,0.128,0.385
ax2 = add_ax(BX,BY,BW,BH)
scatter_base(ax2)
for sp in ax2.spines.values():
    sp.set_linewidth(2.5); sp.set_color(CP)
ax2.set_facecolor('#f3edf8')
for x,y,i in PTS:
    ax2.scatter(x,y,c=GCOLS[i],s=52,alpha=0.80,zorder=3,edgecolors='white',linewidths=0.3)
qpts=[(p[0],p[1]) for p in PTS if p[2]==0]
for x,y in qpts:
    ax2.scatter(x,y,c=CB,s=70,zorder=5,edgecolors=CDARK,linewidths=1.8)
qcx2=np.mean([p[0] for p in qpts]); qcy2=np.mean([p[1] for p in qpts])
ax2.text(qcx2,qcy2+0.62,'query',fontsize=7.5,color=CB,fontweight='bold',ha='center',zorder=6)
for idx in [1,2,3,4,5]:
    ax2.annotate('',xy=(GCTR[idx][0],GCTR[idx][1]),
                 xytext=(GCTR[0][0]+0.08,GCTR[0][1]-0.32),
                 arrowprops=dict(arrowstyle='->',color='#aaaaaa',lw=0.85,
                                 connectionstyle=f'arc3,rad={0.07+idx*0.07}'))
ft(BX+BW/2,BY+BH+0.024,'★  Distinctiveness mAP  ★',ha='center',
   fontsize=9.5,fontweight='bold',color=CP)
ft(BX+BW/2,BY+BH+0.012,'pos: same geneKO reps',ha='center',
   fontsize=7.5,color=CP,style='italic')
ft(BX+BW/2,BY+BH+0.001,'neg: ALL other geneKOs',ha='center',
   fontsize=7.5,color=CP,style='italic')
ft(BX+BW/2,BY-0.048,'Is this geneKO phenotype unique\namong ALL other geneKOs?',
   ha='center',fontsize=8,color=CP,fontweight='bold',va='top',
   bbox=dict(boxstyle='round,pad=0.25',fc='#f0e8f5',ec=CP,lw=1.2))
ft(BX+BW/2,BY-0.087,'→ most discriminative for reporter biology',
   ha='center',fontsize=7.5,color=CP,style='italic')

# ── A3: Cohesiveness ──────────────────────────────────────────────────────
CX2,CY2,CW,CH = 0.322,0.525,0.128,0.385
ax3 = add_ax(CX2,CY2,CW,CH)
scatter_base(ax3)
pathway_genes = {
    'Pathway A\n(Trafficking)': {'col':CB,'pts':[(-2.1,1.4),(-1.8,1.1),(-2.3,1.0),(-1.6,1.7)]},
    'Pathway B\n(Metabolism)':  {'col':CG,'pts':[(1.6,1.8),(1.9,1.4),(1.4,1.5),(2.0,1.1)]},
    'Pathway C\n(Translation)': {'col':CR,'pts':[(-1.1,-1.6),(-0.8,-1.9),(-1.4,-1.2),(-0.9,-1.4)]},
}
np.random.seed(7)
ctrs={}
for pname,pdata in pathway_genes.items():
    col=pdata['col']
    spts=[]
    for cx,cy in pdata['pts']:
        x=cx+np.random.randn()*0.14; y=cy+np.random.randn()*0.14
        spts.append((x,y))
        ax3.scatter(x,y,c=col,s=65,alpha=0.88,zorder=4,edgecolors=CDARK,linewidths=1.2)
    for ii in range(len(spts)):
        for jj in range(ii+1,len(spts)):
            ax3.plot([spts[ii][0],spts[jj][0]],[spts[ii][1],spts[jj][1]],
                     color=col,lw=1.0,alpha=0.45,zorder=3)
    xs=[p[0] for p in spts]; ys=[p[1] for p in spts]
    ax3.add_patch(Ellipse((np.mean(xs),np.mean(ys)),1.3,0.85,angle=10,
                           fill=False,edgecolor=col,lw=2.0,linestyle='--',alpha=0.8,zorder=5))
    ctrs[pname]=(np.mean(xs),np.mean(ys))
    offsets={'Pathway A\n(Trafficking)':(0.0,0.65),
             'Pathway B\n(Metabolism)':(0.6,0.0),
             'Pathway C\n(Translation)':(0.0,-0.65)}
    dx,dy=offsets.get(pname,(0,0.5))
    ax3.text(np.mean(xs)+dx,np.mean(ys)+dy,pname,fontsize=7,color=col,
             fontweight='bold',ha='center',va='center',zorder=6)
for x,y,i in PTS:
    if CATMAP.get(i,'Other') not in ('Traf','Meta','Trans'):
        ax3.scatter(x,y,c='#cccccc',s=18,alpha=0.25,zorder=2,edgecolors='none')
pnames=list(ctrs.keys())
for ia,ib in [(0,1),(1,2)]:
    pa=ctrs[pnames[ia]]; pb=ctrs[pnames[ib]]
    mx=(pa[0]+pb[0])/2; my=(pa[1]+pb[1])/2
    ax3.annotate('',xy=pb,xytext=pa,
                 arrowprops=dict(arrowstyle='<->',color='#cccccc',lw=1.2,
                                 connectionstyle='arc3,rad=0.15'))
    ax3.text(mx,my,'neg\npairs',fontsize=6.5,ha='center',va='center',
             color='#999999',style='italic',
             bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.8))
ft(CX2+CW/2,CY2+CH+0.012,'Cohesiveness mAP',ha='center',
   fontsize=9.5,fontweight='bold',color=CDARK)
ft(CX2+CW/2,CY2+CH+0.001,'pos: same pathway/category  |  neg: other pathways',
   ha='center',fontsize=7.5,color=CGRAY,style='italic')
ft(CX2+CW/2,CY2-0.048,'Do pathway members cluster\ntogether in this reporter space?',
   ha='center',fontsize=8,color=CDARK,va='top',
   bbox=dict(boxstyle='round,pad=0.25',fc='white',ec='#bbbbbb',lw=0.8))
ft(0.232,0.488,
   '◀── all three: normalized by global baseline → reporter[cat] / all_reporters[cat] ──▶',
   ha='center',fontsize=8,color=CGRAY,style='italic')

# ── Panel B ───────────────────────────────────────────────────────────────
cat_info={
    'Mem.Traffic':(CB,[(-1.8,1.2),(-1.5,0.7),(-2.1,0.5)]),
    'Metabolism': (CG,[(1.6,1.8),(1.9,1.3),(1.4,1.5)]),
    'Translation':(CR,[(-1.2,-1.5),(-0.8,-1.8),(-1.5,-1.1)]),
    'Signaling':  (CP,[(1.2,-1.2),(1.6,-0.8),(0.9,-1.5)]),
    'Cell Cycle': (CY,[(0.1,0.2),(0.5,-0.2),(-0.2,0.4)]),
}
def draw_embed(ax,al,ab,aw,ah,spread,highlight,title,sub1,sub2):
    np.random.seed(5)
    ax.set_facecolor(CLIGHT)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    ax.set_xlabel('dim 1',fontsize=8,labelpad=2)
    ax.set_ylabel('dim 2',fontsize=8,labelpad=2)
    ax.tick_params(labelsize=7,length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('#cccccc')
    for cat,(col,ctrs2) in cat_info.items():
        for cx,cy in ctrs2:
            for _ in range(5):
                x=cx+np.random.randn()*spread; y=cy+np.random.randn()*spread
                hl=cat in highlight
                ax.scatter(x,y,c=col,s=52 if hl else 26,
                           alpha=0.88 if hl else 0.25,zorder=3,edgecolors='none')
        if cat in highlight:
            ax.add_patch(Ellipse(
                (np.mean([c[0] for c in ctrs2]),np.mean([c[1] for c in ctrs2])),
                1.3,0.9,angle=15,fill=False,edgecolor=col,lw=2,linestyle='--',zorder=4))
    ft(al+aw/2,ab+ah+0.016,title,ha='center',fontsize=9.5,fontweight='bold',color=CDARK)
    ft(al+aw/2,ab+ah+0.005,sub1,ha='center',fontsize=7.5,color=CDARK)
    ft(al+aw/2,ab+ah-0.006,sub2,ha='center',fontsize=7.5,color=CGRAY,style='italic')

B1L,B1B,B1W,B1H=0.028,0.065,0.188,0.360
B2L,B2B,B2W,B2H=0.240,0.065,0.188,0.360
ax_b1=add_ax(B1L,B1B,B1W,B1H)
draw_embed(ax_b1,B1L,B1B,B1W,B1H,0.62,[],'Global embedding',
           'All ~1500 PCs, all reporters combined','→ no reporter-specific sharpening')
ax_b2=add_ax(B2L,B2B,B2W,B2H)
draw_embed(ax_b2,B2L,B2B,B2W,B2H,0.19,['Mem.Traffic'],'LAMP1 subspace',
           '31 PCs — lysosome/trafficking lens','→ trafficking genes sharply separated')
leg_b=[mpatches.Patch(color=v[0],label=k) for k,v in cat_info.items()]
ax_b2.legend(handles=leg_b,fontsize=6.5,loc='lower right',framealpha=0.9,
             title='category',title_fontsize=7,handlelength=1,borderpad=0.4,labelspacing=0.3)
fig.text(0.228,0.245,'→',fontsize=26,color=CGRAY,ha='center',va='center')

# ── Panel C: Flowchart ────────────────────────────────────────────────────
ft(0.458,0.975,'C',fontsize=15,fontweight='black',color=CDARK)
ft(0.474,0.975,'Normalized distinctiveness mean_mAP — computation pipeline',
   fontsize=11,fontweight='bold',color=CDARK)
ax_c=add_ax(0.455,0.03,0.540,0.930)
ax_c.set_xlim(0,10); ax_c.set_ylim(0,10.5)
ax_c.axis('off')

def cbox(x,y,w,h,lines,fc,fontsize=8.5,tc='white'):
    ax_c.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=0.15',facecolor=fc,edgecolor='white',lw=1.6,zorder=3))
    if isinstance(lines,str): lines=[lines]
    dy=h/(len(lines)+1)
    for j,ln in enumerate(lines):
        ax_c.text(x,y+h/2-dy*(j+1),ln,fontsize=fontsize,ha='center',va='center',
                  color=tc,fontweight='bold',zorder=4)
def carr(x1,y1,x2,y2,col=CGRAY):
    ax_c.annotate('',xy=(x2,y2),xytext=(x1,y1),
                  arrowprops=dict(arrowstyle='-|>',color=col,lw=1.8,mutation_scale=14))

cbox(5,10.05,9.0,0.72,
     ['PCA-optimized guide embedding',
      'obs = geneKO replicates   ×   vars = reporter PCA components'],'#3d3d3d',fontsize=9)
carr(2.5,9.69,2.5,9.18); carr(7.5,9.69,7.5,9.18)
cbox(2.5,8.90,4.2,0.56,['All reporters (~1500 PCs)','→  global baseline run'],CGRAY)
cbox(7.5,8.90,4.2,0.56,['Single reporter (30–50 PCs)','→  per-reporter run  ×37'],CB)
carr(2.5,8.62,2.5,7.98); carr(7.5,8.62,7.5,7.98)
cbox(2.5,7.67,4.2,0.62,
     ['Distinctiveness mAP','pos: same geneKO replicates','neg: ALL other geneKOs'],CGRAY,fontsize=8)
cbox(7.5,7.67,4.2,0.62,
     ['Distinctiveness mAP','pos: same geneKO replicates','neg: ALL other geneKOs'],CB,fontsize=8)
ax_c.text(0.6,7.67,'same\nformula',fontsize=7,ha='center',va='center',color='#aaaaaa',style='italic')
carr(2.5,7.36,2.5,6.72); carr(7.5,7.36,7.5,6.72)
cbox(2.5,6.44,4.2,0.56,
     ['Aggregate per ontology category','mean(mAP per gene)  →  baseline[cat]'],CGRAY,fontsize=8)
cbox(7.5,6.44,4.2,0.56,
     ['Aggregate per ontology category','mean(mAP per gene)  →  reporter[cat]'],CB,fontsize=8)
carr(2.5,6.16,3.8,5.44); carr(7.5,6.16,6.2,5.44)
cbox(5.0,5.14,4.6,0.58,
     ['Normalization','spoke[cat]  =  reporter[cat]  /  baseline[cat]'],CP,fontsize=9)
ax_c.text(5.0,4.73,
    '1.0 = global baseline   |   >1.0 = enriched   |   <1.0 = depleted',
    fontsize=8,ha='center',va='center',color=CP,style='italic')
carr(5.0,4.48,5.0,3.80,col=CP)

cats=['Mem.\nTraffic','Metab.','Cell\nCycle','Prot.\nHomeo.',
      'Signal.','Cytosk.','Gene\nExpr.','Transl.']
n=len(cats)
ang=np.linspace(0,2*np.pi,n,endpoint=False).tolist()+[0]
vL=[2.1,0.4,0.5,0.9,0.6,0.7,0.7,0.5]+[2.1]
vT=[0.5,2.3,0.4,0.8,0.6,0.7,0.7,0.6]+[0.5]
ax_r=fig.add_axes([0.590,0.067,0.240,0.330],polar=True)
ax_r.set_facecolor('#f9f9f9')
ax_r.plot(ang,[1.0]*9,'--',color='#aaaaaa',lw=1.1,alpha=0.7,zorder=2)
ax_r.fill(ang,vL,alpha=0.18,color=CB,zorder=3)
ax_r.plot(ang,vL,'o-',color=CB,lw=2.0,ms=4.5,zorder=4,label='LAMP1')
ax_r.fill(ang,vT,alpha=0.18,color=CG,zorder=3)
ax_r.plot(ang,vT,'o-',color=CG,lw=2.0,ms=4.5,zorder=4,label='TOMM20')
ax_r.set_xticks(ang[:-1]); ax_r.set_xticklabels(cats,fontsize=7.5)
ax_r.set_ylim(0,2.8)
ax_r.set_yticks([1.0,2.0]); ax_r.set_yticklabels(['1×','2×'],fontsize=6.5,color='#888888')
ax_r.tick_params(pad=5)
ax_r.legend(loc='upper right',bbox_to_anchor=(1.52,1.12),fontsize=8,
            framealpha=0.9,handlelength=1.5)
ft(0.710,0.408,'Example output — normalized mean_mAP',
   ha='center',fontsize=9,fontweight='bold',color=CDARK)
ax_c.text(3.4,1.6,'LAMP1\n2.1× trafficking\nenriched',fontsize=8,ha='center',va='center',
    color=CB,bbox=dict(boxstyle='round,pad=0.3',fc='#e8f0f8',ec=CB,lw=1))
ax_c.text(6.6,1.6,'TOMM20\n2.3× metabolism\nenriched',fontsize=8,ha='center',va='center',
    color=CG,bbox=dict(boxstyle='round,pad=0.3',fc='#e8f5e9',ec=CG,lw=1))
ax_c.text(5.0,0.56,'baseline ring at 1.0 (dashed)',fontsize=8,
          ha='center',va='center',color='#888888',style='italic')
ax_c.add_patch(FancyBboxPatch((0.1,0.08),9.8,0.42,
    boxstyle='round,pad=0.1',facecolor='#3d3d3d',edgecolor='none',zorder=3))
ax_c.text(5.0,0.29,
    'Run separately:   all cells (650k–40M per reporter, full statistical power)'
    '   |   downsampled (~750k cells, faster, QC validation)',
    fontsize=8.5,ha='center',va='center',color='white',fontweight='bold',zorder=4)

import os
out = os.path.join(os.path.dirname(__file__), 'distinctiveness_map_schematic.png')
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')

if __name__ == '__main__':
    pass
