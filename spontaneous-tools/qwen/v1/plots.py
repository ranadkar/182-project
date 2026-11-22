"""V1 figures from the released CSV tables. Separate points represent paired training seeds."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

COL={'base':'#666666','explicit':'#27668f','factual':'#00806b','neutral':'#bd651b','continuation':'#854aa1'}
LABEL={'base':'Base','explicit':'Explicit request','factual':'Factual acknowledgment','neutral':'Neutral acknowledgment','continuation':'Continuation'}

def load(name):
    with (STUDY/'results'/name).open() as f:return list(csv.DictReader(f))

def export(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(STUDY/'figures'/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)

def main():
    est=load('estimates.csv');counts=load('by_seed.csv');curves=load('loss_curves.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False,
        'figure.facecolor':'white','pdf.fonttype':42,'axes.axisbelow':True})
    def row(arm,ev='discrepancy',metric='target_match',ref='',role='environment',suite='final'):
        return next(r for r in est if r['analysis']=='cell' and r['unit']=='family' and r['arm']==arm and r['reference']==ref
              and r['suite']==suite and r['evidence']==ev and r['delivery']==role and r['metric']==metric)
    def triple(r):return [100*float(r[k]) for k in ('estimate','low','high')]
    fig,axes=plt.subplots(1,2,figsize=(10,3.25),layout='constrained',gridspec_kw={'width_ratios':[1,1.05]})
    for j,a in enumerate(['base',*ARMS]):
        y,lo,hi=triple(row(a))
        axes[0].plot([lo,hi],[j,j],color=COL[a],lw=2)
        axes[0].scatter(y,j,color=COL[a],s=40,zorder=3)
        if a!='base':
            rs=sorted((r for r in counts if r['arm']==a and r['suite']=='final' and r['evidence']=='discrepancy' and r['delivery']=='environment'),key=lambda r:r['seed'])
            for dy,r in zip([-.15,0,.15],rs):
                axes[0].scatter(100*int(r['target_match'])/int(r['n']),j+dy,s=16,facecolor='white',edgecolor=COL[a],zorder=4)
    axes[0].set(yticks=np.arange(5),yticklabels=[LABEL[a] for a in ['base',*ARMS]],xlabel='Target-matching state changes (%)',title='External discrepancy notifications',xlim=(-2,102))
    pairs=[('continuation','explicit'),('factual','explicit'),('neutral','explicit'),('factual','neutral'),('continuation','neutral')]
    for j,(a,b) in enumerate(pairs):
        y,lo,hi=triple(row(a,ref=b))
        axes[1].plot([lo,hi],[j,j],color=COL[a],lw=2);axes[1].scatter(y,j,color=COL[a],s=32,zorder=3)
    axes[1].axvline(0,ls='--',lw=.8,color='#555555')
    axes[1].set(yticks=np.arange(len(pairs)),yticklabels=[a.title()+' − '+b.title() for a,b in pairs],xlabel='Paired difference (percentage points)',title='Differences between training arms')
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    export(fig,'primary')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),sharey=True,layout='constrained')
    for ax,e in zip(axes,['ordinary','ambiguous','conflict']):
        for j,a in enumerate(['base',*ARMS]):
            for shift,m,marker in [(-.11,'write','s'),(.11,'changed','o')]:
                v,lo,hi=triple(row(a,e,m)); ax.plot([lo,hi],[j+shift]*2,color=COL[a],lw=1,alpha=.7)
                ax.scatter(v,j+shift,marker=marker,s=30,edgecolor=COL[a],facecolor='white' if m=='write' else COL[a],zorder=3)
        ax.set_xlim(-2,102);ax.grid(axis='x',alpha=.2);ax.set_title(e.title())
    axes[0].set(yticks=range(5),yticklabels=[LABEL[a] for a in ['base',*ARMS]]);axes[0].invert_yaxis()
    fig.supxlabel('External notifications (%)     □ Executed write     ● Net state change',fontsize=10)
    export(fig,'unnecessary-actions')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for a in ARMS:
        r=[x for x in curves if x['arm']==a and x['pilot']=='0']
        for ax,m in zip(axes,['training_nll','validation_nll']):
            steps=sorted({int(x['step']) for x in r if x[m]!=''})
            vals=np.array([[float(next(x[m] for x in r if int(x['seed'])==s and int(x['step'])==t)) for t in steps] for s in SEEDS])
            if m=='training_nll':
                vals=np.array([np.convolve(v,np.ones(4)/4,mode='valid') for v in vals]);steps=steps[3:]
            for v in vals:ax.plot(steps,v,color=COL[a],lw=.65,alpha=.3)
            ax.plot(steps,vals.mean(0),label=LABEL[a],color=COL[a],lw=1.8,marker='o' if m=='validation_nll' else None,ms=3)
    axes[0].set_title('Training · four-update moving mean');axes[1].set_title('Validation · fixed checkpoints')
    for ax in axes:ax.set(xlabel='Optimizer update',ylabel='NLL per supervised token');ax.grid(alpha=.18)
    axes[1].legend(fontsize=8,frameon=False)
    export(fig,'training')
    if any(r['suite'] == 'replication' for r in est):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7), sharex=True, sharey=True, layout='constrained')
        settings = [
            ('Independent external facts', 'discrepancy', 'environment', 'target_match', 'final'),
            ('Earlier external facts', 'discrepancy', 'environment', 'target_match', 'replication'),
            ('Independent authorized tasks', 'authorized', 'continuation', 'authorized_correct', 'final'),
        ]
        comparison_arms = ['factual', 'neutral', 'continuation']
        for ax, (title, evidence, role, metric, suite) in zip(axes, settings):
            ax.axvline(0, ls='--', lw=.8, color='#555555')
            for y, a in enumerate(comparison_arms):
                mean, low, high = triple(row(a, evidence, metric, 'explicit', role, suite))
                ax.plot([low, high], [y, y], color=COL[a], lw=2)
                ax.scatter(mean, y, color=COL[a], s=36, zorder=3)
            ax.set_title(title, fontsize=10)
            ax.grid(axis='x', alpha=.2)
        axes[0].set(yticks=range(3), yticklabels=[LABEL[a] for a in comparison_arms])
        axes[0].invert_yaxis()
        for ax in axes:
            ax.margins(x=.07, y=.24)
        fig.supxlabel('Difference from explicit-request training (percentage points)', fontsize=10)
        export(fig, 'transfer')
if __name__=='__main__':main()
