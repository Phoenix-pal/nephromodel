"""Plot saved review experiments without rerunning or modifying the model."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
data=json.loads((root/'physiology_audit_results.json').read_text())
labels={'row0':'Maximum seed, permeable CD','last':'Final seed, permeable CD',
        'native':'Uniform native seed','impermeable':'Maximum seed, impermeable CD'}
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
for experiment,color in zip(data['experiments'],colors):
    states=experiment['states'];times=np.array([s['time'] for s in states])
    initial=experiment['initial'];label=labels[experiment['name']]
    axes[0,0].plot(np.r_[0,times], [initial['urine_plasma_ratio']]+[s['urine_plasma_ratio'] for s in states],color=color,label=label)
    axes[0,1].plot(times,[s['urine_flow_nl_min'] for s in states],color=color)
    axes[1,0].plot(times,[s['tubular_reverse_face_count'] for s in states],color=color)
    axes[1,1].semilogy(times,[max(max(abs(v) for v in s['actual_dct_salt_urea_flux_mismatch']),1e-16) for s in states],color=color)
axes[0,0].set(title='Outlet concentration retains a strong seed effect',ylabel='Urine / plasma osmolarity',ylim=(0,5))
axes[0,0].legend(fontsize=8,loc='lower left')
axes[0,1].axhline(10.02,color='#555555',linestyle='--',linewidth=1)
axes[0,1].text(.5,11.5,'Prescribed D inlet: 10.02 nL/min',fontsize=8,color='#555555')
axes[0,1].set(title='Urine flow depends on boundary and storage responses',ylabel='Urine flow (nL/min, equivalent model unit)')
axes[1,0].set(title='The solver accepts reverse tubular flows',ylabel='Tubular faces with negative oriented flow')
axes[1,1].set(title='Actual DCT fluxes can violate its transfer rules',ylabel='Max salt/urea junction mismatch (scaled flux)')
for ax in axes.flat:
    ax.set_xlabel('Nondimensional time')
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Physiology audit: short transient checks, N = 20\nThese are not steady states or predictions of maximum concentrating capacity',fontsize=14)
fig.savefig(root/'physiology_audit.png',dpi=180)
fig.savefig(root/'physiology_audit.pdf')
plt.close(fig)
