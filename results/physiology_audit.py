"""Read-only model experiments for the physiology review; writes results only."""
import contextlib
import io
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kidney_model.parameters import build_parameters
from kidney_model.initialization import load_dynamic_state, make_initial_condition_from_file
from kidney_model.state import make_initial_state, unpack_state, pack_state
from kidney_model.transport import DCT_junction, solve_DCT_junction, water_flow, solute_flow
from kidney_model.residuals import implicit_residual
from kidney_model.solver import run_model_positive_adaptive


def metrics(y, p, old=None, dt=None):
    a, c, pressure = unpack_state(y, p)
    dct = solve_DCT_junction(a, c, pressure, p)
    q = np.array([water_flow(k, a, c, pressure, dct[2], p) for k in range(4)])
    f = np.array([[solute_flow(i, k, c, a, pressure, dct, p)
                   for k in range(4)] for i in range(2)])
    mass = np.sum(a[None] * c, axis=(1, 2)) * p.dx
    # A flux is positive toward cortex. All other fluxes are positive toward tip.
    water_in = q[0, 0] + q[1, 0] - q[2, 0] + q[3, 0] - q[3, -1]
    solute_in = f[:, 0, 0] + f[:, 1, 0] - f[:, 2, 0] + f[:, 3, 0] - f[:, 3, -1]
    ratio = (2 * c[0, 3, -1] + c[1, 3, -1]) / p.c_cortex
    mismatch = f[:, 3, 0] - np.array([p.q, 1.0]) * f[:, 2, 0]
    out = dict(
        urine_plasma_ratio=float(ratio), urine_flow=float(q[3, -1]),
        urine_flow_nl_min=float(q[3, -1] * p.area_tot*p.L/p.tau*6e7),
        alpha_min=float(a.min()), concentration_min=float(c.min()),
        alpha_sum_error=float(np.max(np.abs(a.sum(axis=0)-1))),
        pressure_mmhg_range=[float(pressure.min()*p.pressure),float(pressure.max()*p.pressure)],
        dct=dct.tolist(), dct_equation_residual=float(np.max(np.abs(DCT_junction(dct,a,c,pressure,p)))),
        actual_dct_salt_urea_flux_mismatch=mismatch.tolist(),
        cortical_water_removal=float(q[2,0]-q[3,0]),
        water_cortex_tip=q[:,[0,-1]].tolist(),
        solute_cortex_tip=f[:,:,[0,-1]].tolist(),
        tubular_reverse_face_count=int((q[1:] < -1e-12).sum()),
        water_boundary_net_in=float(water_in), solute_boundary_net_in=solute_in.tolist(),
        salt_urea_mass=mass.tolist(),
        steady_residual_linf=float(np.max(np.abs(implicit_residual(y,y,p)))),
        tip_flux_mismatch=(q[1,-1]-q[2,-1]).item(),
    )
    if old is not None:
        ao, co, _ = unpack_state(old,p)
        mass_old = np.sum(ao[None]*co,axis=(1,2))*p.dx
        out['water_balance_error'] = float((a.sum()-ao.sum())*p.dx/dt-water_in)
        out['solute_balance_error'] = ((mass-mass_old)/dt-solute_in).tolist()
    return out


def run(name, row=None, n=20, t_final=1.0, collecting=True):
    p = build_parameters(N=n,dt=.02,collecting_water_permeable=collecting)
    y = make_initial_state(p) if row is None else make_initial_condition_from_file(
        load_dynamic_state(ROOT/'dynamic_stable_v2.npy',row),p)
    log = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(log):
        initial=metrics(y,p)
        result=run_model_positive_adaptive(p,y,t_final=t_final,dt_initial=.02,
            dt_max=.05,print_every_accept=20)
        records=[]
        for idx,state in enumerate(result['history'][1:]):
            report=result['reports'][idx]
            record=metrics(state,p,result['history'][idx],report['dt'])
            record['time']=report['time']
            record['step_residual_linf']=report['residual_Linf']
            records.append(record)
    data=dict(name=name,N=n,initial=initial,states=records,
              t_final_reached=result['t_final_reached'],
              warnings=[str(w.message) for w in caught])
    (ROOT/'results'/f'physiology_{name}_log.txt').write_text(log.getvalue(),encoding='utf-8')
    print(name,'reached',result['t_final_reached'],'steps',len(records),flush=True)
    if records:
        z=records[-1]
        print(' final', {k:z[k] for k in ['urine_plasma_ratio','urine_flow_nl_min',
           'actual_dct_salt_urea_flux_mismatch','cortical_water_removal',
           'tubular_reverse_face_count','steady_residual_linf','solute_balance_error']},flush=True)
    return data


def junction_probe():
    """Construct positive states to compare junction equations with actual fluxes."""
    p=build_parameters(N=20,dt=.02)
    a,c,pressure=unpack_state(make_initial_state(p),p)
    c[:,2,:]=np.array([100.,20.])[:,None]
    c[:,3,:]=np.array([80.,135.])[:,None]
    # Keep the structural shared tip constraint.
    c[:,2,-1]=c[:,1,-1]
    probes=[]
    for a_pressure,c_pressure in [(1.,.5),(.5,1.),(.5,.5)]:
        pressure[2,:]=a_pressure;pressure[3,:]=c_pressure
        probes.append(metrics(pack_state(a,c,pressure,p),p))
    return probes


if __name__=='__main__':
    output={'experiments':[run('row0',0),run('last',-1),run('native'),
                           run('impermeable',0,collecting=False)],
            'junction_probes':junction_probe()}
    (ROOT/'results'/'physiology_audit_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('Saved results/physiology_audit_results.json',flush=True)
