"""Isolated forward/inverse canard-command table benchmark; production code unchanged.

Run: PYTHONPATH=. .venv/bin/python examples/benchmark_canard_tables.py
Results and a frozen configuration are written under outputs/canard_table_benchmark.
The inverse uses demand normalized by full-deflection authority, preserving a
rectangular grid without interpolating across a moving saturation boundary.
"""
import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Flight.canards import CanardAerodynamics, CanardSystem
from simulation_types import AtmosState

ORIGINAL = CanardSystem.command_deflection
OUT = ROOT / 'outputs/canard_table_benchmark'


def generate(cfg):
    start = time.perf_counter()
    aero = CanardAerodynamics(cfg['canards'].get('aero', {}))
    geo = aero.surface_geometry(cfg['canards']['root_chord_m'], cfg['vehicle']['OMLD']/2)
    mach = np.linspace(0, 10, 401)
    alpha = np.deg2rad(np.linspace(0, 15, 151))
    delta = np.deg2rad(np.linspace(0, cfg['canards']['max_deflection_deg'], 201))
    fraction = np.linspace(0, 1, 201)
    forward = np.empty((len(mach), len(alpha), len(delta)))
    inverse = np.empty_like(forward)
    worst_decrease = 0.
    for i, m in enumerate(mach):
        curve = aero.roll_coefficient(geo, m, delta[None, :], alpha[:, None])
        if not np.all(np.isfinite(curve)):
            raise ValueError('Nonfinite forward response')
        worst_decrease = min(worst_decrease, float(np.min(np.diff(curve, axis=1))))
        if worst_decrease < -1e-10:
            raise ValueError(f'Nonmonotonic response: {worst_decrease}')
        forward[i] = curve
        for j, c in enumerate(curve):
            inverse[i,j] = np.interp(fraction*c[-1], c, delta) if c[-1] > 0 else 0.
    with h5py.File(OUT/'command_tables.h5', 'w') as f:
        for key, value in dict(mach=mach, alpha=alpha, delta=delta, fraction=fraction,
                               forward=forward, inverse=inverse).items():
            f.create_dataset(key, data=value, compression='gzip')
        f.attrs['geometry'] = json.dumps(asdict(geo))
    return dict(generation_s=time.perf_counter()-start,shape=list(forward.shape),
                worst_decrease=worst_decrease, file_bytes=(OUT/'command_tables.h5').stat().st_size)


class Tables:
    def __init__(self):
        with h5py.File(OUT/'command_tables.h5') as f:
            self.mach,self.alpha,self.delta,self.fraction,self.forward,self.inverse = (
                f[k][:] for k in ['mach','alpha','delta','fraction','forward','inverse'])
        self.fallbacks = 0

    @staticmethod
    def bracket(axis, value):
        i = max(0, min(len(axis)-2, int(np.searchsorted(axis, value)-1)))
        return i, (value-axis[i])/(axis[i+1]-axis[i])

    def command(self, system, mode, mach, alpha, moment, scale, active):
        alpha = abs(alpha)
        if not (0 <= mach <= self.mach[-1] and alpha <= self.alpha[-1]):
            self.fallbacks += 1
            return ORIGINAL(system,mach,alpha,moment,scale,active)
        lift_limit = math.radians(system.config.get(
            'max_measured_angle_deg' if mach <= 2.8 else 'max_extension_angle_deg',
            22. if mach <= 2.8 else 15.))
        limit = min(math.radians(system.config['max_deflection_deg']),max(0.,lift_limit-alpha))
        i,u = self.bracket(self.mach,mach)
        j,v = self.bracket(self.alpha,alpha)
        weights = ((i,j,(1-u)*(1-v)),(i+1,j,u*(1-v)),(i,j+1,(1-u)*v),(i+1,j+1,u*v))
        if mode == 'forward':
            curve = sum(w*self.forward[a,b] for a,b,w in weights)
            maximum = float(np.interp(limit,self.delta,curve))
        else:
            k,wdelta = self.bracket(self.delta,limit)
            maximum = sum(w*((1-wdelta)*self.forward[a,b,k]+wdelta*self.forward[a,b,k+1]) for a,b,w in weights)
        torque = scale*maximum
        demand = abs(moment)
        if not active or demand == 0:
            return 0.,torque,False
        if torque <= 0:
            return 0.,torque,True
        required = demand/scale
        if required >= maximum:
            return limit,torque,required > maximum
        if mode == 'forward':
            command = float(np.interp(required,curve,self.delta))
        else:
            full = sum(w*self.forward[a,b,-1] for a,b,w in weights)
            k,s = self.bracket(self.fraction,required/full)
            command = sum(w*((1-s)*self.inverse[a,b,k]+s*self.inverse[a,b,k+1]) for a,b,w in weights)
        return min(limit,max(0.,command)),torque,False


def saturation_shortcut(self, mach, alpha, moment, scale, active):
    lift_limit = math.radians(self.config.get(
        'max_measured_angle_deg' if mach <= 2.8 else 'max_extension_angle_deg',
        22. if mach <= 2.8 else 15.))
    limit = min(math.radians(self.config['max_deflection_deg']), max(0., lift_limit-abs(alpha)))
    maximum = float(self.aero.roll_coefficient(self.geometry,mach,limit,alpha))
    torque = scale*maximum
    demand = abs(moment)
    if not active or demand == 0: return 0.,torque,False
    if torque <= 0: return 0.,torque,True
    required = demand/scale
    if required >= maximum: return limit,torque,required > maximum
    angles = np.linspace(0,limit,240)
    curve = self.aero.roll_coefficient(self.geometry,mach,angles,alpha)
    return float(np.interp(required,curve,angles)),torque,False


def install(tables, mode):
    CanardSystem.command_deflection = (ORIGINAL if mode == 'current' else
        saturation_shortcut if mode == 'shortcut' else
        lambda self,*args: tables.command(self,mode,*args))


def flight(cfg, mode, index):
    from simulation import simulate
    from diagnostics.errors import EvaluationFailure
    tables = None
    start = time.perf_counter()
    if mode in ['forward','direct']:tables = Tables()
    load_s = time.perf_counter()-start
    install(tables,mode)
    original_evaluate = CanardSystem.evaluate
    totals = dict(calls=0,evaluation_s=0.,saturated_calls=0,active_calls=0)
    trace=[];system_metadata={}
    def evaluated(self, atmosphere, alpha, forcing, active, cg):
        start=time.perf_counter()
        result=original_evaluate(self,atmosphere,alpha,forcing,active,cg)
        totals['evaluation_s']+=time.perf_counter()-start;totals['calls']+=1
        totals['saturated_calls']+=int(result.saturated);totals['active_calls']+=int(active)
        if mode=='current' and index==0:
            trace.append([atmosphere.Ma,alpha,forcing,float(active),cg,atmosphere.q])
            if not system_metadata:
                system_metadata.update(config=self.config, geometry=asdict(self.geometry),count=self.count,station_m=self.station_m)
        return result
    CanardSystem.evaluate=evaluated
    start=time.perf_counter();cpu=time.process_time();failure=None
    try:r=simulate(cfg)
    except EvaluationFailure as e:r=e.partial_result;failure=e.diagnostic
    data=dict(mode=mode,index=index,wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,
              load_s=load_s,failure=failure,**totals,fallbacks=tables.fallbacks if tables else 0)
    if r is not None:
        for k in ['apogee','apogee_time','burn_duration','max_q','max_aoa_deg','min_stability_calibers','min_slope_stability_calibers','rail_exit_time']:
            data[k]=getattr(r,k)
        if r.roll:
            data['final_roll_angle_rad']=r.roll['phi'][-1];data['final_roll_rate_rad_s']=r.roll['p'][-1]
    if trace:
        np.save(OUT/'flight_trace.npy',np.asarray(trace));(OUT/'system.json').write_text(json.dumps(system_metadata,indent=2))
    return data


def micro():
    from Flight.canards import CanardGeometry
    meta=json.loads((OUT/'system.json').read_text())
    system=object.__new__(CanardSystem)
    system.config=meta['config'];system.geometry=CanardGeometry(**meta['geometry'])
    system.aero=CanardAerodynamics(system.config.get('aero',{}));system.count=meta['count'];system.station_m=meta['station_m']
    tables=Tables()
    trace=np.load(OUT/'flight_trace.npy')
    # Same flight states and ordering for all methods; no integration differences.
    indices=np.linspace(0,len(trace)-1,min(1500,len(trace)),dtype=int)
    samples=[(AtmosState(288.,101325.,1.,1e-5,340.,q,m),alpha,forcing,bool(active),cg)
             for m,alpha,forcing,active,cg,q in trace[indices]]
    unsaturated=[]
    for i,(atm,alpha,forcing,active,cg) in enumerate(samples):
        scale=system.count*atm.q*system.geometry.area*system.geometry.body_factor*system.geometry.moment_arm*system.config['net_roll_multiplier']
        _,capacity,_=ORIGINAL(system,atm.Ma,alpha,0.,scale,False)
        demand=(.05+.9*(i%101)/100)*capacity
        unsaturated.append((atm,alpha,demand,True,cg))
    timings=[];accuracy={};saturation={}
    for label,inputs in [('flight',samples),('unsaturated',unsaturated)]:
        outputs={}
        for repeat,order in enumerate([['current','forward','direct','shortcut'],['direct','shortcut','current','forward'],['forward','direct','shortcut','current']]):
            for mode in order:
                install(tables,mode)
                for sample in inputs[:20]:system.evaluate(*sample)
                start=time.perf_counter();cpu=time.process_time()
                values=[system.evaluate(*sample) for sample in inputs]
                timings.append(dict(group=label,mode=mode,repeat=repeat,wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,samples=len(inputs)))
                outputs[mode]=values
        saturation[label]=dict(samples=len(inputs),saturated=sum(int(x.saturated) for x in outputs['current']))
        accuracy[label]={}
        for mode in ['forward','direct','shortcut']:
            errors={}
            for key in ['deflection_rad','control_roll_moment_nm','max_roll_moment_nm','drag_n','normal_force_n','pitch_moment_nm']:
                differences=np.array([getattr(a,key)-getattr(b,key) for a,b in zip(outputs[mode],outputs['current'])])
                errors[key]=dict(max_abs=float(np.max(np.abs(differences))),rms=float(np.sqrt(np.mean(differences**2))))
            errors['saturation_mismatches']=sum(int(a.saturated!=b.saturated) for a,b in zip(outputs[mode],outputs['current']))
            accuracy[label][mode]=errors
    # Off-grid domain checks, plus inactive/zero demand/saturation/angle limits.
    rng=np.random.default_rng(1021);grid=[]
    for m,alpha,fraction in zip(rng.uniform(.05,10,3000),rng.uniform(0,np.deg2rad(15),3000),rng.uniform(0,1.2,3000)):
        _,capacity,_=ORIGINAL(system,m,alpha,0.,1.,False)
        grid.append((m,alpha,fraction*capacity,1.,True))
    for m in [0.,.6,1.2,2.,2.36,2.8,2.800001,4.,10.,10.01]:
        for a in [0.,7.,12.,14.999,15.]:
            for active in [False,True]:
                _,capacity,_=ORIGINAL(system,m,np.deg2rad(a),0.,1.,False)
                for f in [0.,.1,.5,.999,1.,1.001,1.2]:grid.append((m,np.deg2rad(a),f*capacity,1.,active))
    original=[ORIGINAL(system,*sample) for sample in grid]
    domain={}
    for mode in ['forward','direct']:
        angular=[];torque=[];mismatch=0
        for sample,expected in zip(grid,original):
            got=tables.command(system,mode,*sample)
            angular.append(abs(got[0]-expected[0]))
            achieved=float(system.aero.roll_coefficient(system.geometry,sample[0],got[0],sample[1]))
            target=float(system.aero.roll_coefficient(system.geometry,sample[0],expected[0],sample[1]))
            torque.append(abs(achieved-target)/max(expected[1],1e-12))
            mismatch+=got[2]!=expected[2]
        domain[mode]=dict(samples=len(grid),max_command_error_deg=float(np.rad2deg(max(angular))),p99_command_error_deg=float(np.rad2deg(np.quantile(angular,.99))),max_torque_error_fraction_capacity=max(torque),p99_torque_error_fraction_capacity=float(np.quantile(torque,.99)),saturation_mismatches=int(mismatch))
    return dict(timings=timings,sample_accuracy=accuracy,saturation=saturation,domain_accuracy=domain)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['current','forward','direct','shortcut','micro'])
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--low-demand',action='store_true',help='Diagnostic: quarter the configured fin cant')
    args=parser.parse_args()
    if args.worker:
        cfg=yaml.safe_load((OUT/'config.yaml').read_text())
        if args.low_demand:cfg['roll']['fin_cant'] *= .25
        print(json.dumps(micro() if args.worker=='micro' else flight(cfg,args.worker,99 if args.low_demand else args.index)))
        return
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'config.yaml').write_text((ROOT/'Configs/flight_pressure_fed_regulator.yaml').read_text())
    cfg=yaml.safe_load((OUT/'config.yaml').read_text())
    generation=generate(cfg);(OUT/'generation.json').write_text(json.dumps(generation,indent=2));print('GENERATED',generation,flush=True)
    results=[]
    for index,order in enumerate([['current','forward','direct','shortcut'],['direct','shortcut','current','forward'],['forward','direct','shortcut','current']]):
        for mode in order:
            done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',mode,'--index',str(index)],cwd=ROOT,text=True,capture_output=True)
            if done.returncode:raise RuntimeError(done.stdout+done.stderr)
            row=json.loads(done.stdout);results.append(row)
            (OUT/'flights.json').write_text(json.dumps(results,indent=2));print('FLIGHT',json.dumps(row),flush=True)
    low_demand=[]
    for mode in ['current','forward','direct']:
        done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',mode,'--low-demand'],cwd=ROOT,text=True,capture_output=True)
        if done.returncode:raise RuntimeError(done.stdout+done.stderr)
        low_demand.append(json.loads(done.stdout))
    (OUT/'low_demand_flights.json').write_text(json.dumps(low_demand,indent=2))
    done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','micro'],cwd=ROOT,text=True,capture_output=True)
    if done.returncode:raise RuntimeError(done.stdout+done.stderr)
    result=json.loads(done.stdout);(OUT/'micro.json').write_text(json.dumps(result,indent=2));print('MICRO',json.dumps(result),flush=True)


if __name__=='__main__':main()
