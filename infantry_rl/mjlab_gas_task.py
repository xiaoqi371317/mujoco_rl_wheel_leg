"""Experimental constant-force gas spring; six policy outputs are unchanged."""
from dataclasses import dataclass
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.managers import RewardTermCfg
from infantry_rl import mjlab_task as base,mjlab_v6_refine_task as refine
from infantry_rl.mjlab_v5_task import V5LegAction,V5LegActionCfg

def gas_spec():
    # Only the spec loader uses the variant. The unchanged kinematics retain the
    # validated fix pose bank; no source XML or checkpoint hash is rewritten.
    original=base.MODEL
    try:
        base.MODEL=original.with_name('training_scene_fix_gas_trial.xml')
        return base.robot_spec()
    finally:base.MODEL=original

class GasLegAction(V5LegAction):
    def apply_actions(self):
        super().apply_actions()
        if not hasattr(self,'gas_ids'):
            self.gas_ids=self._entity.find_tendons(('left_gas_tendon','right_gas_tendon'),preserve_order=True)[0]
        force=getattr(self._env,'_gas_force',None)
        if force is None:force=torch.zeros(self._env.num_envs,device=self.device)
        ramp=getattr(self._env,'_gas_ramp_steps',0)
        if ramp:force=force*min(1.,max(0,self._env.common_step_counter-getattr(self._env,'_motion_start_step',0))/ramp)
        self._entity.set_tendon_effort_target(force[:,None].expand(-1,2),tendon_ids=self.gas_ids)

@dataclass(kw_only=True)
class GasLegActionCfg(V5LegActionCfg):
    def build(self,env):return GasLegAction(self,env)

def motor_effort(env):
    # Tendon forces are measured in N; do not count them as 30 Nm motors.
    return (env.scene['robot'].data.actuator_force[:,:6]/30.).square().mean(1)

def make_cfg(num_envs=1024,play=False,seed=43,stage='tuck'):
    cfg=refine.make_cfg(num_envs,play=play,seed=seed,stage=stage)
    robot=cfg.scene.entities['robot'];robot.spec_fn=gas_spec
    robot.articulation.actuators=(*robot.articulation.actuators,XmlActuatorCfg(target_names_expr=('left_gas_tendon','right_gas_tendon'),transmission_type='tendon',command_field='effort'))
    cfg.actions['legs']=GasLegActionCfg(**vars(cfg.actions['legs']))
    cfg.rewards['effort']=RewardTermCfg(func=motor_effort,weight=cfg.rewards['effort'].weight)
    return cfg

local_height=refine.local_height
wheel_clearance=refine.wheel_clearance
stance_errors=refine.stance_errors
