"""Check user direction, mode exclusion and fast curriculum contracts."""
import unittest
from types import SimpleNamespace
import torch
from infantry_rl.fast_control import resolve_command, map_legacy_keyboard
from infantry_rl.v4_keyboard import KeyboardState
from infantry_rl.mjlab_cmdvel_v4_fast_task import FastCommand, HEIGHT_RATE


class FastTests(unittest.TestCase):
    def test_directions_and_modes(self):
        self.assertEqual(resolve_command(1.3),(-1.3,0.,.235))
        self.assertEqual(resolve_command(-.5),(.5,0.,.235))
        self.assertEqual(resolve_command(high=True),(0.,0.,.28))
        self.assertEqual(resolve_command(1.3,high=True),(-1.3,0.,.235))
        self.assertEqual(resolve_command(high=True,spin=True),(0.,3.,.235))
        s=KeyboardState(map_legacy_keyboard)
        s.update({'keys':['w','a']});self.assertEqual(s.read(),(-1.3,1.2,.235))
        s.update({'keys':['w']});self.assertEqual(s.read(),(-1.3,0.,.235))
        s.update({'keys':['s']});self.assertEqual(s.read(),(.5,0.,.235))
        s.update({'keys':[],'high':True});self.assertEqual(s.read(),(0.,0.,.28))

    def test_sampling(self):
        n=10000
        # Exercise actual sampler against lightweight state without GPU allocation.
        for steps,limit in [(0,.3),(28800,1.3)]:
            f=SimpleNamespace(device='cpu',cfg=SimpleNamespace(full_range=False),
                _env=SimpleNamespace(common_step_counter=steps),vel_command_b=torch.zeros(n,3),
                is_standing_env=torch.zeros(n,dtype=torch.bool),height_target=torch.zeros(n),
                height_command=torch.zeros(n),command_counter=torch.zeros(n,dtype=torch.long))
            FastCommand._resample_command(f,torch.arange(n))
            high=f.height_target>.25
            self.assertTrue((f.vel_command_b[high]==0).all())
            self.assertAlmostEqual(float(f.vel_command_b[:,0].min()),-limit,places=6)
            self.assertLessEqual(float(f.vel_command_b[:,0].max()),.5)
            self.assertEqual(float(f.vel_command_b[:,2].abs().max()),3.)
            self.assertGreater(int(high.sum()),1000)
        self.assertAlmostEqual(.045/HEIGHT_RATE,.375)


if __name__=='__main__':unittest.main()
