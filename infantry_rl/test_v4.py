"""Control contract tests: key release, modes, disconnect, and HTTP transport."""
import json
import time
import unittest
from urllib.request import Request, urlopen
from infantry_rl.v4_keyboard import KeyboardState, start_keyboard_server

class KeyboardTests(unittest.TestCase):
    def test_release_and_modes(self):
        s=KeyboardState()
        s.update({'keys':['w','a']})
        self.assertEqual(s.read(),(.3,1.2,.235))
        s.update({'keys':['w']})
        self.assertEqual(s.read(),(.3,0.,.235))
        s.update({'keys':['s','d'],'high':True})
        self.assertEqual(s.read(),(-.2,-1.2,.28))
        s.update({'keys':['w','a'],'spin':True})
        self.assertEqual(s.read(),(0.,3.,.235))
        s.update({'keys':[],'spin':False})
        self.assertEqual(s.read(),(0.,0.,.235))
        s.update({'keys':['a','d','w','s']})
        self.assertEqual(s.read()[1],0.)
    def test_disconnect(self):
        s=KeyboardState();s.update({'spin':True,'high':True})
        s.updated=time.monotonic()-1.
        self.assertEqual(s.read(),(0.,0.,.235))
    def test_http(self):
        state,server=start_keyboard_server(0)
        url='http://127.0.0.1:'+str(server.server_port)
        try:
            with urlopen(url) as r: self.assertIn(b'keyup',r.read())
            req=Request(url+'/keys',data=json.dumps({'keys':['d']}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req) as r:self.assertEqual(r.status,204)
            self.assertEqual(state.read(),(0.,-1.2,.235))
        finally:
            server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
