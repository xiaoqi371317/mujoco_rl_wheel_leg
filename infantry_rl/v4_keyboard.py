"""Loopback-only browser keyboard controller, with a dead-man heartbeat."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Infantry v4 键盘控制</title><style>body{margin:0;background:#141b25;color:#fff;font:16px sans-serif}header{padding:12px}button{padding:8px}iframe{width:100%;height:85vh;border:0}</style>
<header><b>Infantry v4</b>　W/S 前后 · A/D 转向（松开归零） · F 高站姿开关 · R 自转开关 · 空格急停<br>
<button id="focus">点击这里启用键盘（操作画面后请再次点击）</button> <span id="state">待命</span></header>
<iframe id="viewer"></iframe><script>
const viewer=document.getElementById('viewer'),state=document.getElementById('state');
viewer.src=location.protocol+'//'+location.hostname+':8080';
const keys=new Set();let high=false,spin=false,active=false,busy=false;
document.getElementById('focus').onclick=()=>{active=true;window.focus();send()};
function clear(){keys.clear();spin=false;high=false;active=false;send()}
async function send(){if(busy)return;busy=true;try{const r=await fetch('/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:[...keys],high:active&&high,spin:active&&spin})});if(!r.ok)throw Error(r.status);state.textContent=(active?'键盘已启用':'已停止')+' | 高站姿 '+high+' | 自转 '+spin;}catch(e){state.textContent='连接中断，控制器将自动归零';}finally{busy=false}}
window.addEventListener('keydown',e=>{let k=e.key.toLowerCase();if(!['w','a','s','d','f','r',' '].includes(k))return;e.preventDefault();if(k===' '){clear();return}if(!active)return;if(!e.repeat){if(k==='f')high=!high;if(k==='r')spin=!spin;}if('wasd'.includes(k))keys.add(k);send()});
window.addEventListener('keyup',e=>{keys.delete(e.key.toLowerCase());send()});
window.addEventListener('blur',clear);document.addEventListener('visibilitychange',()=>{if(document.hidden)clear()});setInterval(send,100);
</script></html>'''

class KeyboardState:
    def __init__(self, command_mapper=None):
        self.lock = threading.Lock()
        self.command_mapper = command_mapper
        self.value = (0.,0.,.235)
        self.updated = 0.
    def update(self, payload):
        keys = set(payload.get('keys',[])) & set('wasd')
        spin = payload.get('spin') is True
        vx = (.3 if 'w' in keys else 0.) - (.2 if 's' in keys else 0.)
        wz = 1.2*(('a' in keys)-('d' in keys))
        if spin: vx,wz = 0.,3.
        h = .28 if payload.get('high') is True else .235
        if self.command_mapper is not None:
            vx,wz,h=self.command_mapper(vx,wz,h)
        with self.lock:
            self.value,self.updated = (vx,wz,h),time.monotonic()
    def read(self):
        with self.lock:
            return self.value if time.monotonic()-self.updated < .5 else (0.,0.,.235)

def start_keyboard_server(port=8081, command_mapper=None):
    state = KeyboardState(command_mapper)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/': self.send_error(404); return
            body=PAGE.encode('utf-8')
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            if self.path != '/keys': self.send_error(404); return
            # Reject cross-origin browser writes; there is no CORS allowance.
            origin=self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host',''):
                self.send_error(403); return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<1024: raise ValueError('size')
                state.update(json.loads(self.rfile.read(size)))
            except (ValueError,TypeError,AttributeError):
                self.send_error(400); return
            self.send_response(204); self.end_headers()
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return state,server
