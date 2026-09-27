"""Loopback-only CONNECT relay restricted to DeepSeek, for an SSH-forwarded job.

TLS stays end-to-end between the API client and api.deepseek.com. This relay
does not decrypt, log, or persist requests, credentials, or responses.
"""
import argparse
import http.server
import select
import socket

ap=argparse.ArgumentParser()
ap.add_argument('--port',type=int,default=18119)
args=ap.parse_args()

class Relay(http.server.BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def do_CONNECT(self):
        if self.path!='api.deepseek.com:443':
            self.send_error(403,'Only the configured API target is permitted')
            return
        try:
            upstream=socket.create_connection(('api.deepseek.com',443),timeout=15)
        except OSError:
            self.send_error(502,'API connection unavailable')
            return
        with upstream:
            self.send_response(200,'Connection Established');self.end_headers();self.wfile.flush()
            self.connection.settimeout(60);upstream.settimeout(60)
            try:
                while True:
                    readable,_,_=select.select([self.connection,upstream],[],[],60)
                    if not readable:break
                    for source in readable:
                        data=source.recv(65536)
                        if not data:return
                        (upstream if source is self.connection else self.connection).sendall(data)
            except OSError:
                pass
            finally:
                self.close_connection=True
    def log_message(self,format,*args):
        pass

server=http.server.ThreadingHTTPServer(('127.0.0.1',args.port),Relay)
print(f'Loopback API CONNECT relay on 127.0.0.1:{server.server_port}; TLS is not terminated',flush=True)
server.serve_forever()
