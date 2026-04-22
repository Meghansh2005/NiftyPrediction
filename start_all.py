"""
Start all SignalX servers + open dashboards in browser.
Run: python start_all.py
"""
import subprocess, sys, time, webbrowser, os

PORT_MAIN    = 8001
PORT_SIMPLE  = 8002
PORT_SCREENER= 8003
PORT_STATIC  = 8080   # serves HTML files

procs = []

def start(cmd, name):
    print(f"Starting {name}...")
    p = subprocess.Popen(cmd, shell=True)
    procs.append(p)
    return p

start(f"{sys.executable} -m uvicorn market_signal_engine:app --port {PORT_MAIN}", "Main engine (8001)")
time.sleep(2)
start(f"{sys.executable} -m uvicorn simple_signal_engine:app --port {PORT_SIMPLE}", "Simple signal (8002)")
start(f"{sys.executable} -m uvicorn stock_screener:app --port {PORT_SCREENER}", "Stock screener (8003)")
start(f"{sys.executable} -m http.server {PORT_STATIC}", "Static file server (8080)")

print("\nWaiting for servers to start...")
time.sleep(5)

print("Opening dashboards...")
webbrowser.open(f"http://localhost:{PORT_STATIC}/chart_dashboard.html")
time.sleep(1)
webbrowser.open(f"http://localhost:{PORT_STATIC}/stock_dashboard.html")

print("\nAll servers running. Press Ctrl+C to stop all.")
try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    print("\nStopping all servers...")
    for p in procs:
        p.terminate()
