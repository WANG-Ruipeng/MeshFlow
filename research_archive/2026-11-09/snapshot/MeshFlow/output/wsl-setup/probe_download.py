import json, time, urllib.request
url = 'https://pypi.org/pypi/nvidia-cublas-cu12/12.8.3.14/json'
with urllib.request.urlopen(url, timeout=20) as r:
    files = json.load(r)['urls']
wheel = next(f for f in files if 'x86_64' in f['filename'] and 'manylinux' in f['filename'])
request = urllib.request.Request(wheel['url'], headers={'Range': 'bytes=0-16777215'})
start = time.monotonic()
with urllib.request.urlopen(request, timeout=30) as r:
    data = r.read(16777216)
print('PyPI CUDA wheel probe: %.1f MiB/s' % (len(data)/1048576/(time.monotonic()-start)))