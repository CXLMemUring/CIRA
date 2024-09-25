import yaml
import subprocess
import sys
import re

def execute_external_program(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error occurred: {e}")

def parse_line(line):
    words = line.split()
    size = words[0]
    if size.endswith('GB'):
        size = size[:-2]

    if ':' not in words[-1]:
        return None

    f,ln = words[-1].split(':')
    return (f, float(size), int(ln))

allocregex = r"allocation_id = (\d+)"
def site_translation(fn, ps):
    if '.mlir' not in fn:
        return
    # missing in prof
    outputs = [{'type':'alloc', 'size': 0.01, 'value': 1987}]
    with open(fn) as f:
        lines = f.readlines()
        for [sz, ln] in ps:
            line = lines[ln-1].strip()

            # different elements
            if '@malloc' in line:
                m = re.search(allocregex, line)
                if m.group(1) == '291':
                    continue
                outputs.append({'type':'alloc', 'size':sz, 'value':int(m.group(1))})

    pfn = f"{fn}.prof.heap.yaml"
    print(f"{pfn} ")
    with open(pfn, 'w') as f:
        yaml.dump(outputs, f)


if len(sys.argv) < 3:
    sys.exit(f'usage: {sys.argv[0]} binary heapprofile')
command = f'pprof --text --lines {sys.argv[1]} {sys.argv[2]}'
res = execute_external_program(command)

# parse all
points = []
start = False
for line in res.split('\n'):
    if start == True:
        t = parse_line(line)
        if t is not None:
            points.append(t)
    if "cum%" in line:
        start = True

ds = {}
for [f, size, ln] in points:
    if f not in ds:
        ds[f] = []
    ds[f].append((size, ln),)

for fn in ds:
    site_translation(fn, ds[fn])

