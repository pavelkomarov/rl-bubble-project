import os

n_success = 0
n = 0
for f in os.listdir('.'):
    if f.startswith("plots"):
        n_success += os.path.exists(f + '/frame0010fig0.png')
        n += 1

print("num completed", n)
print("% completed", n/250)
print("since ~8:10pm Monday")
print("success rate", n_success/n)

