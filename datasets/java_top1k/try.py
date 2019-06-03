import tqdm
import time

a = list(range(100))

iter = tqdm.tqdm(a, total=20)
for b in iter:
    time.sleep(0.2)
    if b>20:
        break