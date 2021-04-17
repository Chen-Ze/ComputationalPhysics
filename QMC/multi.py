from joblib import Parallel, delayed
def process(i):
    return i * i
    
print([delayed(process)(i) for i in range(6)])

results = Parallel(n_jobs=6)(delayed(process)(i) for i in range(6))
print(results)