from scipy.optimize import minimize, rosen, rosen_der

def f(x):
    print(f'called with {x}')
    return (x-3)**2 + 1

x0 = [5]
res = minimize(f, x0, method='Nelder-Mead', tol=1e-6)
print(res.x)
print(res.fun)