def f(x):
    ret = x
    for _ in range(1, 4):
        ret = 4 * ret * (1 - ret)
    return ret


# Symbolic
import sympy  # noqa: E402

symbolic_x = sympy.symbols("x")
symbolic_f = f(symbolic_x)
symbolic_fprime = symbolic_f.diff(symbolic_x)
print(symbolic_fprime)
print(f"f'(x=2): {symbolic_fprime.subs(symbolic_x, 2)}")


# Nuermic
def df_numeric(x, delta=1e-8):
    return (f(x + delta) - f(x)) / delta


print(f"f'(x=2): {df_numeric(x=2)}")

# Automatic
import jax  # noqa: E402

print(f"f'(x=2): {jax.grad(f)(2.0)}")
print(f"f''(x=2): {jax.grad(jax.grad(f))(2.0)}")
print(f"f'''(x=2): {jax.grad(jax.grad(jax.grad(f)))(2.0)}")
