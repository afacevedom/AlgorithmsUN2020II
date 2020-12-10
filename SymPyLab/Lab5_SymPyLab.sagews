︠13a7c610-6353-4c3e-a1d9-836e49e8e62a︠
%md
Algoritmos Grupo 1 9am a 11am - Grupo de estudiantes 17

Daniel Armando Zuñiga Espinosa

Diego Alejandro Alvarado Chaparro

Andres Felipe Acevedo Monroy
︡a4445633-6d76-4ef8-b3b7-0020dbaed973︡{"done":true,"md":"Algoritmos Grupo 1 9am a 11am - Grupo de estudiantes 17\n\nDaniel Armando Zuñiga Espinosa\n\nDiego Alejandro Alvarado Chaparro\n\nAndres Felipe Acevedo Monroy"}
︠e57c0a52-a904-40fd-9a2a-26822e4ed57fs︠
from IPython.display import Image
import sympy
import math
from sympy import *
from sympy import Symbol
from sympy import div
from sympy import integrate
from sympy import sin
from sympy import Symbol
from sympy import cos
from sympy import sin
import numpy
import matplotlib.pyplot as plt
︡b637d2ec-eb52-408f-8a41-72f92e889961︡{"done":true}
︠a6d66969-6d1f-4dab-b357-37d829b89a1a︠
%md
SymPy’s polynomials
︡47354ed6-12bb-4105-8318-4f04e146e0e0︡{"done":true,"md":"SymPy’s polynomials"}
︠916dd969-6d02-41dc-ae22-3f67afa9a8f8s︠
Image('OurExample1_1.jpg', width=500)
︡c47814da-1277-468b-806f-7c39fba10f12︡{"stdout":"<IPython.core.display.Image object>\n"}︡{"done":true}
︠769a1216-88d0-48f3-949c-cd7e188f0a63s︠
Image('OurExample1_2.jpg', width=500)
︡6c96146d-b970-4420-9cf3-43604683cbbd︡{"stdout":"<IPython.core.display.Image object>\n"}︡{"done":true}
︠5f89ce58-ab0b-4d13-8e09-890348afc6d4s︠
x = Symbol('x')

p = (x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10)

print(p.as_poly())
print(" ")

for i in range(11):
  p,r = div(p, x-(i+1))
  print(p)
  print(r)
︡fda3eda4-525f-4120-84fd-299ba052ab9e︡{"stdout":"Poly(x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800, x, domain='ZZ')\n"}︡{"stdout":" \n"}︡{"stdout":"x**9 - 54*x**8 + 1266*x**7 - 16884*x**6 + 140889*x**5 - 761166*x**4 + 2655764*x**3 - 5753736*x**2 + 6999840*x - 3628800\n0\nx**8 - 52*x**7 + 1162*x**6 - 14560*x**5 + 111769*x**4 - 537628*x**3 + 1580508*x**2 - 2592720*x + 1814400\n0\nx**7 - 49*x**6 + 1015*x**5 - 11515*x**4 + 77224*x**3 - 305956*x**2 + 662640*x - 604800\n0\nx**6 - 45*x**5 + 835*x**4 - 8175*x**3 + 44524*x**2 - 127860*x + 151200\n0\nx**5 - 40*x**4 + 635*x**3 - 5000*x**2 + 19524*x - 30240\n0\nx**4 - 34*x**3 + 431*x**2 - 2414*x + 5040\n0\nx**3 - 27*x**2 + 242*x - 720\n0\nx**2 - 19*x + 90\n0\nx - 10\n0\n1\n0\n0\n1\n"}︡{"done":true}
︠03db3149-c697-4879-b7f7-4fcb83c500a6︠
%md
SymPy’s polynomial simple univariate polynomial factorization
︡603c9c61-44b7-489a-a5b9-95f9fc882c8a︡{"done":true,"md":"SymPy’s polynomial simple univariate polynomial factorization"}
︠0ff2bc62-604b-49d2-910c-2599808ddde6s︠
Image('OurExample2.jpg', width=500)
︡f6de644a-8270-4a3d-aa82-63377ca44788︡{"stdout":"<IPython.core.display.Image object>\n"}︡{"done":true}
︠9062cb37-e328-4bb8-b2b0-cc5ba14484c3s︠
x = Symbol('x')
factor(x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800)
︡4d366364-e90e-49ba-a669-463068742eb9︡{"stdout":"(x - 10)*(x - 9)*(x - 8)*(x - 7)*(x - 6)*(x - 5)*(x - 4)*(x - 3)*(x - 2)*(x - 1)\n"}︡{"done":true}
︠144e817c-f3bc-475a-aeca-1a5b18b038e8︠
%md
SymPy’s solvers
︡5354a38c-0e60-4dc3-bf6e-3378d003c60d︡{"done":true,"md":"SymPy’s solvers"}
︠d00aad96-fae9-4ca2-b661-8c86d5838756s︠
Image('OurExample3.png', width=500)
︡35938853-8f3e-4c82-807e-003249b6a95b︡{"stdout":"<IPython.core.display.Image object>\n"}︡{"done":true}
︠b5f78bba-493d-42aa-871b-c1b927f201d1s︠
x = Symbol('x')
solveset(Eq(x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800, 0), x)
︡4864c21a-1ee8-411b-a385-e574d766048e︡{"stdout":"FiniteSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)"}︡{"stdout":"\n"}︡{"done":true}
︠5213be46-3810-416b-aaec-0a4803df8740︠
%md
SymPy’s Symbolic and Numercical Complex Evaluations
︡00ca8dd4-2756-4ae2-8a4c-e8f709b6b3fe︡{"done":true,"md":"SymPy’s Symbolic and Numercical Complex Evaluations"}
︠d694f13c-fc2f-46ac-80f0-85ab2e23a638s︠
Image('OurExample4.png', width=500)
︡e3cdc661-9879-4fb7-98b4-e4c1f1ea7d4a︡{"stdout":"<IPython.core.display.Image object>\n"}︡{"done":true}
︠9a8f664b-ffc9-43fd-b2c4-65a832f55b6cs︠
u1, u2, w1, w2, v1, v2 = symbols("u1 u2 w1 w2 v1 v2", real=True)  
u = u1 + I*u2
w = w1 + I*w2
v = v1 + I*v2

print(u*w*v)
print(expand(u*w*v))
print(expand((u*w)*v))
print(expand(u*(w*v)))

w=N(1/(pi + I), 20)
print('w=',w)
︡01a6461d-0614-4648-b9ce-d670d47db40b︡{"stdout":"(u1 + I*u2)*(v1 + I*v2)*(w1 + I*w2)\n"}︡{"stdout":"u1*v1*w1 + I*u1*v1*w2 + I*u1*v2*w1 - u1*v2*w2 + I*u2*v1*w1 - u2*v1*w2 - u2*v2*w1 - I*u2*v2*w2\n"}︡{"stdout":"u1*v1*w1 + I*u1*v1*w2 + I*u1*v2*w1 - u1*v2*w2 + I*u2*v1*w1 - u2*v1*w2 - u2*v2*w1 - I*u2*v2*w2\n"}︡{"stdout":"u1*v1*w1 + I*u1*v1*w2 + I*u1*v2*w1 - u1*v2*w2 + I*u2*v1*w1 - u2*v1*w2 - u2*v2*w1 - I*u2*v2*w2\n"}︡{"stdout":"w= 0.28902548222223624241 - 0.091999668350375232456*I\n"}︡{"done":true}
︠3dd1c77d-b63d-44e4-8b98-03d7956327f3︠
%md
SymPy’s integrals
︡0ba76296-fe76-4169-a696-fc51fd77c6b5︡{"done":true,"md":"SymPy’s integrals"}
︠9cb9254b-c28e-4f8a-ae59-a2eefe38cc28s︠
dictt = {}  
x = sympy.Symbol("x")
i = sympy.integrate(sin(x**2))
print(i)
dictt["analytical"] = float(i.subs(x, 1) - i.subs(x, 0))
print("Analytical result: {}".format(dictt["analytical"]))
︡9d6fa7f1-ef94-4b0e-ae1d-167fe5f17ac6︡{"stdout":"3*sqrt(2)*sqrt(pi)*fresnels(sqrt(2)*x/sqrt(pi))*gamma(3/4)/(8*gamma(7/4))\n"}︡{"stdout":"Analytical result: 0.3102683017233811\n"}︡{"done":true}
︠58223a21-02ed-4b4c-a368-fd32a179fbd1s︠
N = 100000
accum = 0
for i in range(N):
    x = numpy.random.uniform(0, 1)
    accum += sin(x**2)
volume = 1 - 0
dictt["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(dictt["MC"]))
︡b9378f54-2c7a-44b4-b662-68bd5e7422a6︡{"stdout":"Standard Monte Carlo result: 0.310170892844226\n"}︡{"done":true}
︠c0b3bbfa-3f88-4b19-9cd1-d597463762bbs︠
Image('OurIntegral11.png', width=500)
︡b258dad5-4087-4669-8f67-b80eeb6fbdb1︡{"stdout":"<IPython.core.display.Image object>\n"}︡{"done":true}
︠865fd880-962c-4b6d-a59d-26e9c0bb23dcs︠
x = Symbol("x")
i = integrate(sin(x**2))
print(i)
print(float(i.subs(x, 1) - i.subs(x, 0)))
︡87eca164-f336-4272-87b2-4af031f7224a︡{"stdout":"3*sqrt(2)*sqrt(pi)*fresnels(sqrt(2)*x/sqrt(pi))*gamma(3/4)/(8*gamma(7/4))\n"}︡{"stdout":"0.3102683017233811\n"}︡{"done":true}
︠132a1ae2-ea31-4b1f-8cbc-cf235a2f30e7s︠
N = 100000
accum = 0
l =[]
result = {}
for i in range(N):
    x = numpy.random.uniform(0, 1)
    accum += sin(x**2)
volume = 1-0
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡fe7471ea-815a-4557-b491-925ca9c01839︡{"stdout":"Standard Monte Carlo result: 0.310249023169644\n"}︡{"done":true}
︠f92e2682-2c58-4c25-b9b5-ca05eed8e0d5︠
%md
A higher dimensional integral
︡1c0be37b-4caf-4303-ac2a-c3b70356a375︡{"done":true,"md":"A higher dimensional integral"}
︠9c2fd17d-d9b7-4bb3-82d9-14bdbaf42f9fs︠
x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")
expr = sin(x**2) + sin(y**2) + sin(z**2)
res = sympy.integrate(expr,
                      (x, 3,sympy.pi),
                      (y, -3*sympy.pi, sympy.pi),
                      (z, -5*sympy.pi, sympy.pi))
# Note: we use float(res) to convert res from symbolic form to floating point form
dictt = {} 
dictt["analytical"] = float(res)
print("Analytical result: {}".format(dictt["analytical"]))
︡5b7de9ee-0295-470a-9bf4-1bd9e91d706b︡{"stdout":"Analytical result: 5.922924432045176\n"}︡{"done":true}
︠aef78f50-25a0-4fe3-a993-945b973952a4s︠
N = 100000
accum = 0
for i in range(N):
    x = numpy.random.uniform(3, numpy.pi)
    y = numpy.random.uniform(-3*numpy.pi, numpy.pi)
    z = numpy.random.uniform(-5*numpy.pi, numpy.pi)
    accum += sin(x**2) + sin(y**2) + sin(z**2)
volume = (numpy.pi-3)*(numpy.pi-(-3*numpy.pi))*(numpy.pi-(-5*numpy.pi))
result = {} 
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡1cad2c82-b3c1-436b-9e1f-7418069a73be︡{"stdout":"Standard Monte Carlo result: 5.77938667177771\n"}︡{"done":true}
︠508696b6-b3a4-4d2a-925d-f6b22ccae84fs︠
# adapted from https://mail.scipy.org/pipermail/scipy-user/2013-June/034744.html
def halton(dim: int, nbpts: int):
    h = numpy.full(nbpts * dim, numpy.nan)
    p = numpy.full(nbpts, numpy.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)
︡d8f27b73-8c8c-4560-b307-d928b30c346f︡{"done":true}
︠c45ed27d-a2fc-4c1f-ba84-c6599dd85ef1s︠
N = 1000
seq = halton(6, N)
plt.title("2D Halton sequence")
plt.axes().set_aspect('equal')
plt.scatter(seq[:,0], seq[:,1], marker=".", alpha=0.5);
︡40c80df5-a975-400e-8087-c38b3f90172a︡{"stdout":"Text(0.5, 1.0, '2D Halton sequence')\n"}︡{"stderr":":1: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.\n"}︡{"stdout":"<matplotlib.collections.PathCollection object at 0x7fae2e6b5880>\n"}︡{"done":true}
︠4acc6bee-6179-49f5-bd85-e161a12c1b9es︠
N = 100000

seq = halton(3, N)
accum = 0
for i in range(N):
    x = -numpy.pi + seq[i][0] * numpy.pi * 2
    y = -numpy.pi + seq[i][1] * numpy.pi * 2
    z = -numpy.pi + seq[i][2] * numpy.pi * 2
    accum += sin(x**2) + sin(y**2) + sin(z**2)
volume = (2 * numpy.pi)**3
result = {} 
result["MC"] = volume * accum / float(N)
print("Qausi Monte Carlo Halton Sequence result: {}".format(result["MC"]))
︡b1a752a9-0f26-4746-b1b2-c3dc2b9678c7︡{"stdout":"Qausi Monte Carlo Halton Sequence result: 183.022620530781\n"}︡{"done":true}









