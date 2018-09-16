"""
Create a monge patch x(u,v) = (u, v, f(u,v))))
Run some numerical sanity checks.
Plot some results.
"""

from sympy import symbols, sqrt, pprint
from sympy import Matrix, Eq
from sympy.matrices import randMatrix

# uv parameter in the domain
u,v = symbols('u v', real=True)
# choose a point in uv space to analyse
uvpt = ((u,1), (v,1)) # example: [1,1]
# choose a height function to set z values
#f = u**2+2*v**2 # an example
f = u*v # another example

# xyz coords in ambient output as a monge patch
# Use uppercase X used for coordinates in xyz ambient space
X = Matrix([u, v, f])

# Calculate df
Xu = X.diff(u)
Xv = X.diff(v)

# Calculate Jacobian
XJ = Xu.row_join(Xv)

# Calculate E, F, G, W
XE = Xu.dot(Xu)
XF = Xu.dot(Xv)
XG = Xv.dot(Xv)

# Calculate induced metric
XI = Matrix([[XE, XF], [XF, XG]])
# XW = sqrt(det([XI])) = rate of area stretching
XW = sqrt(XE*XG - XF**2)

# Verify that J'J = I = [[EF],[FG]]
XI2 = XJ.transpose() * XJ # since I = J'J
print "Verify: J'J = I = [[EF],[FG]]"
pprint(Eq(XI, XI2, evaluate=False))
print '\n'

# Calculate unit normal vector
XN = Xu.cross(Xv) / Xu.cross(Xv).norm()
# Covariant derivatives of the normal vector
XNu = XN.diff(u)
XNv = XN.diff(v)
# Weingarten map dN maps tangent vectors in uv space 
# to tangent vectors in ambient space
XdN = XNu.row_join(XNv)

# Calculate L, M, N
Xl = XNu.dot(Xu)
Xm = XNu.dot(Xv)
Xn = XNv.dot(Xv)

# Find an orthonormal basis in tangent space
# Use |Xu| as E1
XE1 = Xu / Xu.norm()
# Use unit normal vector XN as XE3
XE3 = XN
# Then E2 just needs to be orthogonal to both E1 and N
XE2 = XE1.cross(XE3)

# Express Xu and Xv in tangent space w.r.t. E1 and E2
Xu_E1 = Xu.dot(XE1)
Xu_E2 = Xu.dot(XE2)
Xv_E1 = Xv.dot(XE1)
Xv_E2 = Xv.dot(XE2)
# Use small letters to express vectors in tangent space
xu = Matrix([Xu_E1, Xu_E2])
xv = Matrix([Xv_E1, Xv_E2])

# Express XNu and XNv in tangent space w.r.t. E1 and E2
# We know that XNu and XNv are in the tangent plane so they can be
# expressed as a linear combination of E1 and E2
XNu_E1 = XNu.dot(XE1)
XNu_E2 = XNu.dot(XE2)
XNv_E1 = XNv.dot(XE1)
XNv_E2 = XNv.dot(XE2)
# Use small letters to express vectors in tangent space
nu = Matrix([XNu_E1, XNu_E2])
nv = Matrix([XNv_E1, XNv_E2])

# Find shape operator S w.r.t. E1 and E2
# S * [ xu  xv ] = [ nu nv ]
#     [ xu  xv ]   [ nu nv ]
# xJ = [xu xv]
#      [xu xv]
# xJ is like XJ but stretches vectors into the E1,E2 space rather than xyz space
xJ = xu.row_join(xv)
nunv = nu.row_join(nv)
S = nunv * xJ.inv()

# Evaluate things at a specific point only because it will take
# way too long to calculate the eigenvectors in general
S_pt = S.subs(uvpt)
XdN_pt = XdN.subs(uvpt)
XJ_pt = XJ.subs(uvpt)
xJ_pt = xJ.subs(uvpt)
XI_pt = XI.subs(uvpt)

# Numerically check shape operator using O'Neill V.4.2 Lemma
xu_pt = xu.subs(uvpt)
xv_pt = xv.subs(uvpt)
# l = S(Xu).dot(Xu)
print 'Verify: L = S(xu).xu: {} = {}'.format(Xl.subs(uvpt).evalf(),
                                             (S_pt*xu_pt).dot(xu_pt).evalf())
# m = S(Xu).dot(Xv)
print 'Verify: M = S(xu).xv: {} = {}'.format(Xm.subs(uvpt).evalf(),
                                             (S_pt*xu_pt).dot(xv_pt).evalf())
# n = S(Xv).dot(Xv)
print 'Verify: N = S(xv).xv: {} = {} \n'.format(Xn.subs(uvpt).evalf(),
                                                (S_pt*xv_pt).dot(xv_pt).evalf())

# Find principal curvatures and directions
# These are w.r.t. the tangent space with basis E1 and E2
eigenvects = S_pt.eigenvects()
k1_pt = eigenvects[0][0]
k2_pt = eigenvects[1][0]
v1_pt = eigenvects[0][2][0]
v2_pt = eigenvects[1][2][0]

# Find principal directions w.r.t. basis xu and xv
# These are also the principal directions in the uv domain
xv1_pt = xJ_pt.inv() * v1_pt
xv2_pt = xJ_pt.inv() * v2_pt
# Find principal directions in ambient space
XV1_pt = XJ_pt * xv1_pt
XV2_pt = XJ_pt * xv2_pt

# Verify that principal directions are orthogonal w.r.t. the metric
# i.e. g(v1, v2) = df(v1).dot(df(v2)) = 0
print 'Verify that principal vectors are orthogonal i.e. g(v1, v2) = 0: '
print 'g(v1, v2) = ', (xv1_pt.transpose() * XI_pt * xv2_pt).evalf(), '\n'

# Verify DDG 2.4.1 df(SX) = dN(X)
# Test with eigenvector1
dNv1 = (XdN_pt * xv1_pt).evalf() # dN(v1)
# Sv1 = k1v1, so df(Sv1) = k1.df(v1)
dfSv1 = (k1_pt * XV1_pt).evalf()
print 'Verify that df(SX) = dN(X)'
print 'df(Sv1) = dN(v1)'
pprint(Eq(dfSv1, dNv1, evaluate=False))
# Test with eigenvector2
dNv2 = (XdN_pt * xv2_pt).evalf() # dN(v2)
# Sv2 = k2v2, so df(Sv2) = k1.df(v2)
dfSv2 = (k2_pt * XV2_pt).evalf()
print 'df(Sv2) = dN(v2)'
pprint(Eq(dfSv2, dNv2, evaluate=False))
print ('Since df(SX) = dN(X) is true for v1 and v2, it is also true '
       'for all linear combinations of v1 and v2 \n')

# Find the induced metric w.r.t. the E1 E2 basis
xE = xu.dot(xu)
xF = xu.dot(xv)
xG = xv.dot(xv)
xI = Matrix([[xE, xF], [xF, xG]])
xI_pt = xI.subs(uvpt) # Same as XI_pt!?

# Find the shape operator in uv space
# Any vector in uv space can be transformed into {E1, E2} space,
# multipled by the shape operator, then transformed back into uv space
S_pt_uv = (xJ_pt.inv() * S_pt * xJ_pt).evalf()

# S is self-adjoint w.r.t. the induced metric XI
print 'Verify that S is self-adjoint w.r.t. the induced metrix I'
print 'i.e. g(SX,Y) = g(X,SY) for all vectors X,Y'
randX = randMatrix(2,1)
randY = randMatrix(2,1)
gSXY = (S_pt_uv * randX).transpose() * xI_pt * randY
gXSY = randX.transpose() * xI_pt * (S_pt_uv * randY)
print '{} = g(SX,Y) = g(X,SY) = {}'.format(gSXY, gXSY)

# ============
# Plot results
# ============
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Set limits
xMin = -2
xMax = 2
yMin = -2
yMax = 2
stepSize = 0.1

# Make data
X = np.arange(xMin, xMax, stepSize)
Y = np.arange(yMin, yMax, stepSize)
Z = list()
for xy in itertools.product(X,Y):
    Z.append(float(f.subs(((u, xy[0]), (v, xy[1]))).evalf()))
zMin = min(Z)
zMax = max(Z)
# Format data for plotting
X, Y = np.meshgrid(X, Y)
Z = np.array(Z).reshape(len(X), len(Y))

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(xMin, xMax)
ax.set_ylim3d(yMin, yMax)
ax.set_zlim3d(zMin, zMax)
ax.set_xlabel('x')
ax.set_ylabel('y')
surf = ax.plot_surface(X, Y, Z)

# Draw principle directions at uvpt
# Would be nice to draw it on the some graph, but the surface would overlap the arrows
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_xlim3d(xMin, xMax)
ax2.set_ylim3d(yMin, yMax)
ax2.set_zlim3d(zMin, zMax)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# Scale principal vector length to represent curvature
ax2.quiver(uvpt[0][1], uvpt[1][1], float(f.subs(uvpt)), 
           *XV1_pt/XV1_pt.norm(), color='g', arrow_length_ratio=0)
ax2.quiver(uvpt[0][1], uvpt[1][1], float(f.subs(uvpt)),
           *XV2_pt/XV2_pt.norm(), color='r', arrow_length_ratio=0)
