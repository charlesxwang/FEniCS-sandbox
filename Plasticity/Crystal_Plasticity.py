"""
Crystal Plasticity
Charles Wang
Clemson University
"""

from dolfin import *


import os, sys
from math import pi, cosh
#from math import pi

from numpy import array
from numpy import linalg as lin
from datetime import datetime, time

label = str(sys.argv[0]) + str(datetime.now()) #+ str(r_) + str(var_) 

#scaling constanrs
R1_ = 2e-10
R2_ = 1.0  #float(sys.argv[1])
R3_ = 1.0  #float(sys.argv[1])

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 8

ffc_options = {"optimize": True, \
			   "eliminate_zeros": True, \
			   "quadrature_degree": 8  ,\
			   "precompute_basis_const": True, \
			   "precompute_ip_const": True
			   }
#set_log_level(ERROR)
#set_log_level(PROGRESS)
#set_log_level(DEBUG)

#############
### domain ##
#############

height = 1.0
length = 1.0

#timestep
dt = 1e-3

n = 4
#mesh = RectangleMesh(0, 0, 1, 1, n, n, 'crossed') # old syntax
mesh = RectangleMesh(Point(0, 0), Point(1, 1), n, n, 'crossed') # new syntax
#mesh = Mesh("rectangle2.xml")

#plot(mesh)
#interactive()
#exit()
'''
# Refine the mesh
cell_makers = CellFunction("bool", mesh)
cell_makers.set_all(False)
a = 1.
for cell in cells(mesh):
	p = cell.midpoint()
	if 	p.x() < a/n or p.x() > length - a/n or p.y() < a/n or p.y() > height - a/n :
		cell_makers[cell] = True

mesh = refine(mesh, cell_makers)
'''

print(mesh.geometry)
#plot(mesh,interactive=True)
print "before exit"
#exit()
print "after exit"


#Remark
# Move vertices in boundary
#A=5e-6
#for x in mesh.coordinates():
#	if  x[0] == length:
#		x[0] = length + A*cos(pi*x[1]/L0) 


#plot(mesh, interactive=True)

#################
### parameters ##
#################

# Material parameters Aluminium 
# rho = 3000 kg/m3
# E = 70 GPa
# Sigma = 70 MPa

#Y_ = 70 GPa / 70 MPa
Y_  = 1000
nu_ = 0.3

mu_	= Y_/(2*(1 + nu_))
lmbda_ =  Y_*nu_/( (1 + nu_)*(1 - 2*nu_) )

phi = 0.79

I = Identity(2)    # Identity tensor

######################
### function spaces ##
######################

P = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
S = FunctionSpace(mesh, "DG", 1)

TAU   = FunctionSpace(mesh, "DG", 1)
NU   = FunctionSpace(mesh, "DG", 1)
ALPHA = FunctionSpace(mesh, "CG", 1)


Pe = FiniteElement( "CG", mesh.ufl_cell(), 1)
Ve = VectorElement( "CG", mesh.ufl_cell(), 1)
Se = FiniteElement( "DG", mesh.ufl_cell(), 1)

TAUe   = FiniteElement( "DG", mesh.ufl_cell(), 1)
NUe   = FiniteElement( "DG", mesh.ufl_cell(), 1)
ALPHAe = FiniteElement( "CG", mesh.ufl_cell(), 1)


#W = FunctionSpace(mesh,Ve*Pe*Se*Se*Se*NUe*NUe*ALPHAe)
#W = MixedFunctionSpace([V,P,S,S,S,NU,NU,ALPHA])
We = MixedElement(Ve,Pe,Se,Se,Se,NUe,NUe,ALPHAe)
W = FunctionSpace(mesh, We)
w = Function(W)



(v,p,s11,s12,s22,nu1,nu2,alpha)=(as_vector((w[0], w[1])), w[2],w[3],w[4],w[5],w[6],w[7],w[8])
(vt,pt,s11t,s12t,s22t,nu1t,nu2t,alphat) =  TestFunctions(W)


# initialization of the previous step
vp = Function(V)
s11p = Function(S)
s12p = Function(S)
s22p = Function(S)
tauc1p = project(1.0, TAU)
tauc2p = project(1.0, TAU)

sp = as_matrix([[s11p,s12p],[s12p,s22p]])
s  = as_matrix([[s11,s12],[s12,s22]])
st = as_matrix([[s11t,s12t],[s12t,s22t]])

#alphap = project(0.417, ALPHA)
alphap = project(0.417, ALPHA) #Charles
#no initial angle
#alphap = project(0.0, ALPHA)


hard = Function(TAU)
nutot = Function(NU)
nu1int = Function(NU)
nu2int = Function(NU)


s1 = as_vector(( cos(alpha+phi), sin(alpha+phi))) 
m1 = as_vector((-sin(alpha+phi), cos(alpha+phi)))
s2 = as_vector(( cos(alpha-phi), sin(alpha-phi)))
m2 = as_vector((-sin(alpha-phi), cos(alpha-phi)))

SM1 = outer(s1,m1)
P1 = sym(SM1)
Q1 = skew(SM1) 

SM2 = outer(s2,m2)
P2 = sym(SM2)
Q2 = skew(SM2) 

eps = 1e-5

def abso(x):
	return sqrt( x**2 +eps**2)

def maxi(x):
	return (sqrt( x**2 +eps**2)+x)/2

def signe(x):
	return x/sqrt( x**2 +eps**2)

def fhardening(x):
	try:	
		return 1.0/( cosh(11.125*x)*cosh(11.125*x) )
	except OverflowError:
		return 0.0
'''
	if x < 40:
		return 1.0/( cosh(11.125*x)*cosh(11.125*x) )
	elif:
		return 0.0
'''

##########################
### boundary conditions ##
##########################

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return ( on_boundary and near(x[1],height) )

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return ( on_boundary and near(x[1],0.0) )

v0 = Constant("-1.0")
bcs = list()

bcs.append( DirichletBC(W.sub(0).sub(1), v0, Top() ) )
bcs.append( DirichletBC(W.sub(0).sub(1), Constant(0.0), Bottom() ) )

# REMARKS
#boundaries = MeshFunction("size_t", mesh, "rectangle2_facet_region.xml")

class PeriodicBoundary(SubDomain):

	# Left boundary is "target domain" W
	def inside(self, x, on_boundary):
		return abs(x[0]) < DOLFIN_EPS and on_boundary

	# Map right boundary (E) to left boundary (W)
	def map(self, x, y):
		y[0] = x[0] - length
		y[1] = x[1]

#bcs.append( DirichletBC(W.sub(0).sub(1), v0, boundaries, 1) )
#bcs.append( DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 3) )


########################
### forms definitions ##
########################

dx = Measure("dx", domain=mesh) # Charles

F0 = -p*div(vt)*dx +  pt*div(v)*dx

#balance plus traction 
F1 = R1_*inner( (v-vp)/dt,vt)*dx + inner(s, grad(vt))*dx #- inner(g, vt)*ds(1)

F2 = inner(nu1 - signe(  inner(s ,SM1) )*( inner(s ,SM1)/tauc1p )**10,nu1t)*dx + inner(nu2 - signe(  inner(s ,SM2) )*( inner(s ,SM2)/tauc2p )**10,nu2t)*dx 

De = sym(grad(v)) - R2_*( nu1*sym(SM1) + nu2*sym(SM2) )
We = skew(grad(v)) - R2_*( nu1*skew(SM1) + nu2*skew(SM2) )

F3 = inner((s- sp)/dt - We*s + s*We,st)*dx 
F4 = -R3_*lmbda_*tr(De)*tr(st)*dx - 2*R3_*mu_*inner(De,st)*dx 

F5 = inner(alpha - alphap ,alphat)*dx - R2_*0.5*dt*inner(  nu1 + nu2 - rot(v) ,alphat)*dx

#Remark Nitsche method
#F6 =  -(inner(s*n,n)*inner(vt,n) + inner(st*n,n)*inner(v,n) )*ds(1) + (beta*(1/h)*inner(v,n)*inner(vt,n))*ds(1)

#POWER LAW 

F = F0 + F1 + F2 + F3 + F4 + F5 
 
dw = TrialFunction(W)
J = derivative(F, w, dw)

''' SOLVER '''
print 'problem'
problem = NonlinearVariationalProblem(F, w, bcs=bcs, J=J, form_compiler_parameters=ffc_options)
print 'solver'
solver = NonlinearVariationalSolver(problem)

#info(solver.parameters, True)
#exit()


solver.parameters['newton_solver']['absolute_tolerance'] = 1e-3 #1e-7
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6 #1e-10
solver.parameters['newton_solver']['maximum_iterations'] = 200
solver.parameters['newton_solver']['relaxation_parameter'] = 0.1 #0.5
solver.parameters['newton_solver']['report'] = True
#solver.parameters['linear_solver'] = 'lu'
solver.parameters['newton_solver']['linear_solver'] = 'lu' # Charles
#solver.parameters['preconditioner']= 'none'
#solver.parameters['preconditioner']['structure'] = 'none' # Charles



#Remark
#problem = Minak(J, F, bcs, dimensions=2)
#solver = NewtonSolver()



''' Create files for storing solution '''
vfile = File("%s/velocity.pvd" % label)
ufile = File("%s/displacement.pvd" % label)
s11file = File("%s/stress11.pvd" % label)
s12file = File("%s/stress12.pvd" % label)
s22file = File("%s/stress22.pvd" % label)
alphafile = File("%s/alpha.pvd" % label)
pfile = File("%s/pressure.pvd" % label)

nu1file = File("%s/nu1.pvd" %label)
nu2file = File("%s/nu2.pvd" %label)
nu3file = File("%s/nu3.pvd" %label)
tauc1file = File("%s/tauc1.pvd" %label)
tauc2file = File("%s/tauc2.pvd" %label)
tauc3file = File("%s/tauc3.pvd" %label)
tau1file = File("%s/tau1.pvd" %label)
tau2file = File("%s/tau2.pvd" %label)
tau3file = File("%s/tau3.pvd" %label)

s1file = File("%s/s1.pvd" %label)
m1file = File("%s/m1.pvd" %label)
s2file = File("%s/s2.pvd" %label)
m2file = File("%s/m2.pvd" %label)
s3file = File("%s/s3.pvd" %label)
m3file = File("%s/m3.pvd" %label)

hardfile = File("%s/hard.pvd" %label)
nutotfile = File("%s/nutot.pvd" %label)
Dnormfile = File("%s/Dnorm.pvd" %label)

divvfile = File("%s/divv.pvd" %label)

SD = VectorFunctionSpace(mesh, "DG",1)
step = 0
nsteps = 100 #10000


uv = Function(V)
file_ = open('./%s%s.txt' % (label,n),'w')
file_.write("%s,\t %s,\t %s,\t %s,\t %s\n" % ('Step', 'Top', 'Bottom', 'Height', 'Arena'))


#Time series and restoring a solution
#sol = TimeSeries("%s/Solution" %label)
#sol.parameters["clear_on_write"] = False

while step < nsteps:
	print "Step: " + str(step)

	try:	
		solver.solve()
	except RuntimeError: 
		print 'diverge'
		print(RuntimeError)
		quit()
	
	(v,p,s11,s12,s22,nu1,nu2,alpha) = w.split()


	#update varaibles
	'''
	vp.assign(v)
	s11p.assign(s11)
	s12p.assign(s12)
	s22p.assign(s22)
	alphap.assign(alpha)
    '''

	assign(vp,v)
	assign(s11p,s11)
	assign(s12p,s12)
	assign(s22p,s22)
	assign(alphap,alpha)
        
	#mesh update
	uv.vector()[:] = array( dt*vp.vector().array() )
	
	#mesh.move(uv)
	ALE.move(mesh, uv)
    #plot(mesh)
	
	mesh_height = max(mesh.coordinates()[:,1])
	
	ltop = abs( max([a for a, b in zip(mesh.coordinates()[:,0],mesh.coordinates()[:,1]) if abs(b - mesh_height) < eps  ]) - min([a for a, b in zip(mesh.coordinates()[:,0],mesh.coordinates()[:,1]) if abs(b - mesh_height) < eps  ]) )
	lbottom = abs( max([a for a, b in zip(mesh.coordinates()[:,0],mesh.coordinates()[:,1]) if abs(b) < eps  ]) - min([a for a, b in zip(mesh.coordinates()[:,0],mesh.coordinates()[:,1]) if abs(b) < eps  ]) )
	
    #arena = assemble(Constant(1.0)*dx,mesh=mesh)
	arena = assemble(Constant(1.0)*dx)
    
	file_.write("%s,\t %s,\t %s,\t %s,\t %s\n" % (step, ltop, lbottom, mesh_height, arena))	

	#renameing, for the sake of paraview
	v.rename("v","velocity") 
	s11.rename("S11", "stress")
	s12.rename("S12", "stress")
	s22.rename("S22", "stress")
	alpha.rename("alpha","alpha")
	p.rename("pressure","pressure")
	nu1.rename("nu1","nu1")
	nu2.rename("nu2","nu2")

	#save solutions
	vfile << v
	s11file << s11
	s12file << s12
	s22file << s22	
	alphafile << alpha
	pfile << p
	nu1file << nu1
	nu2file << nu2

	s1file << project(s1,VectorFunctionSpace(mesh, "CG", 1) )
	m1file << project(m1,VectorFunctionSpace(mesh, "CG", 1) )
	s2file << project(s2,VectorFunctionSpace(mesh, "CG", 1) )
	m2file << project(m2,VectorFunctionSpace(mesh, "CG", 1) )


	'''hardening update'''
	#accmulated slip
	inu1 = interpolate(nu1,NU)
	nu1int.vector()[:] = array( map( abs, dt*inu1.vector().array() ) ) 

	inu2 = interpolate(nu2,NU)
	nu2int.vector()[:] = array( map( abs, dt*inu2.vector().array() ) ) 

	nutot.vector()[:] += nu1int.vector()[:] + nu2int.vector()[:] 
	nutot.rename("nuint","nuint")

	nutotfile << nutot

	hard.vector()[:] = array( map(fhardening, nutot.vector().array() ) )
	hardfile << hard

	tauc1 = project( tauc1p + dt*hard*(abs(nu1)+1.4*abs(nu2)), TAU)
	tauc2 = project( tauc2p + dt*hard*(1.4*abs(nu1)+abs(nu2) ), TAU) 

	assign(tauc1p,tauc1)
	assign(tauc2p,tauc2)

	#tauc1file << tauc1
	#tauc2file << tauc2
	#tauc3file << tauc3
	#tau1file << project(inner(s ,SM1), TAU)
	#tau2file << project(inner(s ,SM2), TAU)
	#tau3file << project(inner(s ,SM3), TAU)

	'''aditional variables'''
	Dnorm = project( inner(sym(grad(v)),sym(grad(v))), NU )
	Dnormfile << Dnorm	

	divvfile << project( div(v), NU)

	#D = project( inner(sym(grad(v)),sym(grad(v))), NU )
	#Dfile << D
	
	"timeSeries"
	#sol.store(w.vector(),step)
	
	step += 1
	
print 'end'
