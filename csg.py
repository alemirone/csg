import numpy as np
import numpy
import matplotlib.pyplot as plt
from math import sqrt
##from array import array
import matplotlib
matplotlib.use('Qt4Agg')
import pylab 

### Conjugate Sub Gradient LASSO minimisation
##   as described in  http://arxiv.org/abs/1506.07730

class Prob:

    '''
    A class describing a linear inverse problem b = A*x,
    and some routines used by optimization algorithms minimizing
         ||A x - b||^2 + beta*||x||_1
    The matrix A is built to be ill-conditioned : A = vals*vects
    
    Attributes
    ----------
    
    NVAR: integer
        Second dimension of matrix "A" (eg. 1000).
    NSPACE: integer
        Dimension of the observation vector "b" (eg. 500), and first dimension of matrix "A".
    L: float
        Lipschitz constant of the data fidelity term (i.e largest singular value of A)
    b: numpy array
        Observation vector
    beta: float
        the weight of L1 regularization parameters
    vects : numpy array
        Vectors characterizing matrix "A"
    vals: numpy array
        Values characterizing matric "A"
    
    Methods
    --------
    
    error_Qgradient: computes the quadratic part of the gradient
    error_gradient_Q: computes the whole gradient
    forth_back: computes q(p) = A^T A p
    line_search: Line search to determine the gradient step in conjugate gradient
    '''

    def __init__(self, beta=None, alpha=None):
        '''
        Creates a  matrix "A" with tunable condition number,
        and an observation vector "b".
        
        Parameters
        ----------
    
        param beta: float
            the weight of L1 regularization parameters
        param alpha: float
            parameter tuning the condition number of the matrix A
        '''
        self.NVAR = 1000
        self.NSPACE = 500
        vects = np.zeros([self.NVAR, self.NSPACE], "d")
        numpy.random.seed(0) #FOR DEBUG
        for i in range(self.NVAR):
            vects[i] = np.random.random(self.NSPACE)
            vects[i] /= np.sqrt((vects[i] * vects[i]).sum())
            vects[i] -= vects[i].mean()
        vals = np.zeros(self.NSPACE, "d")
        eig = 1.0
        for i in range(self.NSPACE):
            vals[i] = eig
            eig *= alpha
        self.vals = np.sqrt(vals)
        vals2 = vals
        
        A = np.dot(vals2 * vects, vects.T)
        evals2 = np.linalg.eigvals(A).real
        

        self.A =  (self.vals* vects ) .T

        plt.figure()
        plt.plot((np.arange(len(evals2)) + 1), np.log(abs(evals2)))
        plt.title("Singular values of matrix A")
        plt.show()
        
        self.vects = vects
        self.L = evals2.max()
        
        xrandom = np.random.random(self.NVAR)
        xrandom -= xrandom.mean()
        self.b = self.vals * np.dot(self.vects.T, xrandom)
        self.beta = beta

    def error_gradient_Q(self, x, M=None):
        '''
        calculate the (opposite of the) whole gradient : quadratic part + beta*sign(x)
        '''
        if M is not None:
            x = x * M
        error = -self.b + self.vals * np.dot(self.vects.T, x)
        err = (error*error).sum()/2.0 + self.beta * (np.abs(x)).sum()
        Qgrad = np.dot(self.vects, self.vals*error)  # A^T (A x - b)
        grad = Qgrad + self.beta * (np.sign(x))         # A^T (A x - b) + beta*sign(x)
        if M is not None:
            grad = grad * M
            Qgrad = Qgrad * M
        return err, grad, Qgrad

    def error_Qgradient(self, x, M=None):
        '''
        returns only the (opposite of the) quadratic part of the gradient
        '''
        if M is not None:
            x = x * M
        error = -self.b + self.vals * np.dot(self.vects.T, x)
        err = (error * error).sum()/2.0 + self.beta * (np.abs(x)).sum()
        grad = np.dot(self.vects, self.vals * error)
        if M is not None:
            grad = grad * M
        return err, grad

    def forth_back(self, p, M=None):
        '''
        Computes q = A^T * A * p
        '''
        if M is not None:
            p = p * M
        error = self.vals * np.dot(self.vects.T, p)
        grad = np.dot(self.vects, self.vals * error)
        if M is not None:
            grad = grad * M
        return grad

    def line_search(self, grad, Qgrad, p, K, x, M=None):
        '''
        Line search to determine the gradient step,
        using secant method and interval bissection.
        '''
        if M is None:
            M = 1
        speed0 = -(grad*p).sum()
        speed0_q = -(Qgrad*p).sum()
        d_0 = -speed0  # derivee le long de p initial
        dQ_0 = -speed0_q   # partie quadratique de cette derivee

        step = 1.2 * speed0/K
        d = dQ_0 + K * step + self.beta * (np.sign((x + step * p) * M) * M * p).sum()
        al = 0.0
        ah = step
        fp_l = d_0
        fp_h = d

        alphaold = -1
        alpha = (al * fp_h - ah * fp_l)/(fp_h - fp_l)  # secant method
        
        for iter in range(4000):
            if ((abs(ah) + abs(al)) == 0 or abs((ah - al)/(abs(ah) + abs(al))) < 1.2e-7):
                break
            if (abs((alpha + alphaold) == 0 or (alpha - alphaold)/(alpha + alphaold)) < 1.2e-7):
                break
            dQ = dQ_0 + alpha * K
            d = dQ + self.beta * (np.sign((x + alpha * p) * M) * M * p).sum()

            if (abs(d) > min(abs(fp_l), abs(fp_h))/2.0):
                alphadev = (al + ah)/2.0
                dQ_dev = dQ_0 + alphadev * K
                d_dev = dQ_dev + self.beta*(np.sign((x + alphadev * p) * M) * M * p).sum()
                if(abs(d) > abs(d_dev)):
                    alpha = alphadev
                    d = d_dev
            alphaold = alpha
            if (d < 0):
                al = alpha
                fp_l = d
            else:
                ah = alpha
                fp_h = d
            alpha = (al * fp_h - ah * fp_l)/(fp_h - fp_l)
        else:
            print("[LS] reaching the end")
            return alpha, False
        return alpha, True


def Fista(prob=None, nsteps=None, y=None, restart_if_increase=False):
    '''
    Nesterov algorithm (FISTA)
    '''
    xold = y
    errs = []
    spa = []
    told = 1.0
    for step in range(nsteps):
        err, grad = prob.error_Qgradient(y)
        errs.append(err)
        p = -grad
        L = prob.L
        alpha = 1.0/L
        x = y + alpha * p
        x = np.maximum((np.abs(x) - prob.beta/L), 0) * np.sign(x)
        t = (1 + sqrt(4 * told * told + 1)) * 0.5
        y = x + (x - xold) * (told - 1)/t

        if (restart_if_increase):
            errY, gradY = prob.error_Qgradient(y)
            if errY > err:
                y = x
                t = 1
                told = t
            else:
                told = t
                xold = x
        else:
            told = t
            xold = x

        spa.append(np.abs(np.sign(x)).sum())
    return y, errs, spa
    
    
def CSG(prob=None, nsteps=None, x=None, shrink=0.9, increase=0.02):
    '''
    Conjugate subgradient algorithm
    
    Parameters
    ----------
    
    nsteps: integer
        Number of iterations
    x: numpy array
        Initial guess
    shrink: float
        parameter tuning the decrease of preconditioner values when the quadratic gradient is "small"
    increase: float
        parameter tuning the increase of preconditioner values when the quadratic gradient is "big"
    '''
    errs = []
    spa = []
    mult = np.ones_like(x)  # preconditioner
    err, grad, Qgrad = prob.error_gradient_Q(x, mult)  # initial gradient
    p = - grad  # initial conjugate direction
    
    for step in range(nsteps):
        dp = prob.forth_back(p, mult)  # A^T*A*p
        K = np.sum(dp * p)  # parabola slope
        alpha, ok = prob.line_search(grad, Qgrad, p, K, x, mult)

        # update x and "residual"
        Qgrad = Qgrad + dp * alpha
        newx = x + p * alpha

        # UPDATE PRECONDITIONER
        # -----------
        oldmult = mult
        nonsmall = np.less(mult, 1.0)
        # if |Qgrad|<beta*preconditioner  AND  sign change in (x_old -> x) : consider as small value
        small = np.less(np.abs(Qgrad), prob.beta * mult) * np.less(x * newx, 0)
        nonsmall = nonsmall * (1 - small)
        # update preconditioner : (1 - shrink*small) + (increase*non_small)
        mult = mult * (1 - shrink * small + nonsmall * increase)
        mult = np.minimum(mult, 1)
        # if |Qgrad| < beta*precond  AND  |x| < epsilon
        sticker = 1 - np.less(np.abs(Qgrad), prob.beta * mult) * (np.less(np.abs(newx), 1.0e-20))
        mvar = mult/oldmult
        
        # update preconditioned variables
        x = newx/mvar * sticker
        pold = p * mvar * sticker
        dpold = dp * mvar * sticker
        
        # re-calculate gradient
        err, grad, Qgrad = prob.error_gradient_Q(x, mult)
        grad = grad * sticker
        Qgrad = Qgrad * sticker
        errs.append(err)
        spa.append(np.abs(np.sign(x)).sum())

        beta_cg = -(dpold * (-grad)).sum()/(dpold * pold).sum()
        if(beta_cg < 0):
            beta_cg = 0
            
        # update conjugate direction
        p = beta_cg * pold + (-grad)

    return x, errs, spa



class PsiClass:
    def __init__(self,  beta  ) :
        self.beta = beta
        self.coef_x  = 0.0
        self.coef_b  = 0.0

    def argmin(self,z, L):
        ## z = self.z_omega - self.coef_x/self.L
        z = z - self.coef_x/L
        z =numpy.maximum( (numpy.abs(z)-self.coef_b*self.beta/L),0)*numpy.sign(z)        
        return z
        
    def update(self, Dcoef_x = 0, Dcoef_b = 0 ) :
        self.coef_x = self.coef_x + Dcoef_x
        self.coef_b = self.coef_b + Dcoef_b


def ADMM(prob=None, nsteps=None, y =None, rho= None):
    x  = numpy.array( y  ) 
    z  = numpy.array( y  ) 
    u  = numpy.zeros_like( x  ) 
    A = prob.A
    AtAp = (numpy.dot( A.T,A))
    for i in range(len(x)):
        AtAp[i,i] = AtAp[i,i] + rho
    invAtAp =  numpy.linalg.inv( AtAp )
    print A.shape
    print prob.b.shape
    Atb = numpy.dot(A.T, prob.b) 

    errs=[]
    spa=[]

    for step in range(nsteps ):
        x = numpy.dot( invAtAp, Atb+rho*(z-u)     ) 
        z = x+u
        z = numpy.maximum( (numpy.abs(z)-prob.beta/rho),0)*numpy.sign(z)
        u = u+x-z

        err, grad = prob.error_Qgradient(x)
        print err
        spa.append( (numpy.abs(numpy.sign(x) )).sum()   )
        errs.append( err) 

    return x, errs, spa



def Nesterov(prob=None, nsteps=None, y =None ) : 
    errs=[]
    spa=[]

    z_omega = numpy.zeros_like( y  ) 
    L = prob.L
    psi = PsiClass(  prob.beta)
    z = z_omega
    for step in range(nsteps ):
        
        z =  psi.argmin(z_omega, L)
        tau = (2.0*(step+2.0))/((step+1.0)*(step+4.0)) 
        x = tau*z+(1-tau)*y        
        err, grad = prob.error_Qgradient(x)
        gradold = grad
        ll =  2*L/(step+2.0)
        hx =  z - grad/( ll ) 
        hx = numpy.maximum( (numpy.abs(hx)-prob.beta/ll),0)*numpy.sign(hx)
        y  =tau *hx   +  ( 1 - tau)*y
        psi.update(Dcoef_x = + grad*(step+2.0)/2.0  ,  Dcoef_b =  (step+2.0)/2.0 )
        errs.append(err)
        print err
        spa.append( (numpy.abs(numpy.sign(x) )).sum()   )

 
    return y, errs, spa


    

if __name__ == '__main__':
    
    # Create an instance of the problem A*x = b
        prob = Prob(beta=0.1, alpha=1.01)
	y=numpy.zeros(prob.NVAR,"d")
        method = "Admm"

        y, errsF, spaF  =Fista    (prob=prob, nsteps=2000, y = numpy.zeros(prob.NVAR,"d"),  restart_if_increase=True)    
        
        y, errsN, spaN  =Nesterov (prob=prob, nsteps=2000, y = numpy.zeros(prob.NVAR,"d") )    

        y, errsA, spaA  = ADMM (prob=prob, nsteps=2000, y = numpy.zeros(prob.NVAR,"d"), rho= prob.L/350 )    
        
        

	x=numpy.zeros(prob.NVAR,"d")
        x, errsC, spaC=CSG(prob=prob, nsteps=2000, x = numpy.zeros(prob.NVAR,"d"), shrink=0.85, increase=0.04  )  
        ## x, errsC, spaC=Fista    (prob=prob, nsteps=2000, y = numpy.zeros(prob.NVAR,"d"),  restart_if_increase=True)    
        
	limite = numpy.array(errsF).min()
        
	limite = min(limite,numpy.array(errsC).min())

        print " MIN ", numpy.array(errsC).min(), limite
        fig= pylab.figure()
        ax = pylab.plt.subplot(111)
        ax.set_yscale("log", nonposy='clip')


	tmp = errsN-limite
	tmp[tmp < 1e-14] = 1e-14
	fgn, = pylab.plot((numpy.arange(len(errsN))+1), (tmp),
                   'k:',label=r'FGN',linewidth=2.5)


	fista, = pylab.plot((numpy.arange(len(errsF))+1), (errsF-limite),
                   'k--',label=r'FISTA',linewidth=4)


	tmp = errsA-limite
	tmp[tmp < 1e-14] = 1e-14
	admm, = pylab.plot((numpy.arange(len(errsA))+1), (tmp),
                   'k-.',label=r'ADMM',linewidth=2.5)



	tmp = errsC-limite
	tmp[tmp < 1e-14] = 1e-14
	csg, = pylab.plot((numpy.arange(len(errsC))+1), (tmp),
                   'k-',label=r'CSG',linewidth=2.5)
        print len(errsC)
        ax.set_ylim(ymin=1.0e-15)


	pylab.xlabel('iterations', fontsize=20)
	pylab.ylabel('$F(x)-F(x_\infty)$', fontsize=18)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18) 
        #legend1 = pylab.plt.legend( [fgn, fista] ,loc=1, shadow=True, fontsize=23)
        #legend2 = pylab.plt.legend( [csg, admm] ,loc=10, shadow=True, fontsize=23)
        legend2 = pylab.plt.legend(loc=10, shadow=True, fontsize=23)



	#~ pylab.ylabel(r'\log\left(E-E_{\text{min}}\right)')
        fig.tight_layout()        
 
	pylab.show()
	
