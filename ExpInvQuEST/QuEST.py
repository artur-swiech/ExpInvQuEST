import numpy as np
from scipy.optimize import root, minimize, least_squares, basinhopping
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz

def complex_to_array(x):
    return np.array([x.real, x.imag])

def array_to_complex(x):
    return complex(x[0], x[1])

def edge(eigvals, r, gamma):
    def fun(x):
        M11 = (eigvals / (x - eigvals)).mean()
        M12 = (eigvals / (x - eigvals)**2).mean()
        M22 = (eigvals**2 / (x - eigvals)**2).mean()
        s = np.sqrt(1.0 + r**2 * (gamma**2 - 1.0) * M11**2)
        return - r * gamma * M22 + s - x * r**2 * (gamma**2 - 1) * M11 * M12 / s

    X_min = root(fun, min(eigvals) * 0.99).x
    M11 = (eigvals / (X_min - eigvals)).mean()
    s = np.sqrt(1.0 + r**2 * (gamma**2 - 1.0) * M11**2)
    x_min = X_min[0] * (r * gamma * M11 + s)

    X_max = root(fun, max(eigvals) * 1.01).x
    M11 = (eigvals / (X_max - eigvals)).mean()
    s = np.sqrt(1.0 + r**2 * (gamma**2 - 1.0) * M11**2)
    x_max = X_max[0] * (r * gamma * M11 + s)

    return (x_min, x_max)

def inverse_quest(empirical_eigvals, r, gamma):

    min_k, max_k = edge(empirical_eigvals, r, gamma)
        #TODO: better control points
    k = np.linspace(min_k, max_k, 500)

    def f(eigvals):
        q = QuEST(eigvals, r, gamma, k)
        eig = q.Q()[0]
        return ((eig - empirical_eigvals[:-1])**2).sum()

    def df(eigvals):
        q = QuEST(eigvals, r, gamma, k)
        eig,deig = q.Q()
        return 2.0 * (eig - empirical_eigvals[:-1]).dot(deig)

    minimizer_kwargs = {'jac': df,
                        'bounds': [(min(empirical_eigvals),max(empirical_eigvals))
                                   for _ in range(len(empirical_eigvals))],
                        'method': 'SLSQP'
                       }

    return basinhopping(func=f,
                        x0=empirical_eigvals,
                        niter=5,
                        minimizer_kwargs=minimizer_kwargs,
                        disp=True
                       )

class QuEST:
    def __init__(self, eigvals, r, gamma, k):
        self.eigvals = np.array(eigvals)
        self.N = len(eigvals)
        self.r = r
        self.gamma = gamma
        self.k = k

    def M_from_Z(self, Z):
        return (self.eigvals / (Z - self.eigvals)).sum() / self.N

    def dMdZ_from_Z(self, Z):
        return -(self.eigvals / (Z - self.eigvals)**2).sum() / self.N

    def dMdLambda_from_Z(self, Z):
        return Z / (Z - self.eigvals)**2 / self.N

    def z_from_Z(self, Z):
        M = self.M_from_Z(Z)
        return Z * (self.r * M * self.gamma + np.sqrt((self.r * M)**2
                * (self.gamma**2 - 1.0) + 1.0))

    def find_Z_from_z(self, z):
        return root(
            fun = lambda Z_: complex_to_array(
                z - self.z_from_Z(array_to_complex(Z_))),
            #TODO: Better starting point?
            x0 = np.array([z.real, 1]))

    def Z_from_z(self, z):
        Z = self.find_Z_from_z(z).x
        if Z[1] < 0:
            Z[1] = -Z[1]
        return array_to_complex(Z)

    def dZdLambda_from_Z(self, Z):
        M = self.M_from_Z(Z)
        dMdZ = self.dMdZ_from_Z(Z)
        dMdLambda = self.dMdLambda_from_Z(Z)
        gamma2m1 = self.gamma**2 - 1.0
        a = np.sqrt((self.r * M)**2 * gamma2m1 + 1.0)
        b = self.r**2 * M * gamma2m1
        dzdZ = self.r * M * self.gamma + a + Z * (r * dMdZ * self.gamma +
                (b * dMdZ) / a)
        dzdLambda = Z * (self.r * dMdLambda * self.gamma + (b * dMdLambda) / a)
        return - dzdLambda / dzdZ

    def m_from_z(self, z):
        return self.M_from_Z(self.Z_from_z(z))

    def g_from_z(self, z):
        return (self.m_from_z(z) + 1.0) / z

    def rho_from_x(self, x):
        #TODO: better limit
        z = complex(x,1e-7)
        return abs((-self.g_from_z(z)/np.pi).imag)

    def Q(self):
        dk = np.diff(self.k)

        z = np.array([complex(_k, 1e-10) for _k in self.k])

        Z = np.array([self.Z_from_z(_z) for _z in z])

        MZ = np.array([self.M_from_Z(_Z) for _Z in Z])

        N = np.sqrt(1.0 + self.r**2 * MZ**2 * (self.gamma**2 - 1.0))

        dMdZ = np.array([self.dMdZ_from_Z(_Z) for _Z in Z])
        dMdLambda = np.array([self.dMdLambda_from_Z(_Z) for _Z in Z])

        dzdLambda = Z[:, None] * (self.r * dMdLambda * self.gamma + self.r**2
                * MZ[:, None] * dMdLambda * (self.gamma**2 - 1.0) / N[:, None])
        dzdZ = self.r * MZ * self.gamma + N + Z * (self.r * dMdZ * self.gamma
                + self.r**2 * MZ * dMdZ * (self.gamma**2 - 1.0) / N)

        dZdLambda = -dzdLambda / dzdZ[:, None]

        dmdLambda = dMdZ[:, None] * dZdLambda + dMdLambda

        g = (MZ + 1.0) / z

        dgdLambda = dmdLambda / z[:, None]

        pdf = -g.imag/np.pi

        a = 0.5 * (pdf[:-1] + pdf[1:])
        dpdfdLambda = -dgdLambda.imag/np.pi
        dadLambda = 0.5*(dpdfdLambda[:-1] + dpdfdLambda[1:])

        cdf = cumtrapz(pdf, initial=0.0, dx=dk)
        dcdfda = np.tril(np.tile(dk, (len(a), 1)), -1)
        dcdfdLambda = dcdfda.dot(dadLambda)

        inverseCdf = InterpolatedUnivariateSpline(x=cdf, y=self.k, k=1)
        y = np.linspace(0, 1, self.N + 1)
        x = inverseCdf(y)

        indices = np.searchsorted(cdf, y)

        M = np.minimum(np.maximum(indices[ :-1] - 1, 0), len(a) - 1)
        L = np.maximum(np.minimum(indices[1:  ] - 1, len(a) - 1), 0)

        am = a[M]
        dadLambdam = dadLambda[M]

        al = a[L]
        dadLambdal = dadLambda[L]

        cm = cdf[M]
        dcdfdLambdam = dcdfdLambda[M]

        cl = cdf[L]
        dcdfdLambdal = dcdfdLambda[L]

        kmp1 = self.k[M+1]
        kl = self.k[L]

        k2diff = self.k[1:]**2 - self.k[:-1]**2

        x2 = x**2
        x2diff = x2[1:] - x2[:-1]

        xi = x[:-1]
        xip1 = x[1:]
        yi = y[:-1]
        yip1 = y[1:]

        ak2diff = np.array([((a * k2diff)[m+1:l]).sum() for m,l in zip(M,L)])
        dadLambdak2diff = 0.5 * np.array([((dadLambda[m+1:l]).T *
                k2diff[m+1:l]).sum(axis=1) for m,l in zip(M,L)])
        amdxidLambda = ((cm - yi) / am * dadLambdam.T).T - dcdfdLambdam
        aldxip1dLambda = ((cl - yip1) / al * dadLambdal.T).T - dcdfdLambdal

        Q = 0.5 * np.where(M==L, am * x2diff,
                am * (kmp1**2 - xi**2) + ak2diff + al * (xip1**2 - kl**2))
        dQdLambda = np.where(M==L,
                -(amdxidLambda.T * xi) + 0.5 * (dadLambdam.T * x2diff)
                + aldxip1dLambda.T * xip1,
                0.5 * (dadLambdam.T * (kmp1**2 - xi**2)) - amdxidLambda.T * xi +
                dadLambdak2diff.T + 0.5 * (dadLambdal.T * (xip1**2 - kl**2))
                + aldxip1dLambda.T * xip1).T

        Q *= self.N
        dQdLambda *= self.N
        #TODO: Last element sometimes computed incorrectly, investigate
        return Q[:-1],dQdLambda[:-1]
