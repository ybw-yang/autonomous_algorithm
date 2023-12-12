import math
import numpy as np

# TODO: implement QuinticPolynomial Solver here
class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, initial_T):
        self.a_ = []
        self.a_.append(xs)
        self.a_.append(vxs)
        self.a_.append(axs/2)
        self.a_.append(10*(xe-self.a_[0])*initial_T**-3 - (6*self.a_[1]+4*vxe)*initial_T**-2 + (1/2*axe-3*self.a_[2])/initial_T)
        self.a_.append(15*(self.a_[0]-xe)*initial_T**-4 + (8*self.a_[1]+7*vxe)*initial_T**-3 + (3*self.a_[2]-axe)*initial_T**-2)
        self.a_.append(6*(xe-self.a_[0])*initial_T**-5 - (3*self.a_[1]+3*vxe)*initial_T**-4 + (1/2*axe-self.a_[2])*initial_T**-3)

    def eval_x(self, t):
        xt = self.a_[0]*t**0 + self.a_[1]*t**1 + self.a_[2]*t**2 + self.a_[3]*t**3 + self.a_[4]*t**4 + self.a_[5]*t**5
        return xt

    def eval_dx(self, t):
        dxt = self.a_[1] + 2*self.a_[2]*t**1 + 3*self.a_[3]*t**2 + 4*self.a_[4]*t**3 + 5*self.a_[5]*t**4
        return dxt

    def eval_ddx(self, t):
        ddxt = 2*self.a_[2] + 6*self.a_[3]*t**1 + 12*self.a_[4]*t**2 + 20*self.a_[5]*t**3
        return ddxt

    def eval_dddx(self, t):
        dddxt = 6*self.a_[3] + 24*self.a_[4]*t**1 + 60*self.a_[5]*t**2
        return dddxt

